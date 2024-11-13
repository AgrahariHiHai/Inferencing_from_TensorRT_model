import torch
import pandas as pd
import os
import glob
# from training_pipeline_2d import *
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from monai.config import KeysCollection
from monai.transforms import (ShiftIntensityd,RandFlipd, RandRotate90d, RandKSpaceSpikeNoised, 
                              ScaleIntensityd,RandCropByPosNegLabeld, CropForegroundd, ToTensord, SqueezeDimd,KeepLargestConnectedComponent)
from monai.networks.nets import UNETR
from monai.losses import FocalLoss
import json
import glob
from monai.data.utils import decollate_batch, default_collate
from monai.networks.nets import UNETR
import torch
import tqdm
from pathlib import Path
import pydicom
import numpy as np
import torch
import monai
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
import torch
from monai.data.utils import decollate_batch
import os
import numpy as np
import torch.nn.functional as F
import monai.inferers as inferers

from monai.config import print_config
from monai.utils import first, set_determinism
import torch.nn as nn
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    CropForegroundd,
    GaussianSmoothd,
    ScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    EnsureTyped,
    SpatialPadd,
    RandSpatialCropSamplesd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.data import Dataset, CacheDataset, DataLoader, load_decathlon_datalist
from monai.losses import FocalLoss, DiceFocalLoss, TverskyLoss

from monai.metrics import DiceMetric
from monai.transforms import EnsureType, AsDiscrete, Activationsd, AsDiscreted, Activations

from monai.transforms import EnsureType, AsDiscrete

import random
from monai.utils import first, progress_bar


# Initialization of engine and buffers outside inference loop
def load_trt_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    #context = engine.create_execution_context()
    return engine#, context#, inputs, outputs, bindings, stream

class CustomRead:
    def __init__(self, keys: KeysCollection,**kwargs):
        self.keys= keys
        self._dataReader = LoadImaged(keys=keys,image_only=False, **kwargs)
        print(**kwargs)
        self._channelFirst = EnsureChannelFirstd(keys=keys)
        
    def __call__(self,data):
        read_data = self._dataReader(data)
        channeled_data = self._channelFirst(read_data)
        spacing_info = channeled_data["image_meta_dict"]["spacing"]
        spacing_info = list(spacing_info)
        #print(spacing_info)
        spacer = Spacingd(keys=self.keys,pixdim=spacing_info, mode=("bilinear", "nearest"))
        spaced_data = spacer(channeled_data)
        return spaced_data

def list_data_collate(batch):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    
    elem = batch[0]
    all_key_data = {}
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    for i, dd in enumerate(data):
        all_key_data.update(dd['image_meta_dict'])
        dd['image_meta_dict'].update(all_key_data)
        #dd[1]['image_meta_dict'].update(all_key_data)
        all_key_data = dd['image_meta_dict']
    return default_collate(data)



class MONAI_Evaluate_Model():
    def __init__(self):
        pass
    def save_predicted_image(self,groundtruth, predicted, curr_ep, curr_dice_score, save_images_dir, image_name):
        
            save_images_path = f"{save_images_dir}"

            if not os.path.isdir(save_images_path):
                os.makedirs(save_images_path)

            plt.figure("predictions", (18, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"groundtruth : {os.path.basename(save_images_path)}")
            plt.axis('off')
            plt.imshow(groundtruth["image"][0, 0, :, :], cmap="gray")
            plt.imshow(groundtruth["label"][0, 0, :, :], alpha=0.3)
            plt.subplot(1, 2, 2)
            plt.title(f"output")
            plt.axis('off')
            plt.imshow(groundtruth["image"][0, 0, :, :], cmap="gray")
            plt.imshow(torch.argmax(predicted, dim=1).detach().cpu()[0, :, :],  alpha=0.3)
            plt.savefig(f"{save_images_path}/{image_name}_{(curr_ep+1)}_{round(curr_dice_score,2)}.png")

    def evaluate(self,num_classes, spatial_dims, engine_path, loader, curr_epoch, save_images_dir,
                 loss_function, dice_metric,dice_metric_batch, device, post_pred, post_label, 
                 slice_index, time_based=False,  post_pred_transfoms = None):
        engine = load_trt_engine(engine_path)
        val_loss = 0.0
        val_step = 0
        total_latency = 0
        # eval_model.eval()
        dice_metric_labels = []
        #with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader)):
            context = engine.create_execution_context()
            inputs, outputs, bindings, stream = allocate_buffers(engine, context, input_shape=(1, 1, 64, 64), batch_size=1)
            pred_outputs = {}
            val_step += 1
            eval_inputs, eval_labels = (
                data["image"].to(device),
                data["label"].to(device),
            )

            meta_dict = data['image_meta_dict']
            slice_name = os.path.basename(meta_dict['filename_or_obj'][0]).split('.')[0]
            directory_name = os.path.basename(os.path.dirname(meta_dict['filename_or_obj'][0]))
            with engine.create_execution_context() as context:
                predicts, latency_for_one = sliding_window_inference_trt(eval_inputs, engine, context, inputs, outputs, bindings, stream, num_classes)
                torch.cuda.synchronize()
            total_latency += latency_for_one 
            pred_output = predicts
            loss = loss_function(predicts, eval_labels)
            val_loss += loss.item()

            predicts = [post_pred(i) for i in decollate_batch(predicts)]
            targets = [post_label(i) for i in decollate_batch(eval_labels)]

            dice_metric(y_pred=predicts, y=targets)
            dice_metric_batch(y_pred=predicts, y=targets)

            curr_dice_score = dice_metric.aggregate().item()
            curr_dice_score_batch = dice_metric_batch.aggregate()

            if ((curr_epoch+1)%2==0):
                save_images_plots_dir = f"{save_images_dir}/{directory_name}"
                self.save_predicted_image(data, pred_output, curr_epoch, curr_dice_score, save_images_plots_dir, slice_name)

            curr_metric_labels = []
            curr_metric = {}

            for i in range(0, num_classes):                
                curr_metric_labels.append(curr_dice_score_batch[i].item())

            if spatial_dims == 2:
                curr_metric[meta_dict['filename_or_obj'][0].split('images/')[1]] = np.mean(curr_metric_labels)
            else:
                curr_metric[os.path.basename(meta_dict['filename_or_obj'][0]).split('.')[0]] = curr_metric_labels                                      


            dice_metric_labels.append(curr_metric)              
            print(f"total latency is:{total_latency}")
        val_loss /= val_step
        metric = dice_metric.aggregate().item()
        metric_batch = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()
        engine.__del__()
        return meta_dict, metric, metric_batch, dice_metric_labels, val_loss , total_latency
    
def get_post_pred_transforms(num_classes):
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes), KeepLargestConnectedComponent()])
        return post_pred
        
def get_post_label_transforms(num_classes):
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])
    return post_label

def get_dice_metric():
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    return dice_metric, dice_metric_batch

# Function to allocate buffers for input and output
def allocate_buffers(engine, context, input_shape, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)

        # Set the input shape for the context (only for input tensors)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(tensor_name, input_shape)

        # Get the shape and calculate the size
        shape = context.get_tensor_shape(tensor_name)
        size = trt.volume(shape) * batch_size  # Use specified batch size
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

    return inputs, outputs, bindings, stream
    
# Function to run inference on the TensorRT engine
def run_inference(engine, inputs, outputs, bindings, stream, input_slice, num_classes, context):
    # print(f"input image shape{input_slice.shape}")
    #context = engine.create_execution_context()
    # Copy input data to host buffer
    np.copyto(inputs[0]['host'], input_slice.cpu().ravel())
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    # Execute the model
    context.execute_v2(bindings=bindings)
    # Transfer predictions from the GPU back to host
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    # Synchronize the stream
    stream.synchronize()
    #Reshaping output
    output_flat = outputs[0]['host']
    output_segmented = np.reshape(output_flat, (1, num_classes, 64, 64))
    # output_segmented = np.argmax(output_segmented, axis =1)
    output_segmented = torch.from_numpy(output_segmented)
    # print(f" output shape is {output_segmented.shape}") 
    return output_segmented.to(device)
    #return outputs[0]['host'].reshape((1, 5, 64, 64))  # Adjust reshape as per output channel needs



def sliding_window_inference_trt(dicom_path, engine, context, inputs, outputs, bindings, stream, num_classes, roi_shape=(64, 64), overlap=0.5):
    # Step 1: Read the DICOM image without changing dimensions
    image_tensor = dicom_path
#     # print(image_tensor.shape)
#     # Step 2: Load the TensorRT engine
#     logger = trt.Logger(trt.Logger.WARNING)
#     with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())
#     # print("TensorRT engine loaded successfully!")

#     # Step 3: Allocate buffers
#     inputs, outputs, bindings, stream = allocate_buffers(engine, engine.create_execution_context(), input_shape=(1, 1, 64, 64), batch_size=1)
#     # print(inputs[0]['host'].shape)
#     # print(outputs[0]['host'].shape)
#     # Step 4: Define inference function to pass 64x64 slices of image
    
    def inference_func(image_slice):
        # Ensure the slice shape is [1, 1, 64, 64]
        # if image_slice.shape != (1, 1, 64, 64):
        #     image_slice = image_slice.unsqueeze(0).unsqueeze(0)
        # print(image_slice.shape)
        return run_inference(engine, inputs, outputs, bindings, stream, image_slice, num_classes, context)

    # Step 5: Perform sliding window inference over the entire image
    start_time = time.time()
    segmented_output = inferers.sliding_window_inference(
        inputs=image_tensor,  # Add batch dimension for sliding window
        roi_size=roi_shape,
        sw_batch_size=1,
        predictor=inference_func
    )
    end_time = time.time()
#     print(f"Sliding Window Inference Time: {end_time - start_time:.6f} seconds")

#     # Ensure the output shape matches the expected segmented output
#     print(f"Inference Output Shape: {segmented_output.shape}")
    latency = end_time - start_time
    return segmented_output , latency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = "csv_out_dir"
save_images_dir =  "save_dir"

if not os.path.isdir(f"{out_dir}"):
    os.makedirs(f"{out_dir}")
    
spatial_dims = 2

#---------Transforms--------------#


test_transforms = Compose(
    [
        CustomRead(keys=["image", "label"]),
        SqueezeDimd(keys=["image", "label"], dim=3),  
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ShiftIntensityd(offset=0.1, keys=["image"]),
        ScaleIntensityd(keys=["image"],minv=0.0, maxv=1.0),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)


#---------Loss--------------#
loss_function = FocalLoss(to_onehot_y = True, use_softmax = True)

#---------Metrics--------------#
dice_metric, dice_metric_batch = get_dice_metric()

evaluate_obj = MONAI_Evaluate_Model()

#------------------------------------------------------------------------------------------------------------------------#

data_root = "/opt/monai/mnt/C4AI/virtual_env/Downstream_Training/CINE_4/Data/Smoothened_DICOM_3Dto2D/DICOM_No_TV" # DICOM directory
json_path = "/opt/monai/mnt/C4AI/virtual_env/Downstream_Training/CINE_4/Datalists/datasets_2d_cine4_tr40_val5_test5.json" # Json path 
model_path = "/opt/monai/mnt/C4AI/virtual_env/Manoj/model_conversion_into_tensorRT_inference_engine/ssl_trt_models/CINE_4_FP16.trt" # TRT model path

class_cine4 = ['background', 'LeftAtrium', 'RightAtrium', 'LeftVentricle','LeftVentricleWalls','RightVentricle'] # Label names

csv_name =  "exp_cine_4_trt_FP16_test_results.csv" # csv name to save the metrics and loss values
save_dir = "exp_cine4_trt_FP16_inference_images_test" # save dir to save the image inference plots

num_classes = len(class_cine4)

post_pred = get_post_pred_transforms(num_classes)
post_label = get_post_label_transforms(num_classes)


df = pd.DataFrame(columns=['model_name', 'model_path','data_root', 'json_file_path', 'labels',
                           'loss_function', 'test_loss', 'test_metric','test_background', 'test_Left atrium', 'test_RightAtrium',
                           'test_LeftVentricle', 'test_LeftVentricleWalls','test_RightVentricle']) # headers for csv file depends on label names

df.to_csv(f"{out_dir}/{csv_name}", mode='a',  header=True, index=False)

with open(json_path, "r") as json_f:
    json_data = json.load(json_f)

test_data = json_data["testing"][0]

test_data["image"] = os.path.join(data_root, test_data["image"])
test_data["label"] = os.path.join(data_root, test_data["label"])

test_list = [
    {"image": img_file, "label": lab_file}
    for img_file, lab_file in zip(
        sorted(glob.glob(test_data["image"] + "/*.dcm")),
        sorted(glob.glob(test_data["label"] + "/*.dcm"))
    )
]

print("Total Number of Test Data Samples: {}".format(len(test_list)))

test_dataset = CacheDataset(data=test_list, transform=test_transforms, cache_rate=1.0, num_workers=4)
test_loader = DataLoader( test_dataset, batch_size=1, num_workers=2, collate_fn = list_data_collate)
test_data = first(test_loader)
print(test_data['image'].shape, test_data['label'].shape)


#---------Evaluate--------------#

test_meta_dict, test_metric, test_metric_batch, test_metric_labels, test_loss, total_lat = evaluate_obj.evaluate(num_classes, spatial_dims, model_path, 
                                                                                                 test_loader, 1, 
                                                                                                 f"{save_dir}",
                                                                                                 loss_function, 
                                                                                                 dice_metric, 
                                                                                                 dice_metric_batch,
                                                                                                 device, post_pred, 
                                                                                                 post_label, 15)

print(f"Total latency:{total_lat}")
print(f"Test Metric : {test_metric}")
print(f"Test Loss : {test_loss}")

#---------To CSV--------------#

df_data = {}
data = []
df_data['model_name'] = os.path.basename(model_path)
df_data['model_path'] = model_path
df_data['data_root'] = data_root
df_data['json_file_path'] = json_path
df_data['labels'] = class_cine4 # change the labels var here 
df_data['loss_function'] = loss_function

df_data['test_loss'] = test_loss
df_data['test_metric'] = test_metric

    
for i in range(0, len(test_metric_batch)):
    df_data[f"test_{class_cine4[i]}"] = test_metric_batch.detach().cpu().numpy().tolist()[i] # here too
    
data.append(df_data)

df = pd.DataFrame(data)
df.to_csv(f"{out_dir}/{csv_name}", mode='a',  header=False, index=False)

# save_metrics_slices(model_dir, model_name, epochs_ran, test_metric_labels, "test")
