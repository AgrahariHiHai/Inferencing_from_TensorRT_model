import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pydicom
from monai import inferers

class TensorRTInference:
    """
    Class to handle TensorRT inference with proper context and resource management
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
       
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
           
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        # Create execution context once
        self.context = self.engine.create_execution_context()
       
        # Initialize stream
        self.stream = cuda.Stream()
       
        # Allocate buffers once during initialization
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
       
    def _allocate_buffers(self):
        """
        Allocate buffers for inputs and outputs.
        """
        inputs = []
        outputs = []
        bindings = []
       
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
           
            # Calculate size
            size = trt.volume(tensor_shape)
           
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, tensor_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
           
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': tensor_dtype
                })
            else:
                outputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': tensor_dtype
                })
               
        return inputs, outputs, bindings
   
    def preprocess_input(self, input_tensor):
        """
        Preprocess input tensor for TensorRT inference.
        """
        # Ensure input is on CPU and correct dtype
        input_np = input_tensor.cpu().numpy()
       
        # Validate input shape
        expected_shape = self.inputs[0]['shape']
        if input_np.shape != expected_shape:
            raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {input_np.shape}")
       
        # Convert to correct dtype if necessary
        input_np = input_np.astype(self.inputs[0]['dtype'])
       
        return input_np
   
    def run_inference(self, input_tensor):
        """
        Run inference with proper error checking and resource management.
        """
        try:
            # Preprocess input
            input_np = self.preprocess_input(input_tensor)
           
            # Copy input to host buffer
            np.copyto(self.inputs[0]['host'], input_np.ravel())
           
            # Transfer to GPU
            cuda.memcpy_htod_async(self.inputs[0]['device'],
                                 self.inputs[0]['host'],
                                 self.stream)
           
            # Run inference
            self.context.execute_async_v2(bindings=self.bindings,
                                        stream_handle=self.stream.handle)
           
            # Transfer back to host
            cuda.memcpy_dtoh_async(self.outputs[0]['host'],
                                 self.outputs[0]['device'],
                                 self.stream)
           
            # Synchronize
            self.stream.synchronize()
           
            # Process output
            output = np.reshape(self.outputs[0]['host'],
                              self.outputs[0]['shape'])
           
            # Convert to torch tensor
            output_tensor = torch.from_numpy(output).to(self.device)
           
            return output_tensor
           
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            raise

def process_dicom(dicom_path):
    """
    Process DICOM image with validation.
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
       
        # Validate pixel array
        if pixel_array.ndim != 2:
            raise ValueError(f"Expected 2D image, got {pixel_array.ndim}D")
       
        # Convert to float32 and normalize if needed
        pixel_array = pixel_array.astype(np.float32)
        if pixel_array.max() > 1.0:
            pixel_array = pixel_array / pixel_array.max()
       
        # Convert to tensor
        tensor = torch.from_numpy(pixel_array)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
       
        return tensor
       
    except Exception as e:
        print(f"DICOM processing failed: {str(e)}")
        raise

def get_sliding_window_inference(dicom_path, trt_inference, roi_size=(64, 64),
                               overlap=0.5):
    """
    Perform sliding window inference with improved error handling.
   
    Args:
        dicom_path (str): Path to DICOM file
        trt_inference (TensorRTInference): TensorRT inference instance
        roi_size (tuple): Size of sliding window
        overlap (float): Overlap between windows
    """
    try:
        # Load and process DICOM
        image_tensor = process_dicom(dicom_path)
        image_tensor = image_tensor.to(trt_inference.device)
       
        print(f"Input image shape: {image_tensor.shape}")
       
        # Define inference function
        def inference_func(window):
            # Ensure window is in correct format
            if window.shape[-2:] != roi_size:
                raise ValueError(f"Window shape mismatch. Expected {roi_size}, got {window.shape[-2:]}")
           
            return trt_inference.run_inference(window)
       
        # Run sliding window inference
        output = inferers.sliding_window_inference(
            inputs=image_tensor,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=inference_func,
            overlap=overlap,
            mode="gaussian",
            device=trt_inference.device
        )
       
        print(f"Output shape: {output.shape}")
       
        return output
       
    except Exception as e:
        print(f"Sliding window inference failed: {str(e)}")
        raise

# Usage example
def main():
    try:
        # Initialize TensorRT inference
        engine_path = "path/to/your/engine.trt"
        trt_inference = TensorRTInference(engine_path)
       
        # Process DICOM
        dicom_path = "path/to/your/dicom.dcm"
        output = get_sliding_window_inference(dicom_path, trt_inference)
       
        # Post-process output if needed
        segmentation_mask = torch.argmax(output, dim=1)
       
        return segmentation_mask
       
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise
