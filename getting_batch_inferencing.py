Batch processing and comparison.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom
import torch.nn.functional as F
from typing import List, Tuple
import cv2

# Comment: Previous TensorRTInference class and other helper functions go here

def calculate_focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> float:
    """
    Calculate focal loss between prediction and target mask.
   
    Args:
        pred: Predicted mask (after softmax)
        target: Ground truth mask
        gamma: Focusing parameter
        alpha: Weighting factor
    """
    # Convert to one-hot if needed
    if target.dim() == 3:  # (B, H, W)
        target = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2)
   
    # Calculate focal loss
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_loss = alpha * (1-p_t)**gamma * ce_loss
   
    return focal_loss.mean().item()

def process_batch_inference(
    dicom_paths: List[str],
    gt_paths: List[str],
    trt_inference: 'TensorRTInference',
    output_dir: str,
    roi_size: Tuple[int, int] = (64, 64)
) -> None:
    """
    Process batch of DICOM images and visualize results with ground truth.
   
    Args:
        dicom_paths: List of paths to DICOM images
        gt_paths: List of paths to ground truth masks
        trt_inference: TensorRT inference instance
        output_dir: Directory to save visualization results
        roi_size: Size of sliding window ROI
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
   
    for idx, (dicom_path, gt_path) in enumerate(zip(dicom_paths, gt_paths)):
        try:
            # Load DICOM image
            dicom = pydicom.dcmread(dicom_path)
            original_image = dicom.pixel_array
           
            # Load ground truth
            gt_mask = np.load(gt_path)  # Assuming .npy format, adjust if different
           
            # Run inference
            with timer("Full Slice Inference"):
                image_tensor = process_dicom(dicom_path)  # From previous code
                prediction = get_sliding_window_inference(dicom_path, trt_inference, roi_size)
           
            # Convert prediction to mask
            pred_mask = torch.argmax(prediction, dim=1)[0].cpu().numpy()
           
            # Calculate focal loss
            focal_loss = calculate_focal_loss(
                prediction,
                torch.from_numpy(gt_mask).unsqueeze(0).to(prediction.device)
            )
           
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
           
            # Plot original image
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
           
            # Plot ground truth
            axes[1].imshow(gt_mask, cmap='nipy_spectral')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
           
            # Plot prediction
            im = axes[2].imshow(pred_mask, cmap='nipy_spectral')
            axes[2].set_title(f'Prediction\nFocal Loss: {focal_loss:.4f}')
            axes[2].axis('off')
           
            # Add colorbar
            plt.colorbar(im, ax=axes.ravel().tolist())
           
            # Save figure
            plt.savefig(output_dir / f'slice_{idx:03d}_comparison.png',
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
           
            print(f"Processed slice {idx+1}/{len(dicom_paths)}")
           
        except Exception as e:
            print(f"Error processing slice {idx}: {str(e)}")
            continue

# Example usage
def main():
    # Initialize TensorRT inference (from previous code)
    engine_path = "path/to/engine.trt"
    trt_inference = TensorRTInference(engine_path)
   
    # Prepare paths
    dicom_dir = Path("path/to/dicom/slices")
    gt_dir = Path("path/to/ground/truth/masks")
    output_dir = Path("path/to/output/visualizations")
   
    # Get sorted lists of paths
    dicom_paths = sorted(list(dicom_dir.glob("*.dcm")))
    gt_paths = sorted(list(gt_dir.glob("*.npy")))
   
    # Verify matching pairs
    assert len(dicom_paths) == len(gt_paths), "Mismatch in number of DICOM and ground truth files"
   
    # Process batch
    process_batch_inference(
        dicom_paths=dicom_paths,
        gt_paths=gt_paths,
        trt_inference=trt_inference,
        output_dir=output_dir
    )
   
    print("Batch processing complete!")

if __name__ == "__main__":
    main()
