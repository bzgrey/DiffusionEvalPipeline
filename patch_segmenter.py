# Import required libraries
from segmentation_pipeline import nnUNet, nnUNetConfidence
from lungmask import LMInferer
from segmentation_pipeline import apply_windowing
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def patch_segmenter(patch_tensor: torch.Tensor, segmentation_model: nnUNet, 
                    # lungmask_model: LMInferer, 
                    confidence_model: Optional[nnUNetConfidence]) -> torch.Tensor:
    """
    Segments a 3D medical image using a patch-based approach.

    Parameters:
    - patch_tensor (torch.Tensor): The input 3D patch tensor of shape (B, D, H, W) = (B, Z, Y, X) = (B, 32, 128, 128), TOD0 = work with batches later
    - segmentation_model (nnUNet): The pre-trained segmentation model.
    - lungmask_model (LMInferer): The pre-trained lung mask model.
    - confidence_model (nnUNetConfidence, optional): The pre-trained confidence model.

    Returns:
    - torch.Tensor: The segmented output tensor of shape (B, D, H, W).
    """

    # Set to eval mode
    segmentation_model.eval()
    if confidence_model is not None:
        confidence_model.eval()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert patch tensor to be ready for model input
    # lungmask_input = patch_tensor.numpy()

    # # Run lung mask model
    # # The lungmask model doesn't take batches, so process each patch individually
    # lung_masks = []
    # for i in range(lungmask_input.shape[0]):
    #     single_patch = lungmask_input[i]
    #     single_lung_mask = lungmask_model.apply(single_patch)
    #     lung_masks.append(single_lung_mask)
    # lung_mask = np.stack(lung_masks, axis=0)  # Shape: (B, D, H, W)
    # lung_mask = (lung_mask > 0).astype(np.uint8) # Binarize lung mask from 1s and 2s (for two lungs) to 1s


    # Prepare input tensor for segmentation model
    window_center, window_width = -600, 1600
    patches = apply_windowing(patch_tensor.numpy().astype(np.float64), window_center, window_width)
    # Convert to tensor and normalize
    patches = torch.tensor(patches) // 256
    patches = patches.unsqueeze(1)  # shape: [B, 1, D, H, W], add channel dimension

    # Interpolate to target size if needed
    # image = F.interpolate(
    #     image,
    #     size=(image.shape[2], 1024, 1024),
    #     mode="trilinear",
    #     align_corners=False,
    # )

    # Move to device
    patches = patches.to(device=device).float()
    segmentation_model = segmentation_model.to(device=device)

    # Run segmentation model
    with torch.no_grad():
        segmentation_outputs = segmentation_model.predict(patches).cpu()
    print(f"segmentation outputs shape: {segmentation_outputs.shape}")
    print(f"segmentation outputs example: {segmentation_outputs[0, :, 16, 64, 64]}")
    # Create binary segmentation
    binary_segmentation = (
        1 * (F.softmax(segmentation_outputs, dim=1)[:, 1] > 0.5) 
        # * torch.tensor(lung_mask)
    )
    print(f"binary segmentation example:")

    # Run confidence model if needed
    if confidence_model is not None:
        patch_stack = torch.stack([patch_tensor, binary_segmentation], dim=1).float()  # shape: [B, 2, D, H, W]
        with torch.no_grad():
            confidence_outputs = confidence_model(patch_stack)
        logits = confidence_outputs["logit"]
        confidence_scores = F.softmax(logits, dim=1)
        print(f"confidence scores: {confidence_scores}")
    
    return binary_segmentation

def visualize_segmentation(patch_tensor: torch.Tensor, binary_segmentation: torch.Tensor, output_path: str):
    """
    visualizes the binary segmentation of each patch in the batch with 
    the slice having the largest nodule area highlighted.

    Parameters:
    - patch_tensor (torch.Tensor): The input patch tensor of shape (B, D, H, W).
    - binary_segmentation (torch.Tensor): The binary segmentation tensor of shape (B, D, H, W).
    - output_path (str): The path to save the visualization output.
    """
    batch_size = patch_tensor.shape[0]
    for i in range(batch_size):
        patch = patch_tensor[i]  # shape: (D, H, W)
        segmentation = binary_segmentation[i]  # shape: (D, H, W)

        # Find the slice with the largest nodule area
        slice_areas = segmentation.sum(dim=(1, 2))  # shape: (D,)
        max_slice_idx = torch.argmax(slice_areas).item()

        # Plot the slice and its segmentation
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(patch[max_slice_idx].cpu(), cmap='gray')
        ax[0].set_title('Original Patch Slice')
        ax[1].imshow(segmentation[max_slice_idx].cpu(), cmap='gray')
        ax[1].set_title('Binary Segmentation Slice')

        plt.suptitle(f'Patch {i} - Slice {max_slice_idx} with Largest Nodule Area')
        plt.savefig(f"{output_path}/patch_{i}_segmentation.png")
        plt.close()


def get_volumes(binary_segmentation: torch.Tensor, pixel_spacing: list[float]) -> torch.Tensor:
    """
    Gets volume of segmented nodule patches for batch of binary segmentations.

    Parameters:
    - binary_segmentation (torch.Tensor): The binary segmentation tensor of shape (B, D, H, W).
    - pixel_spacing (list[float]): The pixel spacing in mm for each dimension (z, y, x).

    Returns:
    - volumes (torch.Tensor): The volumes of the segmented nodules for each patch in the batch, shape (B,).
    """
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]  # in mm^3 ??
    patch_pixel_counts = binary_segmentation.sum(dim=(1, 2, 3))  # shape: (B,)
    print(f"patch pixel counts: {patch_pixel_counts}")
    volumes = patch_pixel_counts * pixel_volume  # shape: (B,)
    return volumes

if __name__ == "__main__":
    # Load segmentation model checkpoint
    segmentation_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt",
        weights_only=False
    )

    print("Loaded segmentation model checkpoint.")

    new_segmentation_model_state_dict = {}
    for k, v in segmentation_model_checkpoint["state_dict"].items():
        if "classifier" not in k:
            new_k = k.replace("model.model", "model")  
            new_segmentation_model_state_dict[new_k] = v

    segmentation_model = nnUNet(
        segmentation_model_checkpoint["hyper_parameters"]["args"]
    )
    segmentation_model.load_state_dict(new_segmentation_model_state_dict)

    print("Segmentation model loaded successfully.")

    # Load confidence model checkpoint
    confidence_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/4296b4b6cda063e96d52aabfb0694a04/4296b4b6cda063e96d52aabfb0694a04epoch=9.ckpt",
        weights_only=False
    )

    new_confidence_model_state_dict = {}
    for k, v in confidence_model_checkpoint["state_dict"].items():
        new_k = k.replace("model.model", "model")  
        if "model.classifier" in new_k:
            new_k = new_k.replace("model.classifier", "classifier")
        new_confidence_model_state_dict[new_k] = v

    confidence_model = nnUNetConfidence(
        confidence_model_checkpoint["hyper_parameters"]["args"]
    )
    confidence_model.load_state_dict(new_confidence_model_state_dict)

    print("Confidence model loaded successfully.")

    # # Load lungmask model
    # lungmask_model = LMInferer(
    #     modelpath="/data/rbg/users/pgmikhael/current/lungmask/checkpoints/unet_r231-d5d2fc3d.pth",
    #     tqdm_disable=True,
    #     batch_size=100,
    #     force_cpu=False,
    # )
    print("Models loaded successfully.")

    # load example patch tensor batch
    patch_tensor = torch.load("nodule_patches_sample_10000402215824639.pt").cpu()  # shape: [N, D, H, W], insert correct path here for tensor of batch of patches of original images
    print(f"Patch tensor shape: {patch_tensor.shape}")
    # Run patch segmenter
    binary_segmentation = patch_segmenter(
        patch_tensor,
        segmentation_model,
        # lungmask_model, 
        confidence_model,
    )  # shape: [B, D, H, W]
    # visualize segmentation
    visualize_segmentation(patch_tensor, binary_segmentation, output_path="segmentation_outputs")
    print(f"Binary segmentation shape: {binary_segmentation.shape}")
    # Calculate volumes
    pixel_spacing = [0.8007810115814209, 0.8007810115814209, 2.5]  # example pixel spacing (z, y, x) in mm
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]  # in mm^3
    volumes = get_volumes(binary_segmentation, pixel_spacing)  # shape: [B,]
    print(f"Nodule volumes (mm^3): {volumes}")
    
