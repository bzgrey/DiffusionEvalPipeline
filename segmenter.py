# Import required libraries
from segmentation_pipeline import nnUNet, nnUNetConfidence
from lungmask import LMInferer
from segmentation_pipeline import pydicom_to_nifti
from segmentation_pipeline import apply_windowing
from segmentation_pipeline import random_pad_3d_box
import torch
import torch.nn.functional as F
import cc3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import os
from pathlib import Path
from utils import get_dicom_files_with_slice_locations, order_slices

def segmenter(dicom_dir, output_dir, pixel_spacing):
    """
    Segments lung nodules from DICOM images using nnUNet and lungmask.

    Parameters:
    dicom_dir (str): Path to the directory containing DICOM files. Each is a slice of the same CT scan.
    output_dir (str): Path to the directory where output images will be saved.
    pixel_spacing (list or tuple): Pixel spacing in the format [spacing_y, spacing_x, spacing_z].

    Returns:
    patch_volumes (dict): Dictionary mapping instance IDs to their volumes in mm^3.
    """
    
    # Load segmentation model checkpoint
    segmentation_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt",
        weights_only=False
    )

    new_segmentation_model_state_dict = {}
    for k, v in segmentation_model_checkpoint["state_dict"].items():
        if "classifier" not in k:
            new_k = k.replace("model.model", "model")  
            new_segmentation_model_state_dict[new_k] = v

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
    
    # Initialize models
    segmentation_model = nnUNet(
        segmentation_model_checkpoint["hyper_parameters"]["args"]
    )
    segmentation_model.load_state_dict(new_segmentation_model_state_dict)

    confidence_model = nnUNetConfidence(
        confidence_model_checkpoint["hyper_parameters"]["args"]
    )
    confidence_model.load_state_dict(new_confidence_model_state_dict)

    # Load lungmask model
    lungmask_model = LMInferer(
        modelpath="/data/rbg/users/pgmikhael/current/lungmask/checkpoints/unet_r231-d5d2fc3d.pth",
        tqdm_disable=True,
        batch_size=100,
        force_cpu=False,
    )

    print('Models loaded successfully.')

    # Set to eval mode
    segmentation_model.eval()
    confidence_model.eval()

    # Load from DICOM directory
    img_paths, slice_locations = get_dicom_files_with_slice_locations(dicom_dir)
    sorted_img_paths, sorted_slice_locs = order_slices(img_paths, slice_locations, reverse=False)
    image = pydicom_to_nifti(
        sorted_img_paths,
        return_nifti=False, save_nifti=False,
        output_path="buffer",
    )

    # Run lung mask - transpose to put depth first
    image_ = np.transpose(image, (2, 0, 1))
    lung_mask = lungmask_model.apply(image_)
    # turn all values greater than 0 to 1
    lung_mask = (lung_mask > 0).astype(np.uint8)
    print(f"lung mask shape: {lung_mask.shape}")

    # Apply windowing
    image = apply_windowing(image.astype(np.float64), -600, 1600)
    print(f"image type after windowing: {type(image)}, shape: {image.shape}")

    # Convert to tensor and normalize
    image = torch.tensor(image) // 256
    image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, D, H, W]
    print(f"Image shape after unsqueezes: {image.shape}")

    # Interpolate to target size
    image = F.interpolate(
        image,
        size=(image.shape[2], 512, 512),
        mode="trilinear",
        align_corners=False,
    )
    image = image.squeeze(1)
    image = image[None]

    # Interpolate lung mask
    lung_mask = torch.tensor(lung_mask).unsqueeze(1)
    lung_mask = F.interpolate(
        lung_mask,
        size=(512, 512),
        mode="nearest-exact",
    )
    lung_mask = lung_mask.squeeze()

    # to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    lung_mask = lung_mask.to(device)
    segmentation_model = segmentation_model.to(device)
    # Run segmentation
    with torch.no_grad():
        segmentation_outputs = segmentation_model.predict(image.float())

    # Create binary segmentation
    binary_segmentation = (
        1 * (F.softmax(segmentation_outputs, 1)[0, 1] > 0.5) * lung_mask
    ).to("cpu")
    image = image.to("cpu")
    lung_mask = lung_mask.to("cpu")
    # Get connected components
    instance_segmentation, num_instances = cc3d.connected_components(
        binary_segmentation.cpu().numpy(),
        return_N=True,
    )
    print(f"Number of instances found: {num_instances}")

    # Convert to sparse tensor
    sparse_segmentation = torch.tensor(instance_segmentation, dtype=torch.int32).to_sparse()

    # Reshape image for patch extraction
    image = image.squeeze(0).squeeze(0).permute(1, 2, 0)  # shape: H, W, D

    patches = []
    patch_sizes = []  # Track sizes to determine max dimensions
    temp_patches = []

    # First pass: extract patches and track sizes
    for inst_id in range(1, num_instances + 1):
        zs, ys, xs = sparse_segmentation.indices()[
            :, sparse_segmentation.values() == inst_id
        ]
        box = {
            "x_start": torch.min(xs).item(),
            "x_stop": torch.max(xs).item(),
            "y_start": torch.min(ys).item(),
            "y_stop": torch.max(ys).item(),
            "z_start": torch.min(zs).item(),
            "z_stop": torch.max(zs).item(),
        }
        patch = torch.zeros_like(image)
        print(f"patch shape: {patch.shape}")
        patch[
            box["y_start"] : box["y_stop"] + 1,
            box["x_start"] : box["x_stop"] + 1,
            box["z_start"] : box["z_stop"] + 1,
        ] = binary_segmentation[
            box["z_start"] : box["z_stop"] + 1,
            box["y_start"] : box["y_stop"] + 1,
            box["x_start"] : box["x_stop"] + 1,
        ].permute(1, 2, 0)
        cbbox = random_pad_3d_box(
            box,
            image,
            min_height=128,
            min_width=128,
            min_depth=32,  # Increased to 32 to ensure enough depth for network layers
            random_hw=False,
            random_d=False,
        )
        patchx = image[cbbox]
        patchl = patch[cbbox]
        temp_patches.append((patchx, patchl))
        patch_sizes.append(patchx.shape)

    # Determine maximum dimensions across all patches
    # Also save images of slices with most non-zero values, largest nodules, for each patch
    # Also get the number of pixels greater than zero in each patch
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    patchPixelCounts = {}

    for inst_id, (patchx, patchl) in enumerate(temp_patches, start=1):
        # Find slice with most non-zero values
        non_zero_counts = (patchl > 0).sum(dim=(0, 1))  # Sum over H and W dimensions
        max_slice_idx = torch.argmax(non_zero_counts).item()
        # Calculate total number of pixels greater than zero in the patch (pixels in nodule)
        num_pixels_greater_than_zero = non_zero_counts.sum().item()
        patchPixelCounts[inst_id] = num_pixels_greater_than_zero

        
        # Extract the slice
        img_slice = patchx[:, :, max_slice_idx].numpy()
        mask_slice = patchl[:, :, max_slice_idx].numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title(f'Instance {inst_id} - Image (Slice {max_slice_idx})')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask_slice, cmap='gray')
        axes[1].set_title(f'Instance {inst_id} - Mask (Slice {max_slice_idx})')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_slice, cmap='gray')
        axes[2].imshow(mask_slice, cmap='Reds', alpha=0.5)
        axes[2].set_title(f'Instance {inst_id} - Overlay (Slice {max_slice_idx})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f'instance_{inst_id:03d}_slice_{max_slice_idx}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    print(f"Saved visualizations to {output_path}")
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]
    patch_volumes = {inst_id: count * pixel_volume for inst_id, count in patchPixelCounts.items()}

    # Determine maximum dimensions across all patches
    max_h = max(s[0] for s in patch_sizes)
    max_w = max(s[1] for s in patch_sizes)
    # max_d = max(s[2] for s in patch_sizes)
    max_d = 32

    # Second pass: pad all patches to max dimensions
    for patchx, patchl in temp_patches:
        # Pad to max dimensions
        pad_h = max_h - patchx.shape[0]
        pad_w = max_w - patchx.shape[1]
        pad_d = max_d - patchx.shape[2]
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            patchx = F.pad(patchx, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)
            patchl = F.pad(patchl, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)
        
        patch = torch.stack([patchx, patchl])
        patches.append(patch)

    patches = torch.stack(patches)

    # Run confidence model
    confidence_model = confidence_model.to(device)
    with torch.no_grad():
        confidence_outputs = confidence_model(patches.float().to(device))

    logits = confidence_outputs['logit']
    # run softmax on logits to get confidence scores
    confidence_scores = F.softmax(logits, dim=1)
    print(f"confidence scores: {confidence_scores}")

    return patch_volumes, confidence_scores

if __name__ == "__main__":
    # dicom_dir = "/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set2/batch2/208089/T2/1.3.6.1.4.1.14519.5.2.1.7009.9004.249204139349143430936217412730/"
    # output_dir = "./segmentation_outputs"
    # pixel_spacing = [0.703125, 0.703125, 2.0]  # Example pixel spacing in mm


    # patch_volumes, confidence_scores = segmenter(dicom_dir, output_dir, pixel_spacing)
    # total_volume = 0
    # for patch_id, volume in patch_volumes.items():
    #     if confidence_scores[patch_id-1, 0] > 0.5:  # Assuming class 1 is the positive class
    #         total_volume += volume
    #     print(f"Patch ID: {patch_id}, Volume (mm^3): {volume}, Confidence Scores: {confidence_scores[patch_id-1].cpu().numpy()}")
    
    # print(f"Total Volume of Nodules (mm^3): {total_volume}")
    # load pytorch tensor
    tensor = torch.load("/data/rbg/scratch/lung_ct/nlst_abnormalities51_nnunet_sparse_segmentation/sample_10000402215824639.pt")
    print(tensor["sparse_segmentation"].coalesce().indices().shape)
    print(tensor["sparse_segmentation"].coalesce().values().shape)
