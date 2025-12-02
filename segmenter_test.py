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
from torchvision import transforms
import pydicom
import os
from pathlib import Path
print("testing segmenter_test.py")
def get_dicom_files_with_slice_locations(directory_path):
    """
    Loop through all DICOM files in a directory and extract their paths and slice locations.
    
    Args:
        directory_path (str): Path to the directory containing DICOM files
        
    Returns:
        tuple: (list of file paths, list of corresponding slice locations)
    """
    dicom_paths = []
    slice_locations = []
    
    # Convert to Path object for easier handling
    dir_path = Path(directory_path)
    
    # Check if directory exists
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Loop through all files in directory (including subdirectories)
    for file_path in dir_path.rglob('*'):
        # Skip directories
        if file_path.is_dir():
            continue
            
        try:
            # Try to read as DICOM file
            dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
            
            # Check if SliceLocation exists in metadata
            if hasattr(dcm, 'Slice Location'):
                dicom_paths.append(str(file_path))
                slice_locations.append(float(dcm['Slice Location'].value))
            elif hasattr(dcm, 'ImagePositionPatient'):
                # Alternative: use Z coordinate from ImagePositionPatient if SliceLocation not available
                dicom_paths.append(str(file_path))
                slice_locations.append(float(dcm.ImagePositionPatient[2]))
        except Exception as e:
            # Skip files that are not valid DICOM files
            continue
    
    return dicom_paths, slice_locations

def order_slices(img_paths, slice_locations, reverse=False):
    sorted_ids = np.argsort(slice_locations)
    if reverse:
        sorted_ids = sorted_ids[::-1]
    sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
    sorted_slice_locs = np.sort(slice_locations).tolist()

    return sorted_img_paths, sorted_slice_locs

segmentation_model_checkpoint = torch.load(
    "/data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt",
    weights_only=False
)

new_segmentation_model_state_dict = {}
for k, v in segmentation_model_checkpoint["state_dict"].items():
    if "classifier" not in k:
        new_k = k.replace("model.model", "model")  
        new_segmentation_model_state_dict[new_k] = v

# print("Loaded segmentation model checkpoint: ", segmentation_model_checkpoint)
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

# print(new_segmentation_model_state_dict.keys())
segmentation_model = nnUNet(
    segmentation_model_checkpoint["hyper_parameters"]["args"]
)
segmentation_model.load_state_dict(new_segmentation_model_state_dict)
# segmentation_model = nnUNet(
#     segmentation_model_checkpoint["hyper_parameters"]["args"]
# )
# segmentation_model.load_state_dict(segmentation_model_checkpoint["state_dict"])
confidence_model = nnUNetConfidence(
    confidence_model_checkpoint["hyper_parameters"]["args"]
)
confidence_model.load_state_dict(new_confidence_model_state_dict)
# lungmask model
model = LMInferer(
    modelpath="/data/rbg/users/pgmikhael/current/lungmask/checkpoints/unet_r231-d5d2fc3d.pth",
    tqdm_disable=True,
    batch_size=100,
    force_cpu=False,
)

# eval mode
segmentation_model.eval()
confidence_model.eval()

# Get 3d image slice paths sorted 
# dicom_dir = "/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set2/batch1/102676/T0/1.2.840.113654.2.55.106468547949258489874106374248199128625/"
# img_paths, slice_locations = get_dicom_files_with_slice_locations(dicom_dir)
# sorted_img_paths, sorted_slice_locs = order_slices(img_paths, slice_locations, reverse=False)
# depth = len(sorted_img_paths)
# sorted_img_paths = sorted_img_paths[depth//2 - 10: depth//2 + 10]
# print(f"Number of slices found: {len(sorted_img_paths)}")
# test case
voxel_spacing = [0.8, 0.8, 1.5]  # y, x, z
affine = torch.diag(torch.tensor(voxel_spacing + [1]))
# image = pydicom_to_nifti(
#     # ["/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set2/batch1/122361/T1/1.2.840.113654.2.55.210451208063625047828616019396666958685/1.2.840.113654.2.55.248331272508909016484676525087081323464.dcm"],
#     # ["/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set1/batch1/132364/T0/1.2.840.113654.2.55.176751358824483381611253601723838247975/1.2.840.113654.2.55.6931025071464278566288717651968901396.dcm"], # this one at least produced something in the lung mask
#     # ["/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set2/batch1/102676/T0/1.2.840.113654.2.55.106468547949258489874106374248199128625/1.2.840.113654.2.55.3251552918828949596699219932499817673.dcm"],
#     sorted_img_paths,
#     return_nifti=False, save_nifti=False,
#     output_path="buffer",
# )
image = np.load("image_array_depth20.npy")
# image = image[:,:,10:12] # select middle 2 slices for testing to save time
# np.save('image_array_depth118.npy', image)
print(f"Image shape: {image.shape}, dtype: {image.dtype}")
# Save a few slices from the 3D lung mask
# mid_slice = image.shape[-1] // 2
# for i, offset in enumerate([-2, -1, 0, 1, 2]):
#     slice_idx = mid_slice + offset
#     if 0 <= slice_idx < image.shape[-1]:
#         plt.imsave(f'image5_slices/{slice_idx}.png', image[:,:,slice_idx], cmap='gray')
# pil_image = Image.open("/data/rbg/users/duitz/SybilX/diffusion/clean_images/copied_image_0.png").convert("L")
# image = np.asarray(pil_image)
# print(np.unique(image)) 

# plt.imsave('original_image4.png', image.squeeze(-1), cmap='gray')
# plt.imsave('original_image2.png', image, cmap='gray')
print(f"Original image type: {type(image)}, shape: {image.shape}")
# run lung mask
image_ = np.transpose(image, (2, 0, 1)) # for the pydicom that puts depth last
# image_ = image
# image_for_lung_mask = np.expand_dims(image_.astype(np.float64), axis=0)  # must have depth dim first
# print(f"unique values in image for lung mask: {np.unique(image_for_lung_mask)}")
# lung_mask = model.apply(image_for_lung_mask) # must have depth dim first
lung_mask = model.apply(image_)
print(f"Lung mask shape: {lung_mask.shape}")
# Save a few slices from the 3D lung mask
mid_slice = lung_mask.shape[0] // 2
# for i, offset in enumerate([-2, -1, 0, 1, 2]):
#     slice_idx = mid_slice + offset
#     if 0 <= slice_idx < lung_mask.shape[0]:
#         plt.imsave(f'lung_masks/lung_mask5_slice_{slice_idx}.png', lung_mask[slice_idx], cmap='gray')
# plt.imsave('lung_masks/lung_mask5.png', lung_mask.squeeze(0), cmap='gray')
print(f"Lung mask unique values: {np.unique(lung_mask)}")
# preprocess image
image = apply_windowing(image.astype(np.float64), -600, 1600)

print(f"image type after windowing: {type(image)}, shape: {image.shape}")
image = torch.tensor(image) // 256
# image = image.permute(2, 0, 1)
# image = image.unsqueeze(1)  # shape: [1, 1, H, W] Note: this is assuming the depth dim is 1, so another unsqueeze isn't needed bc we are doing 2d interpolate
image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, D, H, W]
print(f"Image shape after unsqueezes: {image.shape}")
image = F.interpolate(
    image,
    size=(image.shape[2], 1024, 1024),
    mode="trilinear",
    align_corners=False,
)
image = image.squeeze(1)
image = image[None]
print(f"Image shape after interpolation: {image.shape}")

lung_mask = torch.tensor(lung_mask).unsqueeze(1)
lung_mask = F.interpolate(
    lung_mask,
    size=(1024, 1024),
    mode="nearest-exact",
    # align_corners=False,
)
lung_mask = lung_mask.squeeze()
print(f"Lung mask shape after interpolation: {lung_mask.shape}")

with torch.no_grad():
    segmentation_outputs = segmentation_model.predict(image.float())

print(f"segmentation unique values: {torch.unique(segmentation_outputs)}")

binary_segmentation = (
    1 * (F.softmax(segmentation_outputs, 1)[0, 1] > 0.5) * lung_mask
)
print(f"Binary segmentation shape: {binary_segmentation.shape}")
# for i, offset in enumerate([-2, -1, 0, 1, 2]):
#     slice_idx = mid_slice + offset
#     if 0 <= slice_idx < binary_segmentation.shape[0]:
#         plt.imsave(f'binary_segmentation/bin_seg5_slice_{slice_idx}.png', binary_segmentation[slice_idx], cmap='gray')
# plt.imsave('binary_segmentation.png', binary_segmentation.squeeze(0).cpu().numpy(), cmap='gray')
# plt.imshow(binary_segmentation.squeeze(0).cpu().numpy(), cmap='gray')
# plt.title('Binary Segmentation')
# plt.axis('off')
# plt.show()
# get connected components
instance_segmentation, num_instances = cc3d.connected_components(
    binary_segmentation.cpu().numpy(),
    return_N=True,
)
print(f"Number of instances found: {num_instances}")
print(f"Instance segmentation shape: {instance_segmentation.shape}")
# convert to sparse tensor
sparse_segmentation = torch.tensor(instance_segmentation, dtype=torch.int32).to_sparse()
print(f"Sparse segmentation indices shape: {sparse_segmentation.indices().shape}")
print(f"Image shape: {image.shape}")
image = image.squeeze(0).squeeze(0).permute(1, 2, 0)  # shape: H, W, D
patches = []
patch_sizes = []  # Track sizes to determine max dimensions

# First pass: extract patches and track sizes
temp_patches = []
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
        min_depth=10,
        random_hw=False,
        random_d=False,
    )
    patchx = image[cbbox]
    patchl = patch[cbbox]
    temp_patches.append((patchx, patchl))
    patch_sizes.append(patchx.shape)

# Determine maximum dimensions across all patches
max_h = max(s[0] for s in patch_sizes)
max_w = max(s[1] for s in patch_sizes)
max_d = max(s[2] for s in patch_sizes)
print(f"Max patch dimensions: H={max_h}, W={max_w}, D={max_d}")

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
print(f"Total patches shape: {patches.shape}")
with torch.no_grad():
    confidence_outputs = confidence_model(patches.float())

print(f"Confidence outputs shape: {confidence_outputs.shape}")
# print the first 10 confidence scores
for i in range(min(10, confidence_outputs.shape[0])):
    print(f"Instance {i+1} confidence score: {confidence_outputs[i].item()}")
