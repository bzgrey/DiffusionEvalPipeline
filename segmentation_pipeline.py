import random
import pydicom
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cc3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from lungmask import LMInferer
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
)
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
)
from monai.inferers.utils import (
    _create_buffered_slices,
    _compute_coords,
    _get_scan_interval,
    _flatten_struct,
    _pack_struct,
)
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


_nearest_mode = "nearest-exact"


# Helper functions from: https://github.com/innolitics/dicom-numpy/blob/master/dicom_numpy/combine_slices.py
def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_spacing(sorted_datasets):
    if len(sorted_datasets) > 1:
        slice_positions = _slice_positions(sorted_datasets)
        slice_positions_diffs = np.diff(slice_positions)
        return np.median(slice_positions_diffs)

    return getattr(sorted_datasets[0], "SpacingBetweenSlices", 0)


def _slice_positions(sorted_datasets):
    image_orientation = sorted_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in sorted_datasets]


def _ijk_to_patient_xyz_transform_matrix(sorted_datasets):
    first_dataset = sorted_datasets[0]
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(sorted_datasets)

    transform = np.identity(4, dtype=np.float32)
    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing
    transform[:3, 3] = first_dataset.ImagePositionPatient
    return transform


def pydicom_to_nifti(paths, output_path, return_nifti=False, save_nifti=True):
    slices = [pydicom.dcmread(p) for p in paths]
    image = np.stack(
        [s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in slices], -1
    )  # y, x, z
    if return_nifti or save_nifti:
        affine = _ijk_to_patient_xyz_transform_matrix(slices)
        nifti_img = nib.Nifti1Image(np.transpose(image, (1, 0, 2)), affine)  # (x, y, z)
    if save_nifti:
        nib.save(nifti_img, output_path)
    if return_nifti:
        return image, nifti_img
    return image


def apply_windowing(image, center, width, bit_size=16):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460
    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    image = image.copy()
    y_min = 0
    y_max = 2**bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    below = image <= (c - w / 2)  # pixels to be set as black
    above = image > (c + w / 2)  # pixels to be set as white
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image


def random_pad_3d_box(
    box, image, min_height=30, min_width=30, min_depth=3, random_hw=True, random_d=True
):
    """
    Expand a 3D bounding box randomly to at least min_height and min_width
    while preserving the original box inside it and returning new coordinates
    in the original coordinate space.

    Returns a new box dict.
    """
    # Extract original box
    z1, z2 = box["z_start"], box["z_stop"]
    y1, y2 = box["y_start"], box["y_stop"]
    x1, x2 = box["x_start"], box["x_stop"]

    d = z2 - z1
    h = y2 - y1
    w = x2 - x1

    # Randomly determine target size (at least min + up to 20 more)
    if random_hw:
        target_h = random.randint(max(h, min_height), max(h, min_height) + 20)
        target_w = random.randint(max(w, min_width), max(w, min_width) + 20)
    else:
        # If not random, use fixed sizes
        target_h = max(min_height, h)
        target_w = max(min_width, w)
    if random_d:
        target_z = random.randint(max(d, min_depth), max(d, min_depth) + 10)
    else:
        target_z = max(min_depth, d)

    # Compute padding needed
    pad_h = target_h - h
    pad_w = target_w - w
    pad_d = target_z - d

    # Random offset of original box inside new box
    if random_hw:
        offset_y = random.randint(0, pad_h)
        offset_x = random.randint(0, pad_w)
    else:
        offset_y = pad_h // 2
        offset_x = pad_w // 2
    if random_d:
        offset_d = random.randint(0, pad_d)
    else:
        offset_d = pad_d // 2

    # Expand box in y and x directions
    new_y1 = max(y1 - offset_y, 0)
    new_y2 = new_y1 + target_h

    new_x1 = max(x1 - offset_x, 0)
    new_x2 = new_x1 + target_w

    new_d1 = max(z1 - offset_d, 0)
    new_d2 = new_d1 + target_z

    # z dimension is unchanged
    new_box = {
        "z_start": new_d1,
        "z_stop": new_d2,
        "y_start": new_y1,
        "y_stop": new_y2,
        "x_start": new_x1,
        "x_stop": new_x2,
    }

    img_h, img_w, img_d = image.shape
    cbbox = (
        slice(max(new_box["y_start"], 0), min(new_box["y_stop"], img_h)),
        slice(max(new_box["x_start"], 0), min(new_box["x_stop"], img_w)),
        slice(max(new_box["z_start"], 0), min(new_box["z_stop"], img_d)),
    )
    return cbbox


def sliding_window_inference_custom(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]
    ],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    augmentations: Callable | None = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        process_fn: process inference output and adjust the importance map per window
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        with_coord: whether to pass the window coordinates to ``predictor``. Default is False.
            If True, the signature of ``predictor`` should be ``predictor(patch_data, patch_coord, ...)``.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    buffered = buffer_steps is not None and buffer_steps > 0
    num_spatial_dims = len(inputs.shape) - 2
    if buffered:
        if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
            raise ValueError(
                f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}."
            )
        if buffer_dim < 0:
            buffer_dim += num_spatial_dims
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    condition = kwargs.pop("condition", None)

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )
        if condition is not None:
            condition = F.pad(
                condition,
                pad=pad_size,
                mode=look_up_option(padding_mode, PytorchPadMode),
                value=cval,
            )

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(
        image_size, roi_size, scan_interval, return_slice=not buffered
    )

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    if not buffered:
        non_blocking = False
        windows_range = range(0, total_slices, sw_batch_size)
    else:
        slices, n_per_batch, b_slices, windows_range = _create_buffered_slices(
            slices, batch_size, sw_batch_size, buffer_dim, buffer_steps
        )
        non_blocking, _ss = torch.cuda.is_available(), -1
        for x in b_slices[:n_per_batch]:
            if x[1] < _ss:  # detect overlapping slices
                non_blocking = False
                break
            _ss = x[2]

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size,
                mode=mode,
                sigma_scale=sigma_scale,
                device=sw_device,
                dtype=compute_dtype,
            )
            if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                importance_map_ = importance_map_[
                    None, None
                ]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(
        importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype
    )[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        slice_range = range(
            slice_g,
            min(
                slice_g + sw_batch_size, b_slices[b_s][0] if buffered else total_slices
            ),
        )
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
                sw_device
            )
            if condition is not None:
                win_condition = torch.cat(
                    [condition[win_slice] for win_slice in unravel_slice]
                ).to(sw_device)
                kwargs["condition"] = win_condition
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
            if condition is not None:
                win_condition = condition[unravel_slice[0]].to(sw_device)
                kwargs["condition"] = win_condition

        # ------------------------------------------------------------------------>
        win_data = augmentations(win_data)
        # win_data = augmentations({"image": win_data[0, 0].permute(1, 2, 0)})["image"]
        # win_data = win_data.permute(0, 3, 1, 2)[None]
        # ------------------------------------------------------------------------>

        if with_coord:
            seg_prob_out = predictor(win_data, unravel_slice, *args, **kwargs)
        else:
            seg_prob_out = predictor(win_data, *args, **kwargs)
        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        if process_fn:
            seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
        else:
            w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        if buffered:
            c_start, c_end = b_slices[b_s][1:]
            if not sw_device_buffer:
                k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                sp_size = list(image_size)
                sp_size[buffer_dim] = c_end - c_start
                sw_device_buffer = [
                    torch.zeros(
                        size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device
                    )
                ]
            for p, s in zip(seg_tuple[0], unravel_slice):
                offset = s[buffer_dim + 2].start - c_start
                s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                s[0] = slice(0, 1)
                sw_device_buffer[0][s] += p * w_t
            b_i += len(unravel_slice)
            if b_i < b_slices[b_s][0]:
                continue
        else:
            sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if not buffered and seg_shape != roi_size:
                z_scale = [
                    out_w_i / float(in_w_i)
                    for out_w_i, in_w_i in zip(seg_shape, roi_size)
                ]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += (
                    [int(_i * _z) for _i, _z in zip(image_size, z_scale)]
                    if z_scale
                    else list(image_size)
                )
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                output_image_list.append(
                    new_tensor(output_shape, dtype=compute_dtype, device=device)
                )
                count_map_list.append(
                    torch.zeros(
                        [1, 1] + output_shape[2:], dtype=compute_dtype, device=device
                    )
                )
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(
                            slice(int(_si.start * z_s), int(_si.stop * z_s))
                            for _si, z_s in zip(__s, z_scale)
                        )
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
            if buffered:
                o_slice = [slice(None)] * len(inputs.shape)
                o_slice[buffer_dim + 2] = slice(c_start, c_end)
                img_b = b_s // n_per_batch  # image batch index
                o_slice[0] = slice(img_b, img_b + 1)
                if non_blocking:
                    output_image_list[0][o_slice].copy_(
                        sw_device_buffer[0], non_blocking=non_blocking
                    )
                else:
                    output_image_list[0][o_slice] += sw_device_buffer[0].to(
                        device=device
                    )
            else:
                sw_device_buffer[ss] *= w_t
                sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                _compute_coords(
                    unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss]
                )
        sw_device_buffer = []
        if buffered:
            b_s += 1

    if non_blocking:
        torch.cuda.current_stream().synchronize()

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        kwargs.update({"pad_size": pad_size})
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [
                _shape_d / _roi_size_d
                for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)
            ]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore


def min_max_normalize_batch(tensor):
    # tensor shape: (B, 1, D, H, W)
    # Compute min and max over (D, H, W) for each batch
    min_vals = tensor.amin(dim=(2, 3, 4), keepdim=True)
    max_vals = tensor.amax(dim=(2, 3, 4), keepdim=True)
    normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized


class nnUNet(nn.Module):
    def __init__(self, args):
        super(nnUNet, self).__init__()
        nn_args = {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": torch.nn.modules.conv.Conv3d,
            "kernel_sizes": [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "strides": [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": torch.nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        }

        self.model = ResidualEncoderUNet(
            input_channels=args.num_chan, num_classes=args.num_classes, **nn_args
        )
        if args.module_snapshot is not None:
            weights = torch.load(args.module_snapshot, weights_only=False)
            if "hyper_parameters" in weights:
                weights = {
                    k[len("model.") :]: v for k, v in weights["state_dict"].items()
                }
                weights = {
                    k[len("model.") :]: v for k, v in weights.items() if "model." in k
                }
                weights = {"network_weights": weights}

            for key in [
                "encoder.stem.convs.0.conv.weight",
                "encoder.stem.convs.0.all_modules.0.weight",
                "decoder.encoder.stem.convs.0.conv.weight",
                "decoder.encoder.stem.convs.0.all_modules.0.weight",
            ]:
                weights["network_weights"][key] = weights["network_weights"][key][
                    :, : args.num_chan
                ]

            if args.num_classes != 2:
                for key in [
                    "decoder.seg_layers.0.weight",
                    "decoder.seg_layers.0.bias",
                    "decoder.seg_layers.1.weight",
                    "decoder.seg_layers.1.bias",
                    "decoder.seg_layers.2.weight",
                    "decoder.seg_layers.2.bias",
                    "decoder.seg_layers.3.weight",
                    "decoder.seg_layers.3.bias",
                    "decoder.seg_layers.4.weight",
                    "decoder.seg_layers.4.bias",
                ]:
                    original_weight = weights["network_weights"][key]
                    if original_weight.shape[0] < args.num_classes:
                        # Repeat weights to achieve num_classes dimension
                        repeat_factor = (
                            args.num_classes + original_weight.shape[0] - 1
                        ) // original_weight.shape[0]
                        repeated_weight = original_weight.repeat(
                            repeat_factor, *[1] * (len(original_weight.shape) - 1)
                        )
                        weights["network_weights"][key] = repeated_weight[
                            : args.num_classes
                        ]
                    else:
                        # Truncate if we have more weights than needed
                        weights["network_weights"][key] = original_weight[
                            : args.num_classes
                        ]
            self.model.load_state_dict(weights["network_weights"])

        self.roi_size = (args.anatomix_crop_size[-1],) + tuple(
            args.anatomix_crop_size[:-1]
        )
        self.args = args

    def forward(self, x, batch=None):
        if self.args.predict:
            outputs = self.predict(x, batch)
            predicted_scores = F.softmax(outputs, 1)
            outputs = {
                "pred_mask_logit": outputs,
                "pred_mask": predicted_scores,  # prob score
                "hidden": predicted_scores[:, 1],
                "pred_masks_pos": 1 * (predicted_scores[:, -1] > 0.5),  # binary
            }
        else:
            skips = self.model.encoder(x)
            outputs = self.model.decoder(skips)

            predicted_scores = F.softmax(outputs, 1)
            outputs = {
                "logit": outputs,
                "pred_mask": predicted_scores,  # prob score
                "pred_masks_pos": 1 * (predicted_scores[:, -1] > 0.5),  # binary
                "hidden": predicted_scores,
                # "losses": losses,
            }

        return outputs

    @torch.no_grad()
    def predict(self, x, batch=None):
        outputs = sliding_window_inference_custom(
            inputs=x,
            predictor=self.model,
            roi_size=self.roi_size,
            overlap=0.5,
            sw_batch_size=82,
            progress=False,
            augmentations=min_max_normalize_batch,
        )

        return outputs


class nnUNetConfidence(nn.Module):
    def __init__(self, args):
        super(nnUNetConfidence, self).__init__()
        nn_args = {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": torch.nn.modules.conv.Conv3d,
            "kernel_sizes": [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "strides": [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": torch.nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        }

        self.model = ResidualEncoderUNet(
            input_channels=args.num_chan, num_classes=2, **nn_args
        )
        if args.module_snapshot is not None:
            weights = torch.load(args.module_snapshot, weights_only=False)
            for key in [
                "encoder.stem.convs.0.conv.weight",
                "encoder.stem.convs.0.all_modules.0.weight",
                "decoder.encoder.stem.convs.0.conv.weight",
                "decoder.encoder.stem.convs.0.all_modules.0.weight",
            ]:
                weights["network_weights"][key] = weights["network_weights"][key][
                    :, : args.num_chan
                ]
            self.model.load_state_dict(weights["network_weights"])

        self.model = self.model.encoder

        self.classifier = nn.ModuleList()
        for chan in [32, 64, 128, 256, 320, 320]:
            self.classifier.append(nn.Linear(chan, 2))
        self.args = args

    def forward(self, x, batch=None):
        if (self.args.dataset == "nlst_sparse_confidence") and (
            self.args.batch_size == 1
        ):
            x = x[0]
        skips = self.model(x)

        # Use the classifier to compute detection score
        detection_score = 0
        for i, hidden in enumerate(skips):
            detection_score = detection_score + self.classifier[i](
                torch.amax(hidden, dim=(2, 3, 4))
            )
        if (self.args.dataset == "nlst_sparse_confidence") and (
            self.args.batch_size == 1
        ):
            detection_score = detection_score.unsqueeze(0)
        outputs = {"logit": detection_score} # Benny removed .as_tensor() from this line

        return outputs


if __name__ == "__main__":
    segmentation_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt"
    )
    confidence_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/4296b4b6cda063e96d52aabfb0694a04/4296b4b6cda063e96d52aabfb0694a04epoch=9.ckpt"
    )
    segmentation_model = nnUNet(
        segmentation_model_checkpoint["hyper_parameters"]["args"]
    )
    segmentation_model.load_state_dict(segmentation_model_checkpoint["state_dict"])
    confidence_model = nnUNetConfidence(
        confidence_model_checkpoint["hyper_parameters"]["args"]
    )
    confidence_model.load_state_dict(confidence_model_checkpoint["state_dict"])
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

    # test case
    voxel_spacing = [0.8, 0.8, 1.5]  # y, x, z
    affine = torch.diag(torch.tensor(voxel_spacing + [1]))
    image = pydicom_to_nifti(
        "/directory/to/dicoms", return_nifti=False, save_nifti=False
    )

    # run lung mask
    image_ = np.transpose(image, (2, 0, 1))
    lung_mask = model.apply(image_)

    # preprocess image
    image = apply_windowing(image.astype(np.float64), -600, 1600)
    image = image // 256
    image = image.permute(2, 0, 1).unsqueeze(1)
    image = F.interpolate(
        image,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    image = image.squeeze(1)
    image = image[None]
    lung_mask = F.interpolate(
        lung_mask,
        size=(1024, 1024),
        mode="nearest-exact",
        align_corners=False,
    )
    lung_mask = lung_mask.squeeze()

    with torch.no_grad():
        segmentation_outputs = segmentation_model.predict(image)

    binary_segmentation = (
        1 * (F.softmax(segmentation_outputs, 1)[0, 1] > 0.5) * lung_mask
    )

    # get connected components
    instance_segmentation, num_instances = cc3d.connected_components(
        binary_segmentation.cpu()
    )
    sparse_segmentation = instance_segmentation.float().to_sparse()

    patches = []
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
        patch = torch.stack([patchx, patchl])
        patches.append(patch)

    patches = torch.stack(patches)
    with torch.no_grad():
        confidence_outputs = confidence_model(patches)
