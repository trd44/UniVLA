"""
obs_transforms.py

Contains observation-level transforms used in the orca data pipeline.

These transforms operate on the "observation" dictionary, and are applied at a per-frame level.
"""

from typing import Dict, Tuple, Union

import dlimp as dl
import tensorflow as tf
from absl import logging

def safe_set_shape(tensor, shape):
    if hasattr(tensor, "set_shape") and hasattr(tensor, "shape"):
        try:
            # Only set shape if rank matches
            if tensor.shape.rank == len(shape):
                tensor.set_shape(shape)
        except Exception:
            pass
    return tensor

# ruff: noqa: B023
def augment(obs: Dict, seed: tf.Tensor, augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images and invalid tensors."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        image = obs[f"image_{name}"]
        # --- PATCH START: Skip images that aren't valid 3D or 4D tensors ---
        if not (hasattr(image, "shape") and (image.shape.rank == 3 or image.shape.rank == 4)):
            # Just skip augmentation for this field
            print(f'skipping augment')
            continue
        # --- PATCH END ---
        kwargs = augment_kwargs[name]
        logging.debug(f"Augmenting image_{name} with kwargs {kwargs}")
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: dl.transforms.augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f"image_{name}"],  # skip padding images
        )

    return obs


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}
    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        image = obs[f"image_{name}"]

        def process_img(img_bytes):
            # If empty string, return zeros
            if img_bytes.dtype == tf.string:
                return tf.cond(
                    tf.strings.length(img_bytes) == 0,
                    lambda: tf.zeros([256, 256, 3], dtype=tf.uint8),
                    lambda: tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8)
                )
            elif img_bytes.dtype == tf.uint8:
                return img_bytes
            else:
                raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {img_bytes.dtype}")

        # Map over per-timestep images
        if image.dtype == tf.string and image.shape.rank == 1:
            image = tf.map_fn(process_img, image, fn_output_signature=tf.TensorSpec([256,256,3], dtype=tf.uint8))
        else:
            image = process_img(image)

        # Always set static shape after decoding
        image = safe_set_shape(image, [256, 256, 3])

        # Resize if needed
        if name in resize_size:
            # Only attempt resize if shape is known and is rank 3 or 4
            if (
                hasattr(image, "shape")
                and image.shape.rank in [3, 4]
                and all([s is not None and s > 0 for s in image.shape.as_list()])
            ):
                image = dl.transforms.resize_image(image, size=resize_size[name])
                image = safe_set_shape(image, [resize_size[name][0], resize_size[name][1], 3])
            else:
                # If image is not proper shape for resizing, skip
                pass
            obs[f"image_{name}"] = image

    for name in depth_names:
        depth = obs[f"depth_{name}"]

        def process_depth(depth_bytes):
            if depth_bytes.dtype == tf.string:
                return tf.cond(
                    tf.strings.length(depth_bytes) == 0,
                    lambda: tf.zeros([256, 256, 1], dtype=tf.float32),
                    lambda: tf.io.decode_image(depth_bytes, expand_animations=False, dtype=tf.float32)[..., 0:1]
                )
            elif depth_bytes.dtype == tf.float32:
                return depth_bytes
            else:
                raise ValueError(f"Unsupported depth dtype: found depth_{name} with dtype {depth_bytes.dtype}")

        if depth.dtype == tf.string and depth.shape.rank == 1:
            depth = tf.map_fn(process_depth, depth, fn_output_signature=tf.TensorSpec([256,256,1], dtype=tf.float32))
        else:
            depth = process_depth(depth)

        depth = safe_set_shape(depth, [256, 256, 1])

        if name in depth_resize_size:
            # Only attempt resize if shape is known and is rank 3 or 4
            if (
                hasattr(depth, "shape")
                and depth.shape.rank in [3, 4]
                and all([s is not None and s > 0 for s in depth.shape.as_list()])
            ):
                depth = dl.transforms.resize_depth_image(depth, size=depth_resize_size[name])
                depth = safe_set_shape(depth, [depth_resize_size[name][0], depth_resize_size[name][1], 1])
            else:
                # If depth is not proper shape for resizing, skip
                pass
            obs[f"depth_{name}"] = depth

    return obs
