"""
This module contains functions for generating targets from ccl masks.
"""

import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import morphology as morph


def generate_directions(num_rays: int) -> List[Tuple[float, float]]:
    """
    Generates a list of tuple pairs that represent x and y directions for each ray.
    :arg num_rays: Number of rays for generating directions.
    :return: Generated list of directions.
    """
    directions = []
    st_rays = np.float32(2 * np.pi) / num_rays
    for i in range(num_rays):
        phi = np.float32(i * st_rays - np.pi / 2)
        direction_y = np.cos(phi)
        direction_x = np.sin(phi)
        directions.append((float(direction_x), float(direction_y)))
    return directions


def generate_rays_map_single_instance(
    seg_mask: np.ndarray,
    directions: List[Tuple[float, float]],
    object_size_threshold: Optional[int] = None,
) -> np.ndarray:
    """
    Rays distance map represents distances from each pixel to the nearest background
    pixel in a set of directions (rays) passed down as an argument.
    :arg seg_mask: Matrix mask where each instance is identified with a unique value.
    The value of background pixels is zero.
    :arg directions: List of offset pairs (y, x) for each ray direction.
    :arg object_size_threshold: Objects which have size below this threshold will be omitted,
    i.e. rays will be set to 0.
    :return: Generated rays map for each pixel, for each ray.
    The resulting shape of this array is (seg_mask.shape[0], seg_mask.shape[1], len(directions)).
    """
    k_max = (
        int(np.hypot(*seg_mask.shape)) + 1
    )  # maximum ray distance factor (k_max * ray unit length)
    padding_per_side: int = (
        k_max + 1
    )  # padding length for each of the 4 sides of the mask
    num_rays = len(directions)
    height, width = seg_mask.shape
    if object_size_threshold:
        seg_mask = morph.remove_small_objects(seg_mask, min_size=object_size_threshold)
    # padded mask with -1 values for out of bound values
    seg_mask_padded = np.full(
        (height + 2 * padding_per_side, width + 2 * padding_per_side), -1, dtype=int
    )
    seg_mask_padded[
        padding_per_side:-padding_per_side, padding_per_side:-padding_per_side
    ] = seg_mask
    repeated_seg_mask = seg_mask[:, :, np.newaxis]
    foreground_pixels = repeated_seg_mask != 0
    # calculate y and x offsets for all potential factors k
    ray_directions_column_y = np.array([d[0] for d in directions], dtype=float).reshape(
        (-1, 1)
    )
    ray_directions_column_x = np.array([d[1] for d in directions], dtype=float).reshape(
        (-1, 1)
    )
    all_k = np.arange(1, k_max + 1).reshape((1, -1))
    ray_offsets_y = np.zeros((height, num_rays, k_max), dtype=float) + padding_per_side
    ray_offsets_y[:] += ray_directions_column_y @ all_k
    ray_offsets_y += np.arange(height)[:, np.newaxis, np.newaxis]
    ray_offsets_y = ray_offsets_y.astype(int)
    ray_offsets_x = np.zeros((width, num_rays, k_max), dtype=float) + padding_per_side
    ray_offsets_x[:] += ray_directions_column_x @ all_k
    ray_offsets_x += np.arange(width)[:, np.newaxis, np.newaxis]
    ray_offsets_x = ray_offsets_x.astype(int)
    distance_units_per_pixel = np.zeros((height, width, num_rays), dtype=np.uint8)
    in_progress_per_pixel = np.ones((height, width, num_rays), dtype=bool)
    # process ray distance factors k one by one until all pixels have reached their nearest bg pixel
    ray_num_pixels: int = num_rays * height * width
    num_of_processed: int = 0
    k: int = 0
    while num_of_processed != ray_num_pixels and k < k_max:
        k += 1
        translated_pixels = np.fromfunction(
            lambda i, j, r: seg_mask_padded[
                ray_offsets_y[i, r, k - 1], ray_offsets_x[j, r, k - 1]
            ],
            (height, width, num_rays),
            dtype=int,
        )
        pixels_are_equal = (
            np.equal(translated_pixels, repeated_seg_mask) & foreground_pixels
        )
        distance_units_per_pixel += in_progress_per_pixel
        in_progress_per_pixel = in_progress_per_pixel & pixels_are_equal
        num_of_processed = ray_num_pixels - np.sum(in_progress_per_pixel)
    ray_directions_y = ray_directions_column_y[:, :, np.newaxis].T
    ray_directions_x = ray_directions_column_x[:, :, np.newaxis].T
    rays_map_y = distance_units_per_pixel * ray_directions_y
    rays_map_x = distance_units_per_pixel * ray_directions_x
    # small correction as we overshoot the boundary
    ray_directions_max = np.maximum(np.abs(ray_directions_y), np.abs(ray_directions_x))
    rays_map_corr = 1 - 0.5 / ray_directions_max
    rays_map_y -= rays_map_corr * ray_directions_y
    rays_map_x -= rays_map_corr * ray_directions_x
    rays_map = np.sqrt(rays_map_y**2 + rays_map_x**2) * foreground_pixels
    return rays_map


def generate_rays_map(
    instance_seg_mask: np.ndarray,
    num_rays: int = 32,
    object_size_threshold: Optional[int] = None,
) -> np.ndarray:
    """
    Rays distance map represents distances from each pixel to the nearest background
    pixel in a set of directions (rays).
    The first ray goes in an upwards direction and every subsequent ray is at 360/num_rays degrees
    from it in a clockwise direction.
    :arg instance_seg_mask: InstanceSegmentation matrix mask where each instance identified with a
    unique value.
    The value of background pixels is zero.
    :arg num_rays: Number of rays used for rays representation.
    :arg object_size_threshold: Objects which have size below this threshold will be omitted,
    i.e. rays will be set to 0.
    :return: Generated rays map for each pixel, for each ray.
    The resulting shape of this array is (instance_seg_mask.shape[0], instance_seg_mask.shape[1],
    num_rays).
    """
    height, width = instance_seg_mask.shape
    directions = generate_directions(num_rays)
    rays_map = np.zeros((height, width, num_rays), np.float32)
    unique_values = np.unique(instance_seg_mask)
    for unique_value in unique_values:
        # Skip the value zero as it represents the background
        if unique_value != 0:
            instance_mask = instance_seg_mask == unique_value
            y_min, y_max, x_min, x_max = bounding_box(instance_mask)
            cropped_mask = instance_mask[y_min : y_max + 1, x_min : x_max + 1]
            single_nucleus_ray_map = generate_rays_map_single_instance(
                cropped_mask, directions, object_size_threshold
            )
            rays_sub_map = rays_map[y_min : y_max + 1, x_min : x_max + 1, :]
            # only "fix" zero elements to values from single_nucleus_ray_map
            zero_elements = rays_sub_map == 0
            rays_sub_map[zero_elements] = single_nucleus_ray_map[zero_elements]
    return rays_map


def transform_to_polar(ccl_mask: np.ndarray) -> np.ndarray:
    """
    This function converts a CCL mask to its polar coordinate representation by creating a polar
    image where each pixel corresponds to a specific radius and angle from the centroid of the CCL.

    :arg ccl_mask: Input CCL mask where each connected component has a unique label.

    :return: Polar coordinate representation of the CCL mask as a 2D NumPy array.
    """
    # Get the height and width of the CCL mask
    height, width = ccl_mask.shape

    # Calculate the center point of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the maximum radius based on the centroid
    max_radius = int(np.sqrt(center_x**2 + center_y**2))

    polar_image = np.zeros((max_radius, 360), dtype=np.uint8)

    for theta in range(360):
        for r in range(max_radius):
            # Calculate Cartesian coordinates (x, y) from polar coordinates
            x = int(center_x + r * np.cos(np.radians(theta)))
            y = int(center_y - r * np.sin(np.radians(theta)))

            # Check if the Cartesian coordinates are within the bounds of the CCL mask
            if 0 <= x < width and 0 <= y < height:
                # Copy the label value to the corresponding pixel in the polar image
                polar_image[r, theta] = ccl_mask[y, x]

    return polar_image


def bounding_box(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Generates a bounding box of non-zero elements in an input matrix.
    :arg image: Input matrix for which the bounding box is being generated.
    :return: Four integers denoting min y, max y, min x and max x coordinates of the instance
    respectively.
    """
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max


def get_centroids_from_binary_mask(binary_mask: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the centroid coordinates (x, y) from a binary mask.
    :arg binary_mask: A binary mask representing an object.
    :return: A tuple containing the x and y coordinates of the centroid.
    """

    # Calculate the moments of the binary mask
    moments = cv2.moments(binary_mask)

    # Check if the area (m00) of the object is not zero to avoid division by zero
    if moments["m00"] != 0:
        # Calculate the x and y coordinates of the centroid
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        # If the area is zero, set centroid coordinates to (0, 0)
        center_x, center_y = 0, 0

    # Return the centroid coordinates
    return center_x, center_y


def distance_transform(ccl_mask: np.ndarray, minimum_axis_len: int = 2) -> np.ndarray:
    """
    Apply distance transform on a CCL mask.

    :arg ccl_mask: Input CCL mask.
    :arg minimum_axis_len: Minimum instance dimension to be considered.
    :return: Distance-transformed image.
    """
    # Convert the input to grayscale if it's not already
    if len(ccl_mask.shape) > 2:
        ccl_mask = cv2.cvtColor(ccl_mask, cv2.COLOR_BGR2GRAY)
    probability_map = np.zeros(ccl_mask.shape[:2], dtype=np.float32)

    instance_ids = list(np.unique(ccl_mask))
    instance_ids.remove(0)  # 0 is background
    for instance_id in instance_ids:
        instance_map_matrix = np.array(ccl_mask == instance_id, np.uint16)
        y_min, y_max, x_min, x_max = bounding_box(instance_map_matrix)
        instance_map = instance_map_matrix[y_min : y_max + 1, x_min : x_max + 1]
        instance_map_width, instance_map_height = instance_map.shape
        if (
            instance_map_width < minimum_axis_len
            or instance_map_height < minimum_axis_len
        ):
            continue
        edt = distance_transform_edt(instance_map_matrix)
        probability_instance_map = edt / (np.max(edt) + 1e-10)
        probability_map += probability_instance_map
    return probability_map


def gradient_method(ccl_mask: np.ndarray) -> np.ndarray:
    """
    This function computes gradient vectors representing the direction from each pixel
    to the centroid of its connected component in the CCL mask.
    :arg ccl_mask: Input CCL mask where each connected component has a unique label.
    :return: Gradient vectors represented as a 3D NumPy array with shape (height, width, 2),
             where the third dimension holds the X and Y components of the vectors.
    """
    # Get the height and width of the CCL mask
    height, width = ccl_mask.shape

    # Initialize an array to store the gradient vectors
    vectors_tensor = np.zeros((height, width, 2), dtype=np.float32)

    # Get unique labels from the CCL mask
    labels = np.unique(ccl_mask)

    # Iterate over each label (excluding background label 0)
    for label in labels[1:]:
        # Create a binary mask for the current label
        binary_mask = np.uint8(ccl_mask == label)

        # Calculate centroid coordinates for the current connected component
        center_x, center_y = get_centroids_from_binary_mask(binary_mask)

        # Get coordinates of pixels belonging to the current connected component
        y, x = np.where(ccl_mask == label)

        # If there are pixels belonging to the component
        if len(y) > 0:
            # Compute gradient vectors from each pixel to the centroid
            vectors_tensor[y, x, 0] = center_x - x
            vectors_tensor[y, x, 1] = center_y - y

    # Return the computed gradient vectors
    return vectors_tensor


def generate_smoothed_centers(
    ccl_mask: np.ndarray, sigma: int = 2, kernel_size: Tuple = (5, 5)
) -> np.ndarray:
    """
    This function generates a binary mask with connected component centers using moments and applies
    Gaussian filter.

    :arg ccl_mask: Image with connected components labeled.
    :arg sigma: Standard deviation of the Gaussian filter.
    :arg kernel_size: Gaussian kernel size.
    :return: Binary image with connected component centers and Gaussian filter applied.

    """
    _, labeled_mask = cv2.connectedComponents(ccl_mask.astype(np.uint8))

    # Create a binary mask of the same shape as the input
    centers_mask = np.zeros_like(ccl_mask, dtype=np.uint8)

    # Extract unique labels excluding background label 0
    unique_labels = np.unique(labeled_mask)[1:]

    for label_idx in unique_labels:
        # Find coordinates of pixels with the current label
        label_coords = np.argwhere(labeled_mask == label_idx)

        center_x, center_y = get_centroids_from_binary_mask(label_coords)

        # Check if the calculated center is within the bounds
        if (
            0 <= center_y < centers_mask.shape[0]
            and 0 <= center_x < centers_mask.shape[1]
        ):
            # Set the center pixel in the new mask
            centers_mask[center_x, center_y] = 1

    # Apply Gaussian filter to the binary mask
    centers_mask_smoothed = cv2.GaussianBlur(
        centers_mask.astype(float), ksize=kernel_size, sigmaX=sigma, sigmaY=sigma
    )

    return centers_mask_smoothed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for representing ccl masks in a given form."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="Path to the images containing" " ccl masks.",
    )
    parser.add_argument("--file_extension", type=str, help="E.g. 'npy', 'jpg', 'png'. ")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where outputs will be saved.",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        required=True,
        choices=[
            "distance_transform",
            "rays",
            "gradient_method",
            "polar_coords",
            "centers",
        ],
        help="distance_transform" "rays" "gradient_method" "polar_coords" "centers",
    )
    parser.add_argument(
        "--stack_mask_with_output",
        action="store_true",
        help="Whether to stack ccl" " masks with output",
    )
    parser.add_argument("--num_rays", type=int, help="Number of rays", default=32)
    parser.add_argument(
        "--std_sigma",
        type=int,
        help="Standard deviation of the Gaussian filter",
        default=2,
    )
    parser.add_argument(
        "--gaussian_ksize", type=int, help="Gaussian kernel size", default=5
    )

    args = parser.parse_args()
    images_path: str = args.images_path
    file_extension: str = args.file_extension
    output_path: str = args.output_path
    output_type: str = args.representation_type
    number_of_rays: int = args.num_rays
    std_sigma: int = args.std_sigma
    gaussian_ksize: int = args.gaussian_ksize
    stack_mask_with_output: bool = args.stack_mask_with_output

    # Create directory if it doesn't exist.
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Define a dictionary containing mapping representation types to corresponding functions.
    function_map: Dict[str, Callable] = {
        "distance_transform": distance_transform,
        "rays": generate_rays_map,
        "gradient_method": gradient_method,
        "polar_coords": transform_to_polar,
        "centers": generate_smoothed_centers,
    }

    # Store image names.
    image_names: List[str] = [
        filename
        for filename in os.listdir(images_path)
        if filename.endswith(file_extension)
    ]

    # Load images based on their extension.
    if file_extension == "npy":
        images = [np.load(images_path + "/" + image_name) for image_name in image_names]
    else:
        images = [
            cv2.imread(images_path + "/" + image_name) for image_name in image_names
        ]
    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        for image in images
    ]

    for img, image_name in zip(images, image_names):
        if output_type == "rays":
            ccl_mask_representation = function_map[output_type](img, number_of_rays)
        elif output_type == "centers":
            ccl_mask_representation = function_map[output_type](
                img, std_sigma, kernel_size=(gaussian_ksize, gaussian_ksize)
            )
        else:
            ccl_mask_representation = function_map[output_type](img)

        if stack_mask_with_output:
            output_image = np.dstack((img, ccl_mask_representation))
        else:
            output_image = ccl_mask_representation

        np.save(
            output_path + "/" + image_name.split(".")[0] + ".npy",
            output_image,
        )
