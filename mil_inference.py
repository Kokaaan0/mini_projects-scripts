"""
A script for MIL inference is provided in this module.
"""
import argparse
import csv
import os
from typing import Dict, List, Tuple, Union

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from omnislide import open_slide

import trainingframework
from trainingframework.api import fetch_model, get_activations
from trainingframework.data.datasets.bag_of_tiles_types.roi_bag_of_tiles import (
    RoiBagOfTiles,
)
from trainingframework.data.datasets.bag_of_tiles_types.slide_bag_of_tiles import (
    SlideBagOfTiles,
)
from trainingframework.data.datasets.file_unit_strategy.slide_file_unit_strategy import (
    SlideFileUnitStrategy,
)
from trainingframework.data.datasets.inference_bag_of_tiles_dataset import (
    InferenceBagOfTilesDataset,
)
from trainingframework.data.file_readers.csv_reader import CsvReader
from trainingframework.data.whole_slide_image import WholeSlideImage


def initialize_attention_and_count_maps(
    bag_of_tiles_dataset: InferenceBagOfTilesDataset,
) -> Tuple[Dict, Dict]:
    """
    Initializes attention and count maps for each region of interest (ROI) in the dataset.

    :arg bag_of_tiles_dataset: The dataset containing information of rois and tiles inside them.

    :return: Tuple of count maps and attention_maps.

    """
    if isinstance(bag_of_tiles_dataset.file_unit_strategy, SlideFileUnitStrategy):
        attention_maps = {}
        count_maps = {}
        tiling_meta = bag_of_tiles_dataset.file_unit_strategy.tiling_meta
        tile_height = tiling_meta.height
        tile_width = tiling_meta.width

        idx_counter = 0
        for slide in bag_of_tiles_dataset.file_unit_strategy.slides:
            for roi in slide.tiled_rois:
                num_tiles_per_height = (roi.roi_bbox[3] - roi.roi_bbox[1]) // tile_height
                num_tiles_per_width = (roi.roi_bbox[2] - roi.roi_bbox[0]) // tile_width

                attention_maps[idx_counter] = np.zeros((num_tiles_per_height, num_tiles_per_width))
                count_maps[idx_counter] = np.zeros((num_tiles_per_height, num_tiles_per_width))
                idx_counter += 1

        return attention_maps, count_maps
    else:
        raise AssertionError("File unit strategy is not properly initialized")


def get_rois_bboxes(bag_of_tiles: InferenceBagOfTilesDataset) -> List[Tuple[int]]:
    """
    Extracts and returns bounding boxes of regions of interest (ROIs) from the provided bag of tiles
    dataset.

    :arg bag_of_tiles: The dataset containing information of rois.

    :return: List of bounding boxes of ROIs in the dataset.
    """
    if isinstance(bag_of_tiles.file_unit_strategy, SlideFileUnitStrategy):
        rois_bboxes = []
        for slide in bag_of_tiles.file_unit_strategy.slides:
            for roi in slide.tiled_rois:
                rois_bboxes.append(roi.roi_bbox)
        return rois_bboxes


def run_mil_inference(
    model: torch.nn.Module, batch: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs Multiple Instance Learning (MIL) inference using the provided model on the given batch
    of data.

    :arg model: The MIL model used for inference.

    :arg batch: List of tensors where each tensor represents bag of instances.

    :return: Tuple of predicted logits and attention weights.

    """
    with torch.no_grad():
        output = model(batch)
        pred_logits = output[0]["BagDecoder"]["BagLabel"]
        att_weights = output[0]["BagDecoder"]["Attention weights"]
        return pred_logits, att_weights


def instantiate_bag_of_tiles(
    config_path: str, images_folder: str, wsi_extension: str
) -> InferenceBagOfTilesDataset:
    """
    Instantiates a BagOfTilesDataset object based on the provided configuration and image folder.

    :arg config_path: Path to the configuration yaml file.

    :arg images_folder: Path to the folder containing the whole slide images.

    :arg wsi_extension: Extension of whole slide image, E.g. "tiff", "mrxs".

    :return: Instance of the BagOfTilesDataset.
    """
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    dataset_args = cfg["datamodule"]["dataset_args"]

    common_args = dataset_args["common"]
    val_args = dataset_args["val"]
    val_args["bag_annotation_loader"] = None
    val_args["x_filetypes"] = [wsi_extension]
    val_args["file_unit_strategy"]["roi_annotation_loader"]["cast_coords_to_int"] = True
    val_args["file_unit_strategy"]["roi_term_names"] = cfg["roi_term_names"]
    val_args["file_unit_strategy"]["tile_height"] = cfg["tile_height"]
    val_args["file_unit_strategy"]["tile_width"] = cfg["tile_width"]
    val_args["file_unit_strategy"]["tile_stride"] = cfg["tile_stride"]
    val_args["file_unit_strategy"]["tile_acceptance_condition"]["threshold"] = 1.0
    val_args["bag_sampler"]["size"] = cfg["bag_size"]

    bag_of_tiles_args = {
        **common_args,
        **val_args,
        "_target_": trainingframework.data.datasets.inference_bag_of_tiles_dataset.InferenceBagOfTilesDataset,
    }

    bag_of_tiles = hydra.utils.instantiate(bag_of_tiles_args, mode="val", root_dir=images_folder)
    return bag_of_tiles


def get_labels_filepath(config_path: str, images_folder: str) -> str:
    """
    Retrieves the file path for labels based on the configuration and image folder.

    :arg config_path: Path to the configuration YAML file containing information about datamodule.
    :arg images_folder: Path to the folder containing images.
    :return: File path to the labels file.
    """

    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    labels_filename = cfg["datamodule"]["dataset_args"]["val"]["bag_annotation_file_paths"][0]
    return images_folder + "/" + labels_filename


def update_attention_map(
    attention_map: Dict,
    att_weights: torch.Tensor,
    tiles_top_left: List[List],
    bag_indices: List[int],
) -> Dict:
    """
    Updates the attention map based on the given list of tile positions, attention weights, and bag
    indices.

    :arg attention_map: The attention map to be updated.

    :arg att_weights: Attention weights associated with each tile.

    :arg tiles_top_left: List containing lists of tile top-left coordinates for each Region
        of Interest (ROI) and their corresponding indices. Each inner list contains tuples of
        coordinates ([x, y] format) and tile index, e.g., [[([x1, y1], index1)],
        [([x2, y2], index2)]].

    :arg bag_indices: Indices of the tiles in the bag.

    :return: Updated attention map.
    """
    for i, tiles_per_roi in enumerate(tiles_top_left):
        for [x, y], tile_index in tiles_per_roi:
            if tile_index in bag_indices:
                attention_map[i][attention_map[i].shape[0] - y - 1][x] += att_weights[
                    bag_indices.index(tile_index)
                ]

    return attention_map


def update_count_map(tiles_top_left: List, count_map: Dict) -> Dict:
    """
    Updates the count map based on the given list of tile positions and a count map.

    :arg tiles_top_left: List containing lists of tile top-left coordinates for each Region
        of Interest (ROI) and their corresponding indices. Each inner list contains tuples of
        coordinates ([x, y] format) and tile index, e.g., [[([x1, y1], index1)],
        [([x2, y2], index2)]].

    :arg count_map: The count map to be updated.

    :return: Updated count map.
    """
    for i, tiles_per_batch_item in enumerate(tiles_top_left):
        for [x, y], _ in tiles_per_batch_item:
            count_map[i][count_map[i].shape[0] - y - 1][x] += 1
    return count_map


def save_predictions(
    roi_ids: List[str], predictions: Dict, save_path: str, labels: List, inference_type: str
) -> None:
    """
    Save predictions and related information to a CSV file based on the provided inference type.

    This function generates a CSV file containing predictions, ground truth labels, and additional
    evaluation metrics for each ROI based on the provided inference type.

    :arg roi_ids: List of strings containing Region of Interest (ROI) identifiers.

    :arg predictions: Dictionary containing predictions.

    :arg save_path: The directory path where the resulting CSV file will be saved.

    :arg labels: List of ground truth labels corresponding to each ROI.

    :arg inference_type: String indicating the type of inference. It can be 'classification' or
    'regression'.

    :return: None.

    """
    output_name = inference_type
    if inference_type == "classification":
        prob_0 = [np.round(float(x), 2) for _, (x, _) in predictions.items()]
        prob_1 = [np.round(float(y), 2) for _, (_, y) in predictions.items()]
        accuracy = np.round(calculate_accuracy(predictions, labels), 2)
        accuracy_col = ["" for _ in range(len(predictions))]
        accuracy_col[0] = accuracy
        rows = [
            {
                "Roi_id": r_id,
                "Ground_truth": label,
                "class_0": p0,
                "class_1": p1,
                "accuracy": acc,
            }
            for r_id, label, p0, p1, acc in zip(roi_ids, labels, prob_0, prob_1, accuracy_col)
        ]
    elif inference_type == "regression":
        preds = [np.round(float(x), 2) for (_, x) in predictions.items()]
        mean_absolute_error = np.round(calculate_mean_absolute_error(predictions, labels), 2)
        mae_col = ["" for _ in range(len(predictions))]
        mae_col[0] = mean_absolute_error

        rows = [
            {
                "Roi_id": r_id,
                "Ground_truth": label,
                "Prediction": pred,
                "Mean absolute error": mae,
            }
            for r_id, label, pred, mae in zip(roi_ids, labels, preds, mae_col)
        ]
    else:
        raise ValueError("Wrong inference type!!!")

    with open(save_path + "/" + output_name + ".csv", "w") as f:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_attention_per_bag(att_weights: torch.Tensor, bag_sizes: List[int]) -> List[torch.Tensor]:
    """
    Extracts attention weights per bag from a tensor of attention weights for each item in a batch.

    :arg bag_sizes: List of bag sizes.

    :arg att_weights: Attention weights for each item in the batch.

    :return: List containing attention weights per bag.

    """
    attentions_per_bag = []
    i = 0
    for size in bag_sizes:
        attentions_per_bag.append(att_weights[i : i + size])
        i += size
    return attentions_per_bag


def translate_tile_top_left_coordinates(
    tile_bboxes: List,
    roi_bboxes: List[Tuple[int]],
    bag_type: Union[RoiBagOfTiles, SlideBagOfTiles],
) -> List:
    """
    Translates tile coordinates relative to their respective ROI bounding boxes.

    :arg tile_bboxes: List of tile bounding boxes per ROI and their corresponding indices.

    :arg roi_bboxes: List of ROI bounding boxes.

    :arg bag_type: It can be SlideBagOfTiles if predictions are on slide level, or RoiBagOfTiles if
        predictions are on ROI level.

    :return: Translated tiles top left coordinates per ROI with their corresponding indices.
    """

    translated_coords_per_roi = []
    if isinstance(bag_type, SlideBagOfTiles):
        extracted_tile_bboxes_per_roi = [bbox for el in tile_bboxes for bbox in el]
    else:
        extracted_tile_bboxes_per_roi = tile_bboxes

    for i, (tile_bboxes_per_roi, roi_bbox) in enumerate(
        zip(extracted_tile_bboxes_per_roi, roi_bboxes)
    ):
        translated_coords = []
        for [tile_min_x, tile_min_y, tile_max_x, tile_max_y], tile_index in tile_bboxes_per_roi:
            tile_height = tile_max_y - tile_min_y
            tile_width = tile_max_x - tile_min_x
            roi_height = (roi_bbox[3] - roi_bbox[1]) // tile_height
            translated_tile_x = (tile_min_x - roi_bbox[0]) // tile_width
            translated_tile_y = (tile_min_y - roi_bbox[1]) // tile_height
            translated_coords.append(
                ([translated_tile_x, roi_height - translated_tile_y - 1], tile_index)
            )

        translated_coords_per_roi.append(translated_coords)
    return translated_coords_per_roi


def create_batch(
    bag_of_tiles: InferenceBagOfTilesDataset,
) -> Tuple[List[torch.Tensor], List, List, List]:
    """
    Creates a batch containing bags of tiles along with their corresponding bounding box coordinates
    , indices and bag_sizes.

    :arg bag_of_tiles: Object representing a bag of tiles dataset.

    :return: Tuple containing a list of bags, tile bounding box coordinates, tile indices and
        bag_sizes.
    """
    if isinstance(bag_of_tiles.file_unit_strategy, SlideFileUnitStrategy):
        region_indices = [i for i in range(len(bag_of_tiles))]
        list_of_bags = []
        tiles_bboxes = []
        tile_indices = []
        bag_sizes = []
        for index in region_indices:
            bboxes_per_bag = []
            bag, tiles_item_info, bag_indices = bag_of_tiles.__getitem__(index)
            list_of_bags.append(bag)
            for tile_item_info in tiles_item_info:
                x_min = tile_item_info.tile_bbox.x_min
                y_min = tile_item_info.tile_bbox.y_min
                x_max = tile_item_info.tile_bbox.x_max
                y_max = tile_item_info.tile_bbox.y_max
                bboxes_per_bag.append([x_min, y_min, x_max, y_max])
            tiles_bboxes.append(bboxes_per_bag)
            tile_indices.append(bag_indices)
            bag_sizes.append(bag.shape[0])

        return list_of_bags, tiles_bboxes, tile_indices, bag_sizes


def split_tile_bboxes_per_roi(
    tiles_bboxes: List,
    bag_of_tiles: InferenceBagOfTilesDataset,
    bag_indices: List[int],
) -> List[List[Tuple]]:
    """
    Splits tile bounding boxes per Region of Interest (ROI) based on bag indices.

    :arg bag_of_tiles: Object representing a bag of tiles dataset.

    :arg bag_indices: List of bag indices.

    :arg tiles_bboxes: List of tile bounding boxes per bag of tiles.

    :return: List containing tiles split per ROI.
    """

    if isinstance(bag_of_tiles.file_unit_strategy, SlideFileUnitStrategy):
        split_tiles = []  # List to hold tiles split per ROI
        if isinstance(bag_of_tiles.bag_type, SlideBagOfTiles):
            all_rois_start_indices = bag_of_tiles.file_unit_strategy.start_indices_per_roi

            # Initialize the tiles per ROI structure
            for indices_per_slide in all_rois_start_indices:
                split_tiles.append([[] for _ in range(len(indices_per_slide))])

            # Group tiles into respective ROIs based on indices and bounding boxes
            for batch_index, (start_indices, bag_indices_per_batch, bboxes_per_batch) in enumerate(
                zip(all_rois_start_indices, bag_indices, tiles_bboxes)
            ):
                for i, bag_index in enumerate(bag_indices_per_batch):
                    for roi_index in range(len(start_indices) - 1, -1, -1):
                        if bag_index >= start_indices[roi_index]:
                            split_tiles[batch_index][roi_index].append(
                                (bboxes_per_batch[i], bag_index)
                            )
                            break
            return split_tiles

        # For RoiBagOfTiles type
        elif isinstance(bag_of_tiles.bag_type, RoiBagOfTiles):
            for indices_per_roi, tile_bboxes_per_roi in zip(bag_indices, tiles_bboxes):
                tiles_per_roi = []
                for index, bbox in zip(indices_per_roi, tile_bboxes_per_roi):
                    tiles_per_roi.append((bbox, index))
                split_tiles.append(tiles_per_roi)
            return split_tiles
        else:
            raise ValueError("Wrong bag type initialized.")


def create_batch_indices(
    batch_size: int, bag_of_tiles: InferenceBagOfTilesDataset
) -> List[np.array]:
    """
    This function splits region indices in list of arrays, each one with the size of given
    batch_size.

    Note: batch_size will equal the number of available regions if batch_size exceeds the number of
        regions accessible for sampling.

    :arg batch_size: Number of items per batch.

    :arg bag_of_tiles: Instance of BagOfTilesDataset class. This object contains information of
        available regions for sampling.

    :return: List of sampled region indices.
    """
    num_of_regions_to_sample = len(bag_of_tiles)
    num_of_samples = (num_of_regions_to_sample + batch_size - 1) // batch_size

    list_of_batch_indices = [
        np.arange(i * batch_size, min((i + 1) * batch_size, num_of_regions_to_sample))
        for i in range(num_of_samples)
    ]
    return list_of_batch_indices


def read_roi_regions(slides: List[WholeSlideImage]) -> List[np.ndarray]:
    """
    Reads the regions of interest (ROIs) from a list of WholeSlideImage instances.

    :arg slides: A list of WholeSlideImage instances containing information about the slides.

    :return: List of regions inside Whole slide Images.
    """
    rois = []
    for slide in slides:
        slide_path = slide.path
        openslide_object = open_slide(slide_path)
        openslide_object.open()
        for roi in slide.tiled_rois:
            min_x, min_y, max_x, max_y = roi.roi_bbox
            roi_height = max_y - min_y
            roi_width = max_x - min_x
            region = openslide_object.read_region(
                (min_x, min_y), level=0, size=(roi_width, roi_height)
            )
            region_npy = np.array(region)
            rois.append(region_npy)

    return rois


def get_roi_ids(bag_of_tiles: InferenceBagOfTilesDataset) -> List:
    """
    Retrieves a list of Region of Interest (ROI) IDs from an InferenceBagOfTilesDataset.

    :arg bag_of_tiles: An object representing an InferenceBagOfTilesDataset.

    :return: A list of ROI IDs extracted from the dataset.
    """
    if isinstance(bag_of_tiles.file_unit_strategy, SlideFileUnitStrategy):
        return [
            roi.id for slide in bag_of_tiles.file_unit_strategy.slides for roi in slide.tiled_rois
        ]


def get_labels(
    bag_of_tiles: InferenceBagOfTilesDataset, file_path: str, label_type: str
) -> List[Union[int, float]]:
    """
    Retrieves either ROI or Slide labels from a CSV file based on corresponding IDs.

    :arg bag_of_tiles: InferenceBagOfTilesDataset object containing information about ROIs and
        Slides.

    :arg file_path: Path to the CSV file containing either ROI ids or Slide names and corresponding
        labels.

    :arg label_type: Type of labels to retrieve. It can be roi or slide.

    :return: List of labels for each region.
    """

    assert label_type in ["roi", "slide"]
    if isinstance(bag_of_tiles.file_unit_strategy, SlideFileUnitStrategy):
        csv_reader = CsvReader([file_path], None)
        dataset_dict = csv_reader.read(file_path)
        labels = []

        if label_type == "roi":
            for slide in bag_of_tiles.file_unit_strategy.slides:
                for tiled_roi in slide.tiled_rois:
                    tiled_roi_id = tiled_roi.id
                    for roi_id, roi_label in zip(dataset_dict["roi_id"], dataset_dict["roi_label"]):
                        if roi_id == tiled_roi_id:
                            labels.append(float(roi_label))
        elif label_type == "slide":
            for slide in bag_of_tiles.file_unit_strategy.slides:
                for slide_name, slide_label in zip(
                    dataset_dict["slide_name"], dataset_dict["slide_label"]
                ):
                    if slide_name == slide.name:
                        labels.append(int(slide_label))

        return labels


def calculate_mean_absolute_error(predictions: Dict, gt_labels: List) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between predictions and ground truth labels.

    :arg predictions: Dictionary containing predictions.
    :arg gt_labels: List of ground truth labels corresponding to each ROI ID.
    :return: Mean Absolute Error.
    """
    # Convert ground truth labels to a tensor
    gt_tensor = torch.tensor(gt_labels, dtype=torch.float32)

    mae = 0.0

    # Iterate through the predictions and calculate MAE
    for _, pred_tensor in predictions.items():
        # Calculate absolute differences between prediction and ground truth
        absolute_diff = torch.abs(pred_tensor - gt_tensor)
        mae += absolute_diff.mean().item()

    # Calculate average MAE
    mae /= len(predictions)

    return mae


def calculate_mean_squared_error(predictions: Dict, gt_labels: List) -> float:
    """
    Calculates the Mean Squared Error (MSE) between predictions and ground truth labels.

    :arg predictions: Dictionary containing predictions.

    :arg gt_labels: List of ground truth labels corresponding to each ROI ID.

    :return: Mean Squared Error.
    """
    # Convert ground truth labels to a tensor
    gt_tensor = torch.tensor(gt_labels, dtype=torch.float32)

    mse = 0.0

    # Iterate through the predictions and calculate MSE
    for _, pred_tensor in predictions.items():
        # Calculate squared differences between prediction and ground truth
        squared_diff = (pred_tensor - gt_tensor) ** 2
        mse += squared_diff.mean().item()

    # Calculate average MSE
    mse /= len(predictions)

    return mse


def calculate_accuracy(predictions: Dict, gt_labels: List) -> float:
    """
    Calculates the accuracy of predictions compared to ground truth labels.

    :arg predictions: Dictionary containing predictions.

    :arg gt_labels: List of ground truth labels corresponding to each ROI ID.

    :return: Accuracy as a percentage.
    """
    # Convert ground truth labels to a tensor
    gt_tensor = torch.tensor(gt_labels)

    predicted_classes = []

    # Iterate through the predictions and convert each tensor to a predicted class
    for _, pred_tensor in predictions.items():
        # Determine the predicted class by selecting the class with the maximum probability
        predicted_class = torch.argmax(pred_tensor).item()
        predicted_classes.append(predicted_class)

    # Convert the list of predicted classes to a tensor
    pred_array = torch.tensor(predicted_classes)

    # Calculate accuracy
    accuracy = (pred_array == gt_tensor).float().mean().item() * 100

    return accuracy


def crop_unused_roi_parts(
    roi_images: List[np.ndarray], bag_of_tiles_dataset: InferenceBagOfTilesDataset
) -> List[np.ndarray]:
    """
    Crops the unused parts of the regions of interest (ROIs) in the given list of images based on
    tiles number per rois heights and widths.

    :arg roi_images: List of images representing regions of interest (ROIs).

    :arg bag_of_tiles_dataset: An instance of InferenceBagOfTilesDataset containing information
    about ROIs.

    :return: List of cropped ROI images where unused parts have been removed based on
    tile dimensions.
    """

    if isinstance(bag_of_tiles_dataset.file_unit_strategy, SlideFileUnitStrategy):
        tile_height = bag_of_tiles_dataset.file_unit_strategy.tiling_meta.height
        tile_width = bag_of_tiles_dataset.file_unit_strategy.tiling_meta.width

        for i in range(len(roi_images)):
            roi_height = roi_images[i].shape[1]
            roi_width = roi_images[i].shape[0]
            new_roi_height = (roi_height // tile_height) * tile_height
            new_roi_width = (roi_width // tile_width) * tile_width
            roi_images[i] = roi_images[i][0:new_roi_width, 0:new_roi_height, :]

        return roi_images


def visualize_outputs(
    att_maps: Dict, roi_images: List[np.ndarray], save_path: str, roi_ids: List
) -> None:
    """
    Visualizes the heatmap overlay on the respective regions of interest (ROIs) from images.

    :arg att_maps: Dictionary of attention maps(heatmaps) to overlay.

    :arg roi_images: List of images representing regions of interest (ROIs).

    :arg save_path: Path to save the visualizations.

    :arg roi_ids: List of IDs corresponding to each ROI.

    :return: None.
    """
    for i in range(len(roi_images)):
        img = roi_images[i]
        att_map = att_maps[i]

        roi_save_path = save_path + "/roi_" + roi_ids[i]
        if not os.path.exists(roi_save_path):
            os.mkdir(roi_save_path)

        # Resize the heatmap to match the image size
        heatmap_resized = cv2.resize(att_map, (img.shape[1], img.shape[0]))

        # Normalize the heatmap values between 0 and 1
        heatmap_normalized = cv2.normalize(
            heatmap_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply a color map (jet colormap is commonly used for heatmaps)
        heatmap_colormap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Blend the heatmap with the original image
        heatmap_on_image = cv2.addWeighted(img, 0.7, heatmap_colormap, 0.3, 0)

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Heatmap Overlay on Image")
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap_resized, cmap="jet")
        plt.colorbar()
        plt.axis("off")
        plt.title("Heatmap Color Bar")
        plt.savefig(roi_save_path + "/overlay_image.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(att_map)
        plt.tight_layout()
        plt.savefig(roi_save_path + "/heatmap.png")
        plt.imsave(roi_save_path + "/roi.png", img)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIL inference script.")
    parser.add_argument("--config-path", type=str, help="Path to the configuration " "file")
    parser.add_argument("--weights-path", type=str, help="Path to the weights")
    parser.add_argument("--images-folder", type=str, help="Folder containing images")
    parser.add_argument("--number-of-iterations", type=int, help="Number of " "iterations")
    parser.add_argument("--wsi-extension", type=str, help="Extension of whole slide " "image file")
    parser.add_argument("--save-path", type=str, help="Path where visualizations will " "be saved")
    parser.add_argument("--inference-type", type=str, help="It can be regression or classification")

    args = parser.parse_args()
    images_folder: str = args.images_folder
    config_path: str = args.config_path
    weights_path: str = args.weights_path
    num_iter: int = args.number_of_iterations
    wsi_extension: str = args.wsi_extension
    save_path: str = args.save_path
    inference_type: str = args.inference_type

    assert inference_type in ["regression", "classification"]

    bag_of_tiles_dataset = instantiate_bag_of_tiles(config_path, images_folder, wsi_extension)
    model = fetch_model(config_path, weights_path)
    att_maps, count_maps = initialize_attention_and_count_maps(bag_of_tiles_dataset)
    activation = get_activations(config_path)["BagDecoder"]["BagLabel"]

    roi_bboxes = get_rois_bboxes(bag_of_tiles_dataset)
    roi_ids = get_roi_ids(bag_of_tiles_dataset)
    bag_type = bag_of_tiles_dataset.bag_type
    label_type = "roi" if isinstance(bag_type, RoiBagOfTiles) else "slide"

    labels_path = get_labels_filepath(config_path, images_folder)

    preds = {i: [] for i in range(len(bag_of_tiles_dataset))}

    for _ in range(num_iter):
        batch, tiles_bboxes, bag_indices, bag_sizes = create_batch(bag_of_tiles_dataset)

        split_tiles = split_tile_bboxes_per_roi(tiles_bboxes, bag_of_tiles_dataset, bag_indices)
        tiles_top_left = translate_tile_top_left_coordinates(split_tiles, roi_bboxes, bag_type)
        count_maps = update_count_map(tiles_top_left, count_maps)
        pred_logits, att_weights = run_mil_inference(model, batch)
        attentions_per_bag = split_attention_per_bag(att_weights, bag_sizes)

        predictions = activation(pred_logits)

        for i, (prediction, attention) in enumerate(zip(predictions, attentions_per_bag)):
            preds[i].append(prediction.unsqueeze(dim=1))
            att_maps = update_attention_map(att_maps, attention, tiles_top_left, bag_indices[i])

    # Finalize predictions
    for i in range(len(bag_of_tiles_dataset)):
        preds[i] = torch.cat(preds[i], dim=1).mean(dim=1)

    labels = get_labels(bag_of_tiles_dataset, labels_path, label_type)
    save_predictions(roi_ids, preds, save_path, labels, inference_type)

    if isinstance(bag_of_tiles_dataset.file_unit_strategy, SlideFileUnitStrategy):
        if isinstance(bag_type, RoiBagOfTiles):
            regions = read_roi_regions(bag_of_tiles_dataset.file_unit_strategy.slides)
            cropped_unused_roi_parts = crop_unused_roi_parts(regions, bag_of_tiles_dataset)
            visualize_outputs(att_maps, cropped_unused_roi_parts, save_path, roi_ids)
