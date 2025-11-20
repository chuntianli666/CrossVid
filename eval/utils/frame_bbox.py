import json
import os

import cv2
import numpy as np
import base64
from typing import Union, Tuple, List, Dict


def process_frame(
        frame_path: str,
        length: int,
        encode: bool = False,
        bboxes: Dict[int, Dict[str, int]] = None
) -> Tuple[Union[str, np.ndarray], Dict[int, Dict[str, int]]]:
    """
    Process a video frame: resize image, transform bounding boxes, with optional encoding and visualization.

    Args:
        frame_path (str): Path to the video frame file.
        length (int): Target scaling length for the longer dimension.
        encode (bool, optional): Whether to return a Base64-encoded image. Defaults to False.
        bboxes (Dict[int, Dict[str, int]], optional): Dictionary of bounding boxes indexed by object ID.
                                                      Each bbox is a dict with keys: 'xtl', 'ytl', 'xbr', 'ybr'.
                                                      Can contain None values for missing objects.

    Returns:
        Tuple[Union[str, np.ndarray], Dict[int, Dict[str, int]]]:
            - processed_image: Base64-encoded string (if encode=True) or numpy array (if encode=False)
            - scaled_bboxes: Dictionary of scaled bounding boxes with same structure as input

    Raises:
        FileNotFoundError: If the frame image cannot be read.

    Example:
        >>> bbox = {0: {'xtl': 100, 'ytl': 100, 'xbr': 200, 'ybr': 200}}
        >>> img, scaled_bbox = process_frame('frame.jpg', length=640, encode=True, bboxes=bbox)
    """
    # Read frame
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {frame_path}")

    # Original size
    h, w = img.shape[:2]
    max_dim = max(h, w)

    # Calculate ratio
    scale = length / max_dim if max_dim > length else 1.0

    # Resize
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    # Resize bbox
    scaled_bboxes = {}
    if bboxes is not None:
        for idx, bbox in bboxes.items():
            if bbox is None:
                scaled_bboxes[idx] = None
            else:
                scaled = {
                    'xtl': int(bbox['xtl'] * scale),
                    'ytl': int(bbox['ytl'] * scale),
                    'xbr': int(bbox['xbr'] * scale),
                    'ybr': int(bbox['ybr'] * scale)
                }
                scaled_bboxes[idx] = scaled

    # Output
    if encode:
        # encode
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image, scaled_bboxes
    else:
        return img, scaled_bboxes

def process_frames(frame_dir, n_frames, bboxes, length=640, encode=True):
    """
    Process multiple frames uniformly sampled from a directory with associated bounding boxes.

    Args:
        frame_dir (str): Directory containing frame images.
        n_frames (int): Number of frames to sample uniformly from the directory.
        bboxes (Dict[int, Dict[str, Dict]]): Nested dictionary of bounding boxes.
                                             Structure: {object_id: {frame_index: bbox_dict or None}}
                                             where frame_index is a string representation of the frame number.
        length (int, optional): Target scaling length for the longer dimension. Defaults to 640.
        encode (bool, optional): Whether to return Base64-encoded images. Defaults to True.

    Returns:
        Tuple[List, List]:
            - processed_frames: List of processed frames (Base64 strings or numpy arrays)
            - processed_bboxes: List of dictionaries containing scaled bounding boxes for each frame

    Example:
        >>> bboxes = {
        ...     23: {'0': {'xtl': 10, 'ytl': 10, 'xbr': 50, 'ybr': 50}, '5': None}
        ... }
        >>> frames, boxes = process_frames('frames/', n_frames=8, bboxes=bboxes)
    """
    frames = sorted(os.listdir(frame_dir))
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    frame_paths = [os.path.join(frame_dir, frames[i]) for i in indices]
    processed_frames = []
    processed_bboxes = []
    for frame_path, index in zip(frame_paths, indices):
        input_bboxes = {}
        for idx, bbox in bboxes.items():
            if bbox is None:
                input_bboxes[idx] = None
            else:
                input_bboxes[idx] = bbox.get(str(index))
        processed_frame, processed_bbox = process_frame(frame_path, length, encode, input_bboxes)
        processed_frames.append(processed_frame)
        processed_bboxes.append(processed_bbox)
    return processed_frames, processed_bboxes


if __name__ == "__main__":
    # Example
    frame_dir = "uav/frames/1/22-1"
    with open("uav/bbox/1/22.json", "r") as f:
        bboxes = json.load(f)
    ids = [18, 37, 46]
    input_bboxes = {}
    for bbox in bboxes:
        if bbox["id"] in ids:
            input_bboxes[bbox["id"]] = bbox["bbox"]
    process_frames(frame_dir, 8, input_bboxes, 640)
