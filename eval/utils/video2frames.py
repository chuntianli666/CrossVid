import cv2
import numpy as np
import base64
from decord import VideoReader, cpu


def process_video(input_path: str, n_frames: int, max_length: int = None, encode: bool = True) -> (list, float, list):
    """
        Process a video file by extracting uniformly sampled frames with optional resizing and encoding.

        Args:
            input_path (str): Path to the input video file.
            n_frames (int): Number of frames to sample from the video.
            max_length (int, optional): Maximum dimension (width or height) for frame resizing.
                                       If None, frames are not resized. Defaults to None.
            encode (bool, optional): If True, encode frames as base64-encoded JPEG strings.
                                    If False, return frames as numpy arrays. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - processed (list): List of processed frames. If encode=True, contains base64-encoded
                                  JPEG strings; if encode=False, contains a numpy array of frames.
                - new_fps (float): Effective FPS after sampling (sampled_frames / video_duration).
                - time_stamps (list): List of timestamps (in seconds) for each sampled frame.

        Raises:
            ValueError: If the video file cannot be opened, is invalid, or frame reading fails.

        Notes:
            - Frames are sampled uniformly across the entire video duration.
            - If the video has fewer frames than n_frames, all available frames are used.
            - When resizing, aspect ratio is preserved and the larger dimension is scaled to max_length.
            - FPS is calculated automatically; if unavailable, it's estimated from frame timestamps
              or defaults to 30.
        """
    video_path = input_path

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except:
        raise ValueError("Can not open video file")

    total_frames = len(vr)

    # Calculate original FPS
    try:
        original_fps = vr.get_avg_fps()
    except AttributeError:
        # Calculate FPS through timestamps
        frame_indices = list(range(min(100, total_frames)))  # Avg time span of 100 frames
        if len(frame_indices) < 2:  # single-frame video
            original_fps = 1.0
        else:
            try:
                timestamps = vr.get_frame_timestamp(frame_indices)[:, 0]
                avg_duration = np.mean(np.diff(timestamps))
                original_fps = 1.0 / avg_duration if avg_duration != 0 else 0
            except:
                original_fps = 30  # default FPS

    if total_frames == 0 or original_fps == 0:
        raise ValueError("Invalid video file")

    # Calculate frame counts we need
    n_frames = min(n_frames, total_frames) if total_frames > 0 else 0

    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num=n_frames, dtype=np.int32)
    indices = np.unique(np.clip(indices, 0, total_frames - 1))

    # Get timestamps (s)
    try:
        time_stamps = vr.get_frame_timestamp(indices.tolist())[:, 0].tolist()
    except Exception as e:
        time_stamps = [(i / original_fps) for i in indices] if original_fps > 0 else []

    # Read frame
    try:
        frames = vr.get_batch(indices).asnumpy()
    except:
        raise ValueError("Read frames error")

    processed = []
    for frame in frames:
        # Resize
        h, w = frame.shape[:2]
        if max_length is not None and max(h, w) > max_length:
            scale = max_length / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = frame

        # encoding
        if encode:
            bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.jpg', bgr_frame)
            if success:
                processed.append(base64.b64encode(buffer).decode('utf-8'))
        else:
            processed.append(resized)

    if not encode:
        processed = np.array(processed)

    # Calculate current FPS after sampling
    duration = total_frames / original_fps
    new_fps = len(processed) / duration if duration > 0 else 0

    return processed, new_fps, time_stamps
