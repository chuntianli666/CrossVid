import cv2
import numpy as np
import base64
import bisect
import math
from decord import VideoReader, cpu


def allocate_frames(durations, total_frames):
    """
    Allocate a total number of frames proportionally across multiple intervals based on their durations.

    Uses the largest remainder method to ensure the sum of allocated frames equals total_frames exactly,
    while maintaining proportionality to each interval's duration.

    Args:
        durations (list): List of duration values for each interval.
        total_frames (int): Total number of frames to allocate across all intervals.

    Returns:
        list: List of integers representing the number of frames allocated to each interval.
              Returns all zeros if total_duration is 0.

    Example:
        >>> allocate_frames([3.0, 2.0, 1.0], 10)
        [5, 3, 2]  # Proportional to 3:2:1 ratio
    """
    total_duration = sum(durations)
    if total_duration == 0:
        return [0] * len(durations)

    # Calculate ideal fractional allocation
    ideal_frames = [(d / total_duration) * total_frames for d in durations]
    integer_parts = [math.floor(x) for x in ideal_frames]
    fractions = [x - math.floor(x) for x in ideal_frames]

    total_integer = sum(integer_parts)
    remaining = total_frames - total_integer

    if remaining < 0:
        return integer_parts

    # Distribute remaining frames to intervals with largest fractional parts
    sorted_indices = sorted(range(len(fractions)), key=lambda i: -fractions[i])
    for i in range(remaining):
        idx = sorted_indices[i]
        integer_parts[idx] += 1

    return integer_parts


def process_video(input_path: str, n_frames: int, intervals: list, max_length: int = None, encode: bool = True) -> (
        list, list, list):
    """
    Process a video by extracting frames from specified time intervals with proportional sampling.

    This function supports extracting frames from multiple time intervals (continuous or discontinuous),
    allocating frames proportionally based on interval durations, and optionally resizing and encoding
    the extracted frames.

    Args:
        input_path (str): Path to the input video file.
        n_frames (int): Total number of frames to extract across all intervals.
        intervals (list): List of time intervals. Each interval can be:
                         - A single segment: [start_time, end_time]
                         - Multiple segments: [[start1, end1], [start2, end2], ...]
                         Times are in seconds.
        max_length (int, optional): Maximum dimension (width or height) for frame resizing.
                                   If None, frames are not resized. Defaults to None.
        encode (bool, optional): If True, encode frames as base64-encoded JPEG strings.
                                If False, return frames as numpy arrays. Defaults to True.

    Returns:
        tuple: A tuple containing three lists:
            - all_processed (list of lists): Processed frames for each interval. Each element is either
                                            a list of base64 strings (if encode=True) or a numpy array.
            - all_fps (list of floats): Effective FPS for each interval after sampling.
            - all_time_stamps (list of lists): Timestamps (in seconds) of extracted frames for each interval.

    Raises:
        ValueError: If the video file cannot be opened or is invalid.

    Example:
        >>> # Extract 20 frames from two intervals
        >>> frames, fps, timestamps = process_video(
        ...     "video.mp4",
        ...     n_frames=20,
        ...     intervals=[[1, 4], [[6, 8], [10, 12]]],  # 3s + 4s intervals
        ...     max_length=640,
        ...     encode=True
        ... )
        >>> # First interval gets ~8 frames (3/7 * 20), second gets ~12 frames (4/7 * 20)

    Notes:
        - Frames are allocated proportionally to interval durations using allocate_frames().
        - Within each interval, frames are sampled uniformly across the combined duration.
        - Invalid intervals (start >= end) are automatically filtered out.
        - If an interval receives 0 frames, empty lists are returned for that interval.
        - Frame aspect ratio is preserved during resizing.
    """
    video_path = input_path

    # Open video file
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except:
        raise ValueError("Can not read video")

    total_frames_video = len(vr)

    # Calculate or estimate original FPS
    try:
        original_fps = vr.get_avg_fps()
    except AttributeError:
        # Fallback: estimate FPS from frame timestamps
        frame_indices = list(range(min(100, total_frames_video)))
        if len(frame_indices) < 2:
            original_fps = 1.0
        else:
            try:
                timestamps = vr.get_frame_timestamp(frame_indices)[:, 0]
                avg_duration = np.mean(np.diff(timestamps))
                original_fps = 1.0 / avg_duration if avg_duration != 0 else 0
            except:
                original_fps = 30  # Default FPS fallback

    if total_frames_video == 0 or original_fps == 0:
        raise ValueError("Invalid video")

    # Parse and validate intervals
    parsed_intervals = []
    for interval in intervals:
        # Check if interval contains multiple segments
        if isinstance(interval[0], (list, tuple)):
            segs = interval
        else:
            segs = [interval]

        # Validate each segment
        valid_segs = []
        for seg in segs:
            if len(seg) != 2 or seg[0] >= seg[1]:
                continue
            valid_segs.append((float(seg[0]), float(seg[1])))

        if valid_segs:
            parsed_intervals.append(valid_segs)

    if not parsed_intervals:
        return [], [], []

    # Calculate duration for each interval group
    durations = []
    for segs in parsed_intervals:
        duration = sum(end - start for start, end in segs)
        durations.append(duration)

    total_duration = sum(durations)
    if total_duration == 0:
        return [], [], []

    # Allocate frames proportionally to each interval
    allocated = allocate_frames(durations, n_frames)

    all_processed = []
    all_fps = []
    all_time_stamps = []
    video_duration = total_frames_video / original_fps if original_fps > 0 else 0

    # Process each interval group
    for i, segs in enumerate(parsed_intervals):
        k = allocated[i]  # Number of frames for this interval
        if k <= 0:
            all_processed.append([])
            all_fps.append(0.0)
            all_time_stamps.append([])
            continue

        # Build cumulative duration map for multi-segment intervals
        seg_durations = [end - start for start, end in segs]
        cum_dur = [0.0]
        for d in seg_durations:
            cum_dur.append(cum_dur[-1] + d)
        total_seg_duration = cum_dur[-1]

        # Sample uniformly in virtual time space
        virtual_times = np.linspace(0, total_seg_duration, num=k) if k > 1 else [0.0]

        # Map virtual times to actual video timestamps
        actual_times = []
        for t in virtual_times:
            # Find which segment this virtual time falls into
            idx = bisect.bisect_right(cum_dur, t) - 1
            idx = max(0, min(idx, len(segs) - 1))
            seg_start, seg_end = segs[idx]
            t_in_seg = t - cum_dur[idx]
            actual_time = seg_start + t_in_seg
            actual_time = max(0.0, min(actual_time, video_duration))
            actual_times.append(actual_time)

        # Convert times to frame indices
        frame_indices = [int(t * original_fps) for t in actual_times]
        frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames_video]
        frame_indices = sorted(list(set(frame_indices)))  # Remove duplicates

        if not frame_indices:
            all_processed.append([])
            all_fps.append(0.0)
            all_time_stamps.append([])
            continue

        # Get actual timestamps for selected frames
        try:
            time_stamps = vr.get_frame_timestamp(frame_indices)[:, 0].tolist()
        except:
            time_stamps = [idx / original_fps for idx in frame_indices]

        # Read frames from video
        try:
            frames = vr.get_batch(frame_indices).asnumpy()
        except:
            frames = []

        # Process each frame: resize and optionally encode
        processed = []
        for frame in frames:
            h, w = frame.shape[:2]
            # Resize if max_length is specified
            if max_length and max(h, w) > max_length:
                scale = max_length / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized = frame

            # Encode to base64 JPEG or keep as numpy array
            if encode:
                bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode('.jpg', bgr_frame)
                if success:
                    processed.append(base64.b64encode(buffer).decode('utf-8'))
            else:
                processed.append(resized)

        if not encode:
            processed = np.array(processed)

        # Calculate effective FPS for this interval
        interval_duration = sum(end - start for start, end in segs)
        new_fps = len(processed) / interval_duration if interval_duration > 0 else 0

        all_processed.append(processed)
        all_fps.append(new_fps)
        all_time_stamps.append(time_stamps)

    return all_processed, all_fps, all_time_stamps

