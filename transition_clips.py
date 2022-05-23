import os
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip

TRIGGER_PERCENTILE = 10
CLIP_DIRECTORY = "transition_clips"
MIN_DURATION = 2.0
DELTA = 0.5  # length of subclips to compare over (in seconds)


def grayscale_and_downsample(arr, downsample_factor=4):
    arr = arr[0::downsample_factor, 0::downsample_factor]
    return np.dot(
        arr, [0.2989, 0.5870, 0.1140]
    )  # magic numbers for converting RGB to grayscale


def compare_frames(clip, t1, t2):
    frame1 = grayscale_and_downsample(clip.get_frame(t1))
    frame2 = grayscale_and_downsample(clip.get_frame(t2))
    diff = frame2 - frame1
    l_0 = np.linalg.norm(diff.ravel(), ord=0)
    return l_0 / diff.size


def get_frame_by_frame_diffs(clip):
    comparisons = []
    time_range = np.arange(0, clip.duration - DELTA, DELTA)
    with tqdm(total=len(time_range)) as pbar:
        for t1 in time_range:
            t2 = t1 + DELTA
            diff = compare_frames(clip, t1, t2)
            comparisons.append(diff)
            pbar.update()
    return comparisons


def get_transition_peak(frame_diffs):
    threshold = np.percentile(frame_diffs, TRIGGER_PERCENTILE)
    top_indices = [idx for idx, diff in enumerate(frame_diffs) if diff <= threshold]
    return top_indices


def get_transition_intervals(frame_diffs):
    top_indices = get_transition_peak(frame_diffs)
    median = np.percentile(frame_diffs, 50)
    transition_slices = []
    for top_idx in top_indices:
        # check if already in a slice
        for start_idx, end_idx in transition_slices:
            if top_idx >= start_idx and top_idx <= end_idx:
                break
        else:
            # populate new slice
            left_idx = top_idx - 1
            while frame_diffs[left_idx] < median:
                left_idx -= 1
            right_idx = top_idx + 1
            while frame_diffs[right_idx] < median:
                right_idx += 1
            transition_slices.append((left_idx, right_idx))
    return transition_slices


def get_transition_events(frame_diffs):
    transition_slices = get_transition_intervals(frame_diffs)
    slices_in_seconds = [
        (start_idx * DELTA, end_idx * DELTA) for start_idx, end_idx in transition_slices
    ]
    return [
        (t_start, t_end)
        for t_start, t_end in slices_in_seconds
        if t_end - t_start > MIN_DURATION
    ]


def save_video_clip(video, save_filename, t_start, t_end):
    """
    Takes start and end times (in seconds) for video and returns clipped
    video and saves it locally
    Times can be expressed in seconds (15.35), in (min, sec), in (hour, min, sec), or as a string: ‘01:03:05.35’
    """
    clip = video.subclip(t_start, t_end)
    clip.write_videofile(save_filename, codec="libx264", audio_codec="aac", logger=None)


def generate_transition_clips(raw_filepath):
    with VideoFileClip(raw_filepath) as clip:
        if clip.rotation == 90:
            clip = clip.resize(clip.size[::-1])
            clip.rotation = 0
        print("Identifying transition events - this may take a while...")
        frame_diffs = get_frame_by_frame_diffs(clip)
        transitions = get_transition_events(frame_diffs)
        print("Saving transition clips...")
        os.makedirs(CLIP_DIRECTORY, exist_ok=True)
        with tqdm(total=len(transitions)) as pbar:
            for idx, (t_start, t_end) in enumerate(transitions):
                clip_number = str(idx).zfill(
                    len(str(len(transitions)))
                )  # makes a zero-padded string of the clip index (e.g. '3' -> '003')
                pbar.set_description(f"Saving clip #{clip_number}")
                save_video_clip(
                    clip, f"{CLIP_DIRECTORY}/trans_{clip_number}.MOV", t_start, t_end
                )
                pbar.update()
