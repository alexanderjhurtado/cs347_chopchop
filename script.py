import os
import sys
import numpy as np
from tqdm import tqdm
import moviepy.editor as mp
from moviepy.editor import VideoFileClip

DELTA = 0.01  # length of subclips to sample over (in seconds)
TRIGGER_PERCENTILE = 97
CLIP_DIRECTORY = "action_clips"
MIN_CLIP_DURATION = 0.25  # minimum length of action clip


def get_clip_sample(clip, time_idx):
    """
    This method takes the original clip,
    grabs the subclip of length DELTA (in seconds) starting at time `time_idx` (seconds from start),
    and returns an array representing the loudness of that subclip.

    The array can be 2D if the audio is stereo (vs mono).
    The length of the array is determined by DELTA and the FPS rate of to_soundarray()
    """
    return clip.audio.subclip(time_idx, time_idx + DELTA).to_soundarray(fps=44100)


def get_average_volume(sound_arr):
    """
    Returns the average volume of the audio clip given by the loudness array

    The array can be 1D or 2D without issue
    """
    return np.sqrt(((1.0 * sound_arr) ** 2).mean())


def get_sample_volume(clip, time_idx):
    """
    Grabs the average volume of the subclip of length DELTA starting at time `time_idx`
    """
    sample = get_clip_sample(clip, time_idx)
    return get_average_volume(sample)


def get_volume_array(clip):
    """
    Takes as many samples of length DELTA across the duration of the clip
    and returns an array of those samples' average volumes
    """
    return [
        get_sample_volume(clip, idx) for idx in np.arange(0, clip.audio.duration, DELTA)
    ]


def get_action_peaks(volume_arr):
    """
    Return indices where volume is above the 'TRIGGER_PERCENTILE'th percentile
    """
    top_perc = np.percentile(volume_arr, TRIGGER_PERCENTILE)
    top_indices = [idx for idx, vol in enumerate(volume_arr) if vol >= top_perc]
    return top_indices


def get_action_intervals(volume_arr):
    """
    Return intervals (in index) of action events
    We define an event as any continuous stretch of time around the action peak
    where the volume remains above the median volume
    """
    top_indices = get_action_peaks(volume_arr)
    median = np.percentile(volume_arr, 50)

    event_slices = []
    for top_idx in top_indices:
        # check if already in a slice
        for start_idx, end_idx in event_slices:
            if top_idx >= start_idx and top_idx <= end_idx:
                break
        else:
            # populate new slice
            left_idx = top_idx - 1
            while volume_arr[left_idx] > median:
                left_idx -= 1
            right_idx = top_idx + 1
            while volume_arr[right_idx] > median:
                right_idx += 1
            event_slices.append((left_idx, right_idx))
    return event_slices


def get_action_events(volume_arr):
    """
    Return intervals (in seconds) of action events
    """
    event_slices = get_action_intervals(volume_arr)
    slices_in_seconds = [
        (start_idx * DELTA, end_idx * DELTA) for start_idx, end_idx in event_slices
    ]
    return [
        (t_start, t_end)
        for t_start, t_end in slices_in_seconds
        if t_end - t_start > MIN_CLIP_DURATION
    ]


def save_video_clip(video, save_filename, t_start, t_end):
    """
    Takes start and end times (in seconds) for video and returns clipped
    video and saves it locally
    Times can be expressed in seconds (15.35), in (min, sec), in (hour, min, sec), or as a string: ‘01:03:05.35’
    """
    clip = video.subclip(t_start, t_end)
    if clip.rotation == 90:
        clip = clip.resize(clip.size[::-1])
        clip.rotation = 0
    clip.write_videofile(save_filename, codec="libx264", audio_codec="aac", logger=None)


def generate_action_clips(raw_filepath):
    """
    Identify and save action clips from the video of the given filepath
    """
    with VideoFileClip(raw_filepath) as clip:
        print("Identifying action events...")
        volumes = get_volume_array(clip)
        events = get_action_events(volumes)
        print("Saving action clips...")
        os.makedirs(CLIP_DIRECTORY, exist_ok=True)
        with tqdm(total=len(events)) as pbar:
            for idx, (t_start, t_end) in enumerate(events):
                clip_number = str(idx).zfill(
                    len(str(len(events)))
                )  # makes a zero-padded string of the clip index (e.g. '3' -> '003')
                pbar.set_description(f"Saving clip #{clip_number}")
                save_video_clip(
                    clip, f"{CLIP_DIRECTORY}/clip_{clip_number}.MOV", t_start, t_end
                )
                pbar.update()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Specify the raw video's filename (e.g. `python3 script.py raw_vid.MOV`)"
        )
    generate_action_clips(sys.argv[1])
