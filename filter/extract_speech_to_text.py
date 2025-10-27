'''
This script extracts speech-to-text transcriptions from videos using Whisper.

example usage:
python extract_speech_to_text.py --input_folder filtered_videos --num_workers 4
'''

import argparse
import os
import json
import torch
import whisper
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def extract_audio(vid_path, audio_folder):
    """Extract audio from video and save as a .wav file in the audio folder."""
    try:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        aud_path = os.path.join(audio_folder, f"{base_name}_audio.wav")
        if not os.path.exists(aud_path):
            os.makedirs(audio_folder, exist_ok=True)
            os.system(f'ffmpeg -y -i "{vid_path}" "{aud_path}"')
        return aud_path
    except Exception as e:
        print(f'Error extracting audio from {vid_path}: {e}')
        return None

def transcribe_audio(audio_path, model, output_folder):
    """Transcribe audio using Whisper and save to JSON."""
    try:
        result = model.transcribe(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0].replace("_audio", "")
        json_path = os.path.join(output_folder, f"{base_name}_speech.json")
        os.makedirs(output_folder, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved transcription to {json_path}")
    except Exception as e:
        raise RuntimeError(f"Error transcribing {audio_path}: {e}")

def process_video(vid_path, model_path, output_folder, audio_folder, gpu_id, failed_log):
    """Extract audio and transcribe using Whisper, record failures."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = whisper.load_model(model_path).to(device)
    audio_path = extract_audio(vid_path, audio_folder)
    
    if audio_path:
        try:
            transcribe_audio(audio_path, model, output_folder)
        except Exception as e:
            group_name = os.path.basename(os.path.dirname(vid_path))
            with open(failed_log, 'a') as log_file:
                log_file.write(f"{group_name}:\n{os.path.basename(vid_path)}\n")
            print(f"Failed to transcribe {vid_path}: {e}")

def process_videos(args):
    """Process all videos in the input folder recursively by groups."""
    video_groups = glob(os.path.join(args.input_folder, 'group_*'))
    if not video_groups:
        print("No video groups found in the input folder.")
        return

    # Initialize failed log
    failed_log = os.path.join(args.output_folder or args.input_folder, "failed_videos.txt")
    if os.path.exists(failed_log):
        os.remove(failed_log)

    for group in video_groups:
        videos = glob(os.path.join(group, '*.mp4'))
        if not videos:
            print(f"No videos found in {group}.")
            continue

        print(f"Processing group: {os.path.basename(group)}")

        speech_folder = os.path.join(args.output_folder or f"{group}_speech")
        audio_folder = os.path.join(f"{group}_audio")
        available_gpus = [2, 3]

        with Pool(args.num_workers) as pool:
            tasks = [
                (vid, "large-v3", speech_folder, audio_folder, available_gpus[i % len(available_gpus)], failed_log)
                for i, vid in enumerate(videos)
            ]
            for _ in tqdm(pool.imap_unordered(process_video_star, tasks), total=len(tasks)):
                pass

    print("Processing complete. Check the failed video log if applicable:", failed_log)

def process_video_star(args):
    """Helper to unpack arguments for starmap."""
    return process_video(*args)

def get_args():
    parser = argparse.ArgumentParser(description="Extract speech-to-text from videos in multiple groups using Whisper.")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing group_* subfolders with .mp4 videos.")
    parser.add_argument('--output_folder', type=str, default=None, help="Folder to save transcribed speech JSON files.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    process_videos(args)
