import os
import subprocess

from tqdm import tqdm

def is_mislabeled_wav(file_path):
    """Checks if a .wav file is actually an AAC/MP4 file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True,
            text=True
        )
        format_name = result.stdout.strip()
        return format_name in ["mov", "mp4", "3gp", "3g2"]  # Common mislabeled formats
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False

def convert_to_wav(input_file, output_file):
    """Converts an AAC/MP4-based file to proper WAV."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "44100", output_file],
            check=True
        )
        print(f"Converted: {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

def process_directory(directory, new_dir):
    """Scans directory, detects mislabeled WAV files, and converts them."""
    os.makedirs(new_dir, exist_ok=True)
    for filename in tqdm(os.listdir(directory)):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(directory, filename)
            # if is_mislabeled_wav(file_path):
            new_file_path = os.path.join(new_dir, filename)
            convert_to_wav(file_path, new_file_path)

if __name__ == "__main__":
    folder_path = r'..\datasets\music'
    new_dir = r'..\datasets\bg_music_fixed'
    if os.path.isdir(folder_path):
        process_directory(folder_path, new_dir)
        print("Processing completed.")
    else:
        print("Invalid directory path!")
