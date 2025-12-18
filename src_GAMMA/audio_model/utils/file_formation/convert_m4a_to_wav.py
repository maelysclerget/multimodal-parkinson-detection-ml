
"""
Convert m4a files to wav format (mono) using ffmpeg.

This script provides utilities to batch-convert .m4a audio files to .wav format (mono channel) using ffmpeg, replicating the approach from m4a_full_pipeline_spectrograms.py. It is intended for preprocessing large audio datasets for downstream machine learning or signal processing tasks.

Features:
- Checks for ffmpeg availability in the system PATH.
- Converts individual or batches of .m4a files to .wav (mono) using ffmpeg.
- Skips files that have already been converted (unless specified otherwise).
- Handles errors and prints informative messages.
- Can be run as a script with command-line arguments for input/output directories, ffmpeg binary, and file patterns.

Expected usage:
    python convert_m4a_to_wav.py <input_dir> <output_dir> [--ffmpeg-bin ffmpeg] [--pattern *.m4a] [--no-skip-existing]
"""

import subprocess
import shutil
import os
import glob
from tqdm import tqdm
import sys
from pathlib import Path


def detect_ffmpeg(cmd="ffmpeg"):
    """
    Check if ffmpeg is available in the system PATH.

    Args:
        cmd (str): Name or path of the ffmpeg executable (default: 'ffmpeg').

    Returns:
        bool: True if ffmpeg is found, False otherwise.
    """
    return shutil.which(cmd) is not None


def convert_m4a_to_wav(src_path, dst_path, ffmpeg_bin="ffmpeg"):
    """
    Convert a single m4a file to mono wav using ffmpeg.

    Replicates: ffmpeg -y -i <src> -ac 1 <dst>

    Args:
        src_path (str or Path): Path to input .m4a file.
        dst_path (str or Path): Path to output .wav file.
        ffmpeg_bin (str): ffmpeg executable name or path (default: 'ffmpeg').

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Run ffmpeg: -y (overwrite), -i (input), -ac 1 (mono audio channel)
        cmd = [ffmpeg_bin, "-y", "-i", str(src_path), "-ac", "1", str(dst_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if proc.returncode != 0:
            error_msg = proc.stderr.splitlines()[-1] if proc.stderr else 'unknown error'
            print(f"Error converting {os.path.basename(src_path)}: {error_msg}")
            return False
        
        if not os.path.exists(dst_path):
            print(f"Error: Conversion produced no output file: {dst_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error converting {src_path}: {e}")
        return False


def batch_convert_m4a_to_wav(input_dir, output_dir, pattern="*.m4a", ffmpeg_bin="ffmpeg", skip_existing=True):
    """
    Convert all .m4a files in a directory to mono .wav files using ffmpeg.

    Args:
        input_dir (str or Path): Directory containing .m4a files.
        output_dir (str or Path): Directory to save .wav files. Will be created if it does not exist.
        pattern (str): Glob pattern for finding .m4a files (default: '*.m4a').
        ffmpeg_bin (str): ffmpeg executable name or path (default: 'ffmpeg').
        skip_existing (bool): If True, skip conversion if .wav file already exists (default: True).

    Returns:
        None
    """
    # Check ffmpeg availability
    if not detect_ffmpeg(ffmpeg_bin):
        print(f"ERROR: ffmpeg not found (bin='{ffmpeg_bin}')")
        print("Please install ffmpeg or specify correct path with --ffmpeg-bin")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all m4a files
    search_path = os.path.join(input_dir, pattern)
    m4a_files = glob.glob(search_path)
    
    print(f"Found {len(m4a_files)} m4a files in {input_dir}")
    print(f"Converting to mono wav format in {output_dir}")
    print(f"Using ffmpeg: {ffmpeg_bin}")
    
    if not m4a_files:
        print("No m4a files found!")
        return
    
    # Convert each file
    successful = 0
    failed = 0
    skipped = 0
    
    for m4a_path in tqdm(m4a_files, desc="Converting"):
        # Generate output filename
        basename = os.path.basename(m4a_path)
        wav_filename = basename.replace('.m4a', '.wav')
        wav_path = os.path.join(output_dir, wav_filename)
        
        # Skip if already exists
        if skip_existing and os.path.exists(wav_path):
            skipped += 1
            continue
        
        # Convert
        if convert_m4a_to_wav(m4a_path, wav_path, ffmpeg_bin):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"Successful: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert m4a files to mono wav format using ffmpeg"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing m4a files")
    parser.add_argument("output_dir", type=str, help="Directory to save wav files")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable path/name")
    parser.add_argument("--no-skip-existing", action="store_true", 
                       help="Re-convert files even if wav already exists")
    parser.add_argument("--pattern", default="*.m4a", help="Glob pattern for m4a files")
    
    args = parser.parse_args()
    
    batch_convert_m4a_to_wav(
        args.input_dir, 
        args.output_dir, 
        pattern=args.pattern,
        ffmpeg_bin=args.ffmpeg_bin,
        skip_existing=not args.no_skip_existing
    )
