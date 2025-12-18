
"""
visualize_npy.py
----------------
Script to visualize .npy waveform files.
Shows waveform plot and prints statistics for the waveform.

Functions:
    visualize_waveform(npy_path, sample_rate=44100):
        Loads a .npy waveform file, prints statistics, and saves/plots the waveform.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def visualize_waveform(npy_path, sample_rate=44100):
    """
    Load and visualize a waveform from a .npy file. Prints statistics and saves a visualization PNG.

    Args:
        npy_path (str): Path to the .npy file containing the waveform (1D numpy array).
        sample_rate (int, optional): Sampling rate in Hz. Defaults to 44100.

    Returns:
        None. Prints statistics and saves a PNG visualization in the script's directory.
    """
    # Load the waveform
    waveform = np.load(npy_path)
    
    # Calculate time axis
    duration = len(waveform) / sample_rate
    time = np.linspace(0, duration, len(waveform))
    
    # Print statistics
    print("="*80)
    print(f"Waveform Analysis: {os.path.basename(npy_path)}")
    print("="*80)
    print(f"\nFile: {npy_path}")
    print(f"\nArray shape: {waveform.shape}")
    print(f"Data type: {waveform.dtype}")
    print(f"Number of samples: {len(waveform):,}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"\nAmplitude statistics:")
    print(f"  Min: {waveform.min():.6f}")
    print(f"  Max: {waveform.max():.6f}")
    print(f"  Mean: {waveform.mean():.6f}")
    print(f"  Std: {waveform.std():.6f}")
    print(f"  Range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    
    # Show first 20 values
    print(f"\nFirst 20 samples:")
    print(waveform[:20])
    
    # Create visualization
    plt.figure(figsize=(15, 6))
    
    # Full waveform
    plt.subplot(2, 1, 1)
    plt.plot(time, waveform, linewidth=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform: {os.path.basename(npy_path)}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    
    # Zoom on first 0.1 seconds
    zoom_samples = int(0.1 * sample_rate)
    plt.subplot(2, 1, 2)
    plt.plot(time[:zoom_samples], waveform[:zoom_samples], linewidth=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Zoomed: First 0.1 seconds')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Save plot in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    output_path = os.path.join(script_dir, f"{base_name}_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show plot (if running interactively)
    plt.show()
    
    print("="*80 + "\n")


if __name__ == "__main__":
    """
    Command-line interface for visualizing a .npy waveform file.
    Usage:
        python visualize_npy.py <path_to_npy_file>
    If no argument is given, uses the first .npy file found in /mloscratch/users/gnahas/data/waveform_norm.
    """
    if len(sys.argv) > 1:
        npy_file = sys.argv[1]
    else:
        # Default: find first .npy file in waveform_norm directory
        import glob
        npy_dir = "/mloscratch/users/gnahas/data/waveform_norm_silence_trimmed"
        npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
        if npy_files:
            npy_file = npy_files[0]
            print(f"Using first .npy file found: {npy_file}\n")
        else:
            print("No .npy files found!")
            print(f"Usage: python visualize_npy.py <path_to_npy_file>")
            sys.exit(1)

    visualize_waveform(npy_file)
