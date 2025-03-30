import subprocess
import os
import sys
import matplotlib.pyplot as plt
import importlib.util
import sounddevice as sd
import numpy as np
import time
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from matplotlib import colormaps
import soundfile as sf
from scipy.signal import butter, lfilter
import io  # Import io for BytesIO
import argparse # Import argparse
from scipy.interpolate import interp1d
import multiprocessing # Import multiprocessing
from matplotlib.collections import LineCollection # Correct import

# ========================================================================
# DEPENDENCY CHECK BLOCK
# ========================================================================
def check_dependencies():
    """Check if required dependencies are installed."""
    ffmpeg_installed = False
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ffmpeg_installed = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if not ffmpeg_installed:
        print("Error: ffmpeg is not installed or not in your PATH.")
        print("Please install ffmpeg to convert m4a files to wav.")
        print("You can usually install it using your system's package manager (e.g., 'apt install ffmpeg' on Debian/Ubuntu, 'brew install ffmpeg' on macOS).")
        return False

    if importlib.util.find_spec("soundfile") is None:
        print("Error: soundfile library is not installed.")
        print("Please install it using: pip install soundfile")
        return False

    if importlib.util.find_spec("sounddevice") is None:
        print("Error: sounddevice library is not installed.")
        print("Please install it using: pip install sounddevice")
        return False

    return True

# ========================================================================
# AUDIO FILE CONVERSION BLOCK
# ========================================================================
def convert_m4a_to_wav(m4a_path):
    """Convert ALAC m4a to WAV using ffmpeg."""
    wav_path = m4a_path.replace('.m4a', '.wav')
    cmd = ["ffmpeg", "-y", "-i", m4a_path, wav_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True) # capture stderr
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion:")
        print(e.stderr.decode()) # print ffmpeg error message
        return None  # Indicate conversion failure
    return wav_path

# ========================================================================
# AUDIO FILTERING BLOCK
# ========================================================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to audio data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# ========================================================================
# SPECTRUM ANALYSIS BLOCK
# ========================================================================
def calculate_spectrum_frame(segment, samplerate):
    """Calculates spectrum data for a single frame."""
    samples_per_frame = len(segment)
    freqs_full = np.fft.rfftfreq(samples_per_frame, d=1/samplerate)
    spectrum_full = np.abs(np.fft.rfft(segment))
    spectrum_db_full = 20 * np.log10(np.maximum(spectrum_full, 1e-12))

    custom_bins = [41.20,
51.50,
64.38,
80.47,
100.59,
125.73,
157.17,
196.46,
245.57,
306.96,
383.70,
479.63,
599.54,
749.42,
936.78,
1170.97,
1463.72,
1829.65,
2287.06,
2858.82,
3573.53,
4466.91,
5583.64,
6979.55,
8724.44,
10905.55]

    freqs = np.array([(custom_bins[i] + custom_bins[i+1]) / 2 for i in range(len(custom_bins) - 1)])
    freqs = np.insert(freqs, 0, 10.0) # Insert 10Hz at the beginning
    freq_norm = (np.log10(freqs) - np.log10(20)) / (np.log10(20000) - np.log10(20))
    spectrum_db = np.interp(freqs, freqs_full, spectrum_db_full)
    spectrum_db[0] = spectrum_db[1] # Extrapolate amplitude for 10Hz to be same as first bin
    return freqs, spectrum_db, freq_norm, custom_bins # Return custom_bins

# ========================================================================
# PROCESS FRAME SEGMENT INLINE FUNCTION - FOR MULTIPROCESSING
# ========================================================================
def _process_frame_segment_inline(frame_indices_chunk, audio_data, samplerate, frame_size): # Revert to 4 arguments
    """
    Worker function for multiprocessing to process frame segments.
    This function is executed in parallel processes to calculate spectrum frames.
    Moved to top level to be picklable for multiprocessing.
    """
    print(f"Process {os.getpid()} processing frames: {frame_indices_chunk[0]}-{frame_indices_chunk[-1]}") # DEBUG: Print process ID and frame range
    spectrum_frames_segment = []
    num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1 # Determine number of channels

    for i in frame_indices_chunk:
        start_sample = i * frame_size
        end_sample = min((i + 1) * frame_size, len(audio_data))

        frame_spectrum_data = [] # List to hold spectrum data for each channel in this frame

        for channel in range(num_channels):
            if audio_data.ndim > 1:
                segment = audio_data[start_sample:end_sample, channel]
            else:
                segment = audio_data[start_sample:end_sample] # Mono case

            if len(segment) < frame_size:
                segment = np.pad(segment, (0, frame_size - len(segment)))
            segment = segment.astype(np.float32)
            freqs, spectrum_db, freq_norm, custom_bins = calculate_spectrum_frame(segment, samplerate)
            frame_spectrum_data.append({'freqs': freqs, 'spectrum_db': spectrum_db, 'freq_norm': freq_norm, 'custom_bins': custom_bins}) # Store spectrum data for each channel

        spectrum_frames_segment.append({'frame_index': i, 'channel_data': frame_spectrum_data}) # Store channel data under 'channel_data' and frame index
    return spectrum_frames_segment

# ========================================================================
# PROCESS FRAME SEGMENT WRAPPER - FOR MULTIPROCESSING
# ========================================================================
def _process_frame_segment_wrapper(task_data):
    """
    Wrapper function for multiprocessing to unpack task data and call _process_frame_segment_inline.
    This wrapper is necessary to use pool.map with _process_frame_segment_inline which takes multiple arguments.
    """
    return _process_frame_segment_inline(*task_data) # Unpack task_data tuple and call _process_frame_segment_inline

# ========================================================================
# PRE-BUFFER SPECTRUM DATA - MULTIPROCESSING
# ========================================================================
def pre_buffer_spectrum_data_mp(audio_data, samplerate, frame_size=4096, num_processes=None): # Multiprocessing version
    """Pre-calculates spectrum data for the entire audio using multiprocessing."""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for buffering...") # Updated print message

    num_frames = len(audio_data) // frame_size + (1 if len(audio_data) % frame_size != 0 else 0)
    if audio_data.ndim > 1: # Stereo case
        num_frames = audio_data.shape[0] // frame_size + (1 if audio_data.shape[0] % frame_size != 0 else 0)
    else: # Mono case
        num_frames = len(audio_data) // frame_size + (1 if len(audio_data) % frame_size != 0 else 0)


    frame_indices = list(range(num_frames))
    chunk_size = len(frame_indices) // num_processes
    frame_chunks = [frame_indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    frame_chunks[-1] = frame_indices[ (num_processes-1)*chunk_size: ] # Ensure all frames are included in last chunk

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(_process_frame_segment_wrapper, [(chunk, audio_data, samplerate, frame_size) for chunk in frame_chunks]) # Use pool.map instead of starmap and revert worker function signature
    pool.close()
    pool.join()

    # Combine results and sort by frame_index to maintain order
    spectrum_frames_data = []
    for segment_result in results:
        spectrum_frames_data.extend(segment_result)
    spectrum_frames_data.sort(key=lambda x: x['frame_index']) # Sort by original frame index
    return spectrum_frames_data

# ========================================================================
# LOUDNESS ANALYSIS BLOCK
# ========================================================================
def analyze_loudness(wav_path):
    """Compute LUFS and Loudness Range from raw waveform using scipy and RMS windowing."""

    try:
        data, samplerate = sf.read(wav_path)
    except RuntimeError as e:
        print(f"Error reading file: {e}")
        return None, None, None, None, None, None, None

    if data.ndim == 1: # Mono case, keep as is
        num_channels = 1
    elif data.ndim > 1: # Stereo or multichannel
        num_channels = data.shape[1]
    else:
        print("Unsupported audio data dimensions.")
        return None, None, None, None, None, None, None


    # Normalize to -1.0 to 1.0 if necessary
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    # Short-term RMS window (e.g., 400ms)
    window_size = int(0.4 * samplerate)
    stride = int(0.1 * samplerate)  # 100ms stride

    integrated_lufs_channels = []
    lra_channels = []
    lufs_vals_channels = []
    band_lufs_channels = []


    for channel in range(num_channels):
        if num_channels > 1:
            channel_data = data[:, channel]
        else:
            channel_data = data

        windows = np.lib.stride_tricks.sliding_window_view(channel_data, window_shape=window_size)[::stride]

        # Calculate short-term loudness (RMS -> LUFS)
        rms_vals = np.sqrt(np.mean(windows**2, axis=1))
        lufs_vals = -0.691 + 10 * np.log10(np.maximum(rms_vals**2, 1e-12))

        integrated_lufs = np.mean(lufs_vals)
        lra = np.percentile(lufs_vals, 95) - np.percentile(lufs_vals, 10)

        integrated_lufs_channels.append(integrated_lufs)
        lra_channels.append(lra)
        lufs_vals_channels.append(lufs_vals)


        bands = [
            (20, 250),    # Low
            (250, 2000),  # Mid
            (2000, 20000) # High
        ]
        band_lufs_for_channel = []

        for band in bands:
            filtered = bandpass_filter(channel_data, band[0], band[1], samplerate)
            if np.allclose(filtered, 0.0):
                band_lufs_for_channel.append(np.full_like(rms_vals, -70.0).tolist())  # Use silence baseline, convert to list
                continue
            filtered = filtered / np.max(np.abs(filtered)) if np.max(np.abs(filtered)) > 0 else filtered
            band_windows = np.lib.stride_tricks.sliding_window_view(filtered, window_shape=window_size)[::stride]
            safe_band_windows = np.clip(band_windows, -1000.0, 1000.0)
            band_rms_vals = np.sqrt(np.mean(np.square(safe_band_windows), axis=1))
            band_lufs_vals = -0.691 + 10 * np.log10(np.maximum(band_rms_vals**2, 1e-12))
            band_lufs_for_channel.append(band_lufs_vals.tolist()) # Convert to list
        band_lufs_channels.append(band_lufs_for_channel)


    return integrated_lufs_channels, lra_channels, lufs_vals_channels, stride, samplerate, data, band_lufs_channels

# ========================================================================
# MAIN FUNCTION BLOCK
# ========================================================================
def main(file_path, frame_rate_flag, frame_size_flag, processes_flag): # Added processes_flag
    """
    Main function to orchestrate audio analysis and visualization.
    """
    if not file_path.endswith('.m4a'):
        print("Error: Only .m4a files are supported.")
        return

    if not os.path.exists(file_path):
        print(f"Error: File not found: '{file_path}'")
        return

    print(f"Converting '{file_path}' to WAV...")
    wav_path = convert_m4a_to_wav(file_path)

    if wav_path is None: # Check if conversion failed
        return

    print("Analyzing LUFS and LRA...")
    integrated_lufs_channels, lra_channels, lufs_vals_channels, stride, samplerate, audio_data, band_lufs_channels = analyze_loudness(wav_path)

    if integrated_lufs_channels is None or lra_channels is None: # Check if analysis failed
        os.remove(wav_path) # Clean up wav even if analysis failed
        return

    print(f"\nResults for: {file_path}")
    for i, lufs in enumerate(integrated_lufs_channels):
        print(f"Channel {i+1} Integrated LUFS: {lufs:.2f} LUFS")
    for i, lra in enumerate(lra_channels):
        print(f"Channel {i+1} Loudness Range (EBU LRA): {lra:.2f} LU")

    # Plot LUFS over time (for the first channel for now, can be extended)
    plt.figure(figsize=(10, 5), facecolor='black') # Set figure facecolor to black
    for i, lufs_vals in enumerate(lufs_vals_channels):
        label = f'Short-term LUFS Ch{i+1}'
        alpha = 1.0 if i == 0 else 0.7 # Make first channel more opaque
        plt.plot(np.arange(len(lufs_vals)) * (stride / samplerate), lufs_vals, label=label, color=f'C{i}', alpha=alpha) # Set line color to white
    plt.xlabel('Time (s)', color='white') # Set label color to white
    plt.ylabel('LUFS', color='white') # Set label color to white
    plt.title(f'LUFS Over Time: {os.path.basename(file_path)}', color='white') # Set title color to white
    plt.grid(False) # Remove grid
    plt.gca().set_facecolor('black') # Set axes facecolor to black
    plt.gca().spines['bottom'].set_color('white') # Set spines color to white
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().tick_params(axis='x', colors='white') # Set tick colors to white
    plt.gca().tick_params(axis='y', colors='white')
    plt.tight_layout()
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white') # Set legend colors
    plt.show()

    animate_lufs_plot(lufs_vals_channels, file_path, stride, samplerate, audio_data, band_lufs_channels, frame_rate=frame_rate_flag, frame_size=frame_size_flag, num_processes=processes_flag) # Pass processes_flag

    # Clean up temporary file
    os.remove(wav_path)
    print(f"Temporary WAV file '{wav_path}' removed.")

# ========================================================================
# ANIMATE LUFS PLOT FUNCTION BLOCK
# ========================================================================
def animate_lufs_plot(lufs_vals_channels, file_path, sample_stride, samplerate, audio_data, band_lufs_channels, frame_rate=30, frame_size=4096, num_processes=None): # Added num_processes
    """
    Generates and controls the Sonic Visualizer style spectrum animation plot.
    """
    cmap = plt.get_cmap('hsv')
    norm = mcolors.Normalize(vmin=5, vmax=80)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black') # Set figure facecolor to black
    ax.set_xscale('log')
    ax.set_xlim(40, 11000)
    ax.set_autoscale_on(False)
    ax.set_xlabel('Frequency (Hz)', color='white') # Set label color to white
    ax.set_ylabel('Amplitude (dB)', color='white') # Set label color to white
    ax.set_ylim(-20, 60)
    ax.set_title(f"Yr crystals: {os.path.basename(file_path)}", color='white') # Set title color to white
    ax.grid(False) # Remove grid lines
    ax.set_facecolor('black') # Set axes facecolor to black
    ax.spines['bottom'].set_color('white') # Set spines color to white
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white') # Set tick colors to white
    ax.tick_params(axis='y', colors='white')

    line_segments = [None, None] # Line segments for two channels
    current_frame_index = 0
    is_playing = False # Initially paused
    pre_buffered = False
    buffer_progress = 0.0 # 0 to 1
    stream_object = None # sounddevice stream - renamed to avoid shadowing
    start_time = 0
    playback_started = False
    # frame_rate = 30 # Animation frame rate - now parameter
    animation_frame_interval_ms = 1000 / frame_rate # Calculate interval in milliseconds
    visualization_pool = None # Initialize process pool outside functions
    base_fill_color = [None, None] # Initialize base_fill_color for two channels
    spectrum_frames_data = None # Initialize spectrum_frames_data to None
    num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1 # Determine num_channels

    num_processes = None # ADDED: Define num_processes in animate_lufs_plot scope

    # Playback control button - now Buffer & Visualize
    button_ax = plt.axes([0.8, 0.025, 0.15, 0.04], facecolor='lightgoldenrodyellow') # [left, bottom, width, height]
    play_pause_button = Button(button_ax, 'Buffer & Visualize', color='lightgoldenrodyellow', hovercolor='0.975') # Changed label

    # Download button
    download_ax = plt.axes([0.6, 0.025, 0.15, 0.04], facecolor='lightgoldenrodyellow')
    download_button = Button(download_ax, 'Download MV', color='lightgrey')
    download_button.ax.set_facecolor('lightgoldenrodyellow') # Initially enabled visually
    download_button.disabled = False # Initially enabled

    # Progress bar axes
    progress_ax = plt.axes([0.05, 0.025, 0.4, 0.03], facecolor='lightgoldenrodyellow')
    progress_slider = Slider(progress_ax, 'Buffer', 0, 1.0, valinit=0, valfmt='%0.1f%%', color='lightgoldenrodyellow')
    progress_slider.active = False # Make it read-only
    progress_slider.label.set_color('white') # Set slider label color to white
    progress_slider.valtext.set_color('white') # Set slider value text color to white


    def _update_visuals_worker(frame_index, spectrum_frames_data, ax, cmap, base_fill_color): # Worker function for pool
        return update_visuals_frame(frame_index, spectrum_frames_data, ax, cmap, base_fill_color)

    def minimal_audio_callback(outdata, frames, time, status): # Changed to outdata, time
        nonlocal current_frame_index, start_time, visualization_pool, base_fill_color, line_segments

        if status:
            print("Minimal Audio callback status:", status)
            outdata.fill(0) # Fill with silence on error
            return

        if not is_playing or current_frame_index >= len(spectrum_frames_data):
            outdata.fill(0) # Silence if paused or end reached
            return

        start_sample = current_frame_index * frame_size # Use frame_size
        end_sample = min((current_frame_index + 1) * frame_size, len(audio_data)) # Use frame_size


        segments = []
        for channel in range(num_channels):
            if audio_data.ndim > 1:
                segment = audio_data[start_sample:end_sample, channel]
            else:
                segment = audio_data[start_sample:end_sample] # Mono case

            if len(segment) < frames: # frames from callback is the buffer size, use it directly
                segment = np.pad(segment, (0, frames - len(segment)))
            segments.append(segment)

        if num_channels > 1:
            outdata[:] = np.column_stack(segments) # Stereo output
        else:
            outdata[:] = segments[0].reshape(-1, 1) # Mono output, ensure shape matches outdata


        update_visuals(current_frame_index) # Call update_visuals which now handles sync calls

        current_frame_index += 1


    def update_visuals_callback(result): # Callback - REMOVED - not needed for sync updates
        pass # update is now synchronous


    def update_lines(updated_line_segments): # Separate function to update lines
        nonlocal line_segments
        line_segments = updated_line_segments # Replace with new lines


    def start_visualization(): # Renamed from start_playback to avoid confusion
        nonlocal stream_object, start_time, is_playing, playback_started, current_frame_index, visualization_pool, line_segments, base_fill_color

        if playback_started:
            print("Visualization already started, ignoring start request.")
            return

        if stream_object is not None:
            stream_object.stop()
            stream_object.close()
            stream_object = None

        if visualization_pool is not None: # Close pool if it exists (single process now)
            visualization_pool.close()
            visualization_pool.join()
            visualization_pool = None
        visualization_pool = None # Ensure pool is set to None for single-process

        current_frame_index = 0 # Reset frame index on start/resume
        line_segments = [None, None] # Reset line segments when starting visualization
        base_fill_color = [None, None] # Reset base_fill_color
        try:
            channels = num_channels if num_channels > 1 else 1 # Use num_channels for stream
            stream_object = sd.OutputStream(samplerate=samplerate, channels=channels, callback=minimal_audio_callback, blocksize=frame_size) # Stereo stream if available, otherwise mono
            stream_object.start()
            start_time = time.time()
            is_playing = True
            playback_started = True
            print(f"Visualization started with frame size: {frame_size}, single process, {channels} channels.", ) # Indicate single process and channels
        except Exception as e:
            print(f"Error starting visualization: {e}")
            is_playing = False
            playback_started = False


    def stop_visualization(): # Renamed from stop_playback
        nonlocal stream_object, is_playing, playback_started, visualization_pool
        if is_playing:
            if stream_object:
                stream_object.stop()
                stream_object.close()
                stream_object = None
            is_playing = False
            playback_started = False
            print("Visualization stopped.")
        if visualization_pool: # Close pool if it exists
            visualization_pool.close()
            visualization_pool.join()
            visualization_pool = None


    num_processes = None # ADDED: Define num_processes in animate_lufs_plot scope
    # ========================================================================
    # TOGGLE BUFFER VISUALIZE FUNCTION - BUTTON CLICK HANDLER
    # ========================================================================
    def toggle_buffer_visualize(event): # Renamed button callback function
        nonlocal is_playing, pre_buffered, current_frame_index, playback_started, spectrum_frames_data, num_processes # Added num_processes to nonlocal
        # num_processes = None # REMOVED: No re-initialization here - causing shadowing

        if not pre_buffered: # Buffer and start visualization
            play_pause_button.label.set_text('Buffering & Visualizing...') # Changed label
            play_pause_button.ax.set_facecolor('lightcoral') # Indicate buffering
            fig.canvas.draw_idle()
            spectrum_frames_data = pre_buffer_spectrum_data_mp(audio_data, samplerate, frame_size, num_processes=num_processes) # Buffer using multiprocessing
            pre_buffered = True
            play_pause_button.label.set_text('Pause Visualization') # Change button label after buffering
            play_pause_button.ax.set_facecolor('lightgoldenrodyellow')
            start_visualization() # Start visualization after buffering
        elif is_playing: # Pause visualization
            stop_visualization()
            play_pause_button.label.set_text('Resume Visualization')
        else: # Resume visualization
            start_visualization()
            play_pause_button.label.set_text('Pause Visualization')
        fig.canvas.draw_idle()


    def prebuffer_visualize_process(): # Combined process - REMOVED - toggle_buffer_visualize now handles buffering directly
        pass # No longer needed

    def download_music_video(event):
     if not download_button.disabled:
        print("Download Music Video button clicked! Starting in-memory processing...")
        out_filename = "music_video.mp4"
        # frame_rate = 30 # - now parameter
        total_frames = len(spectrum_frames_data)
        fig_width, fig_height = fig.canvas.get_width_height()
        command = [
            'ffmpeg',
            '-y', # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{int(fig_width)}x{int(fig_height)}', # Set frame size
            '-pix_fmt', 'rgba', # Changed to rgba to tostring_argb
            '-r', str(frame_rate), # Use frame_rate parameter
            '-i', '-', # Input from pipe
            '-f', 'wav',
            '-i', '-', # Audio input from pipe
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-shortest',
            out_filename
        ]
        print(f"FFmpeg command: {' '.join(command)}")

        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) # stderr for ffmpeg output
        # No separate audio_pipe needed, using process.stdin for both

        # Audio data to bytes (assuming float32 and needs to be converted to int16 for wav pipe)
        audio_scaled = np.int16(audio_data * 32767) # Scale to int16 range
        wav_buf = io.BytesIO() # Create in-memory buffer for WAV
        sf.write(wav_buf, audio_scaled, samplerate, format='WAV') # Write WAV data to buffer
        audio_bytes_with_header = wav_buf.getvalue() # Get bytes including header

        print("Processing frames and piping to ffmpeg...")
        start_time_download = time.time()
        for frame_num in range(total_frames):
            updated_lines_data, base_fill_color = update_visuals_frame(frame_num, spectrum_frames_data, ax, cmap, base_fill_color) # Update visuals for MV download - get updated line segments and base_fill_color
            update_lines(updated_lines_data) # Update lines on axes
            fig.canvas.draw()
            frame_bytes = fig.canvas.tostring_argb() # Changed to tostring_argb

            try:
                process.stdin.write(frame_bytes) # Pipe frame data to ffmpeg video input
            except IOError as e:
                print(f"IOError writing frame data to ffmpeg pipe: {e}")
                process.stderr.close() # Close stderr to avoid blocking
                process.stdin.close()
                return

            progress = (frame_num + 1) / total_frames * 100
            if frame_num % 10 == 0 or frame_num == total_frames - 1: # Update progress less frequently
                print(f"Frame {frame_num+1}/{total_frames} piped ({progress:.1f}%)", end='\r')

        print("\nFrames piping complete. Piping audio...")

        try:
            process.stdin.write(audio_bytes_with_header) # Pipe audio data to ffmpeg audio input (using the same pipe)
        except IOError as e:
            print(f"IOError writing audio data to ffmpeg pipe: {e}")
            process.stderr.close()
            process.stdin.close()
            return

        process.stdin.close() # Close stdin after writing all data
        print("Audio piping complete. Waiting for ffmpeg...")
        output, errors = process.communicate() # Wait for ffmpeg to finish and get output/errors
        end_time_download = time.time()

        if process.returncode == 0:
            print(f"Music video saved as '{out_filename}' in {end_time_download - start_time_download:.2f} seconds.")
        else:
            print(f"FFmpeg failed with code {process.returncode}:")
            if errors:
                print(errors.decode())
            if output:
                print(output.decode())


    def update_visuals(frame_index): # Wrapper - now synchronous, direct call
        nonlocal base_fill_color, line_segments # Removed visualization_pool - no longer needed for sync call

        updated_lines_data, base_fill_color = update_visuals_frame(frame_index, spectrum_frames_data, ax, cmap, base_fill_color) # Sync call to update visuals
        update_lines(updated_lines_data) # Update lines on axes
        fig.canvas.draw_idle() # Request redraw in main thread


    def update_visuals_frame(frame_index, spectrum_frames_data, ax, cmap, base_fill_color): # Actual update function
        line_segments_local = [None, None] # Initialize local line_segments for two channels

        if frame_index >= len(spectrum_frames_data):
            return line_segments_local, base_fill_color # Return empty lines and base_fill_color if no update needed

        # Clear previous collections and fills - THIS IS THE FIX
        for existing_collection in list(ax.collections): # Iterate over a list copy to avoid modification during iteration
            existing_collection.remove() # Call remove on the artist object itself!
        for existing_patch in list(ax.patches): # Iterate over a list copy to avoid modification during iteration
            existing_patch.remove() # Call remove on the artist object itself!


        frame_data = spectrum_frames_data[frame_index]
        channel_data = frame_data['channel_data'] # Get channel data for both channels

        for channel_index in range(num_channels): # Iterate over channels
            channel_spectrum = channel_data[channel_index]
            freqs = channel_spectrum['freqs']
            spectrum_db = channel_spectrum['spectrum_db']
            freq_norm = channel_spectrum['freq_norm']
            custom_bins = channel_spectrum['custom_bins'] # Retrieve custom_bins

            # Calculate single fill color based on overall spectrum (base color) - calculate every frame
            mid_freq_index = len(freqs) // 2
            overall_freq_norm = (np.log10(freqs[mid_freq_index]) - np.log10(20)) / (np.log10(20000) - np.log10(20))
            overall_amp_norm = np.clip((np.mean(spectrum_db) + 0) / 60, 0, 1)
            amp_weight = 1.0 / ((overall_freq_norm + 1e-4) ** 5)
            amp_weight = np.clip(amp_weight * 2.0, 0.5, 6.0)
            overall_hue = (overall_freq_norm + overall_amp_norm * amp_weight * 0.9) % 1.0
            current_base_fill_color = list(cmap(overall_hue)) # Base fill color (RGB)
            current_base_fill_color[-1] = 0.5 # Transparency for base fill
            base_fill_color[channel_index] = current_base_fill_color # Store base fill color for channel


            # Plot non-smooth line segments and segmented fill with varying alpha
            collection = LineCollection([], linewidths=2) # Create empty LineCollection - moved inside frame update
            ax.add_collection(collection) # Add to axes

            segment_colors = []
            all_segments_data = [] # Collect segment data for LineCollection

            for i in range(len(custom_bins) - 1):
                start_freq_bin = custom_bins[i]
                end_freq_bin = custom_bins[i+1]

                # Find corresponding frequencies and amplitudes for current bin from original data
                start_index = np.argmin(np.abs(freqs - start_freq_bin))
                end_index = np.argmin(np.abs(freqs - end_freq_bin))

                freqs_segment = freqs[start_index:end_index+1]
                animated_y_segment = spectrum_db[start_index:end_index+1]

                if len(freqs_segment) < 2:
                    continue

                # Calculate color for the segment line
                mid_freq = np.mean([start_freq_bin, end_freq_bin])
                mid_freq_norm = (np.log10(mid_freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
                amp_norm = np.clip((np.mean(animated_y_segment) + 0) / 60, 0, 1)
                amp_weight = 1.0 / ((mid_freq_norm + 1e-4) ** 5)
                amp_weight = np.clip(amp_weight * 2.0, 0.5, 6.0)
                hue = (mid_freq_norm + amp_norm * amp_weight * 0.9) % 1.0
                rgba_line = list(cmap(hue))
                if channel_index == 1: # Make second channel slightly more opaque and different color if stereo
                    rgba_line[3] = rgba_line[3] * 0.7 # More opaque
                    rgba_line[0] = max(0, rgba_line[0] - 0.1) # Shift hue slightly for visual difference
                    rgba_line[1] = max(0, rgba_line[1] - 0.1)
                    rgba_line[2] = min(1, rgba_line[2] + 0.1)


                segment_colors.append(rgba_line) # Collect colors

                segments_data = np.column_stack((freqs_segment, animated_y_segment)) # Create segment data
                segment_pairs = np.array([segments_data[:-1], segments_data[1:]]).transpose((1, 0, 2)) # Create line pairs
                all_segments_data.extend(segment_pairs) # Collect all segments


                # Calculate segmented fill color with alpha based on segment amplitude
                segment_amplitude_norm = np.clip(0.3*(np.mean(animated_y_segment) - 1/(np.mean(animated_y_segment)-20) + 20) / 33 -0.1, 0.02, 0.7) # Reduced alpha range
                segment_fill_color = current_base_fill_color[:] # Copy base fill color
                segment_fill_color[-1] = segment_amplitude_norm # Set alpha based on amplitude
                if channel_index == 1: # Make second channel more opaque if stereo
                    segment_fill_color[3] = segment_fill_color[3] * 0.7 # More opaque

                # Apply frequency shift to the second channel visualization
                freqs_segment_display = freqs_segment
                if channel_index == 1 and num_channels > 1: # Shift right channel slightly
                    freqs_segment_display = freqs_segment * 1.002 # Very small shift to right

                ax.fill_between(freqs_segment_display, animated_y_segment, -20, color=segment_fill_color) # Segmented fill with amplitude-based alpha

                # Draw vertical lines with segment line color
                ax.vlines(x=freqs_segment_display[0], ymin=-20, ymax=animated_y_segment[0], color=rgba_line, linewidth=0.5) # Vertical line at start
                ax.vlines(x=freqs_segment_display[-1], ymin=-20, ymax=animated_y_segment[-1], color=rgba_line, linewidth=0.5) # Vertical line at end

            collection.set_segments(all_segments_data) # Set all segments at once
            collection.set_colors(segment_colors) # Set all colors at once
            line_segments_local[channel_index] = collection # Store LineCollection for each channel


        return line_segments_local, base_fill_color # Return local line_segments and base_fill_color


    # No animation needed anymore, visuals are updated in audio callback
    # ani = animation.FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)

    play_pause_button.on_clicked(toggle_buffer_visualize) # Renamed callback - now toggle_buffer_visualize
    download_button.on_clicked(download_music_video)

    # plt.tight_layout(rect=[0, 0.06, 1, 1]) # Adjust layout to make space for buttons # REMOVE THIS LINE
    plt.tight_layout(rect=[0, 0.06, 1, 0.98]) # Slightly adjusted tight_layout to leave a bit of space at the bottom and top.
    plt.show()
    stop_visualization() # Ensure stream is stopped on close


if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Analyze audio loudness and create music visualizations.")
    parser.add_argument("file_path", help="Path to the input .m4a audio file.")
    parser.add_argument("-f", "--framerate", type=float, default=60.0, help="Frame rate for visualization and video (default: 30.0).")
    parser.add_argument("-s", "--framesize", type=int, default=4096, help="Frame size (sample size) for processing (default: 8192).") # Changed to framesize
    parser.add_argument("-p", "--processes", type=int, default=None, help=f"Number of processes to use for buffering (default: max CPU cores: {multiprocessing.cpu_count()}).") # Updated help text
    args = parser.parse_args()

    main(args.file_path, args.framerate, args.framesize, args.processes) # Pass processes_flag to main

    print(f"\nUsage: python lu_range_analyzer.py [-f FRAMERATE] [-s FRAMESIZE] [-p PROCESSES] <file.m4a>") # Updated usage message
    print(f"  -f FRAMERATE    Frame rate for visualization and video (default: 30.0)")
    print(f"  -s FRAMESIZE    Frame size (sample size) for processing (default: 8192)") # Updated usage message
    print(f"  -p PROCESSES    Number of processes for buffering (default: max CPU cores)") # Updated usage message
