from moviepy import AudioFileClip, VideoClip
import numpy as np
from PIL import Image, ImageEnhance
from scipy.signal import get_window
from scipy.ndimage import zoom
import sys
import os
import time
import colorsys
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import proglog

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# Try to import cv2 for faster image operations, fall back to PIL if unavailable
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

LOUDNESS_WEIGHT = 0.3
ENCODE_WEIGHT = 0.7

# Path to current file or exe
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.argv[0])))
output_path = os.path.join(base_dir, "output.mp4")

window_function = get_window('hann', 2048)
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave one CPU core free

def shift_hue_numpy(arr, rotation_degrees):
    """
    Rotates hue of an RGBA numpy array using vectorized numpy operations.
    """
    rgb = arr[..., :3].astype(np.float32) / 255.0
    alpha = arr[..., 3:4]
    
    # Convert RGB to HSV using numpy (faster than matplotlib)
    rgb_reshaped = rgb.reshape(-1, 3)
    max_val = np.max(rgb_reshaped, axis=1, keepdims=True)
    min_val = np.min(rgb_reshaped, axis=1, keepdims=True)
    delta = max_val - min_val
    
    # Value
    v = max_val.flatten()
    
    # Saturation
    s = np.zeros_like(v)
    mask = v != 0
    s[mask] = (delta[mask].flatten() / v[mask])
    
    # Hue
    h = np.zeros_like(v)
    delta_flat = delta.flatten()
    delta_nz = delta_flat != 0
    r, g, b = rgb_reshaped[:, 0], rgb_reshaped[:, 1], rgb_reshaped[:, 2]
    
    mask_r = (delta_nz) & (max_val.flatten() == r)
    mask_g = (delta_nz) & (max_val.flatten() == g)
    mask_b = (delta_nz) & (max_val.flatten() == b)
    
    h[mask_r] = (60 * (((g[mask_r] - b[mask_r]) / delta_flat[mask_r]) % 6)) / 360.0
    h[mask_g] = (60 * (((b[mask_g] - r[mask_g]) / delta_flat[mask_g]) + 2)) / 360.0
    h[mask_b] = (60 * (((r[mask_b] - g[mask_b]) / delta_flat[mask_b]) + 4)) / 360.0
    
    # Apply hue rotation
    rotation_norm = (rotation_degrees % 360) / 360.0
    h = (h + rotation_norm) % 1.0
    
    # Convert back to RGB
    h_360 = h * 360.0
    c = v * s
    x = c * (1 - np.abs((h_360 / 60) % 2 - 1))
    
    rgb_new = np.zeros_like(rgb_reshaped)
    for i in range(len(h_360)):
        if s[i] == 0:
            rgb_new[i] = [v[i], v[i], v[i]]
        else:
            hue_sector = int(h_360[i] / 60) % 6
            if hue_sector == 0:
                rgb_new[i] = [c[i], x[i], 0]
            elif hue_sector == 1:
                rgb_new[i] = [x[i], c[i], 0]
            elif hue_sector == 2:
                rgb_new[i] = [0, c[i], x[i]]
            elif hue_sector == 3:
                rgb_new[i] = [0, x[i], c[i]]
            elif hue_sector == 4:
                rgb_new[i] = [x[i], 0, c[i]]
            else:
                rgb_new[i] = [c[i], 0, x[i]]
        
        m = v[i] - c[i]
        rgb_new[i] += m
    
    rgb_new = rgb_new.reshape(arr.shape[0], arr.shape[1], 3)
    final = np.dstack((rgb_new * 255, alpha)).astype(np.uint8)
    return final

class MyCustomBarLogger(proglog.ProgressBarLogger):
    def __init__(self):
        super().__init__()
        self.progress_percentage = 0

    def bars_callback(self, bar, attr, value, total, eta=None, **kwargs):
        # This method is called whenever the progress of a bar changes
        if attr == 'n':
            percentage = (value / total) * 100
            self.progress_percentage = percentage
            # You can print the percentage or update a GUI element here
            print(f"{bar}: {percentage:.2f}%")
        # You can also access the estimated time (eta) here if needed

custom_logger = MyCustomBarLogger()

def apply_transformations_numpy(img_array, loudness, config):
    """
    Applies transformations based on loudness using numpy and optional cv2 operations.
    Expects numpy array in RGBA format.
    """
    # 1. Size Interpolation (handled elsewhere for efficiency)
    min_s = config.get('min_scale', 1.0)
    max_s = config.get('max_scale', 1.05)
    current_scale = min_s + (max_s - min_s) * loudness
    
    h, w = img_array.shape[:2]
    new_h = max(1, int(h * current_scale))
    new_w = max(1, int(w * current_scale))
    
    # --- Color / Brightness (numpy based) ---
    
    # Saturation - using cv2 if available, else PIL
    if config.get('max_sat', 1.0) != config.get('min_sat', 1.0):
        min_sat = config.get('min_sat', 1.0)
        max_sat = config.get('max_sat', 1.0)
        sat_factor = min_sat + (max_sat - min_sat) * loudness
        
        if HAS_CV2:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            hsv = hsv.astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            img_array = np.dstack((rgb, img_array[:, :, 3]))
        else:
            # PIL fallback
            img_pil = Image.fromarray(img_array)
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(sat_factor)
            img_array = np.array(img_pil)
    
    # Brightness - numpy multiplication
    if config.get('max_bright', 1.0) != config.get('min_bright', 1.0):
        min_bright = config.get('min_bright', 1.0)
        max_bright = config.get('max_bright', 1.0)
        bright_factor = min_bright + (max_bright - min_bright) * loudness
        img_rgb = img_array[:, :, :3].astype(np.float32)
        img_rgb *= bright_factor
        img_array[:, :, :3] = np.clip(img_rgb, 0, 255).astype(np.uint8)
    
    # Hue Rotation
    if config.get('max_hue', 0) != config.get('min_hue', 0):
        min_hue = config.get('min_hue', 0)
        max_hue = config.get('max_hue', 0)
        hue_rot = min_hue + (max_hue - min_hue) * loudness
        if abs(hue_rot) > 0.1:
            img_array = shift_hue_numpy(img_array, hue_rot)
    
    # Transparency (Alpha) - direct numpy operation
    if config.get('max_alpha', 1.0) != config.get('min_alpha', 1.0):
        min_alpha = config.get('min_alpha', 1.0)
        max_alpha = config.get('max_alpha', 1.0)
        alpha_factor = min_alpha + (max_alpha - min_alpha) * loudness
        if alpha_factor < 0.995:
            img_array[:, :, 3] = (img_array[:, :, 3].astype(np.float32) * alpha_factor).astype(np.uint8)
    
    # --- Geometric ---
    
    # Resize using cv2 if available, else PIL
    if new_h != h or new_w != w:
        if HAS_CV2:
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_pil = Image.fromarray(img_array)
            img_pil = img_pil.resize((new_w, new_h), resample=Image.BICUBIC)
            img_array = np.array(img_pil)
    
    # Rotation - using cv2 if available, else PIL
    if config.get('max_rot', 0) != config.get('min_rot', 0):
        min_rot = config.get('min_rot', 0)
        max_rot = config.get('max_rot', 0)
        rot_angle = min_rot + (max_rot - min_rot) * loudness
        if abs(rot_angle) > 0.1:
            if HAS_CV2:
                h, w = img_array.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
                img_array = cv2.warpAffine(img_array, matrix, (w, h), 
                                           borderMode=cv2.BORDER_CONSTANT, 
                                           borderValue=(0, 0, 0, 0))
            else:
                img_pil = Image.fromarray(img_array)
                img_pil = img_pil.rotate(rot_angle, resample=Image.BICUBIC, expand=True)
                img_array = np.array(img_pil)
    
    return img_array

def _process_frame_loudness(args):
    """
    Worker function for parallel loudness computation.
    """
    frame_idx, audio_array, sample_rate, fps, image_configs, max_log_volume, window_func = args
    
    center_sample = int(frame_idx * (sample_rate / fps))
    start = max(0, center_sample - 1024)
    end = min(len(audio_array), center_sample + 1024)
    
    window = audio_array[start:end]
    if len(window) < 2048:
        window = np.pad(window, (0, 2048 - len(window)), mode='constant')
    
    spectrum = np.abs(np.fft.rfft(window * window_func))
    log_spectrum = np.log1p(spectrum)
    freqs = np.fft.rfftfreq(len(window), d=1 / sample_rate)
    
    loudness_values = []
    for config in image_configs:
        f_min = config['freq_min']
        f_max = config['freq_max']
        idx = (freqs >= f_min) & (freqs <= f_max)
        
        raw_loudness = 0.0
        if np.any(idx):
            avg = np.mean(log_spectrum[idx])
            raw_loudness = (avg / max_log_volume) * config.get('factor', 1.0)
            raw_loudness = np.clip(raw_loudness, 0.0, 1.0)
        loudness_values.append(raw_loudness)
    
    return frame_idx, loudness_values

def precompute_loudness(audio_array, sample_rate, fps, duration, image_configs, max_log_volume, progress_callback=None):
    total_frames = int(duration * fps)
    frame_centers = (np.arange(total_frames) * sample_rate / fps).astype(int)

    padded = np.pad(audio_array, (1024, 1024))
    windows = np.stack([
        padded[c:c+2048] * window_function
        for c in frame_centers
    ])

    spectra = np.abs(np.fft.rfft(windows, axis=1))
    log_spectra = np.log1p(spectra)
    freqs = np.fft.rfftfreq(2048, 1 / sample_rate)

    timeline = {i: np.zeros(total_frames) for i in range(len(image_configs))}

    for i, config in enumerate(image_configs):
        idx = (freqs >= config['freq_min']) & (freqs <= config['freq_max'])
        if not np.any(idx):
            continue

        raw = log_spectra[:, idx].mean(axis=1) / max_log_volume
        raw = np.clip(raw * config.get('factor', 1.0), 0, 1)

        attack = config.get('attack', 0.2)
        release = config.get('release', 0.05)

        for f in range(total_frames):
            if f == 0:
                timeline[i][f] = raw[f]
            else:
                prev = timeline[i][f-1]
                a = attack if raw[f] > prev else release
                timeline[i][f] = prev + a * (raw[f] - prev)

    return timeline


def run_visualizer(config, progress_callback=None, on_complete=None):

    if sys.platform.startswith("win"):
        import multiprocessing
        multiprocessing.freeze_support()

    audio_file = config['audio_file']
    bg_file = config['bg_file']
    bg_w, bg_h = config['bg_size']
    fps = config['fps']
    image_configs = config['images']

    # Load background image as numpy array (faster than PIL for processing)
    bg_pil = Image.open(bg_file).resize((bg_w, bg_h)).convert("RGBA")
    bg_img_np = np.array(bg_pil)
    
    # Load audio
    audio_clip = AudioFileClip(audio_file)
    sample_rate = 44100
    audio_array = audio_clip.to_soundarray(fps=sample_rate)
    audio_length = audio_clip.duration

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # 1. Analyze global max volume (optimized with numpy)
    max_log_volume = 1e-6
    for i in range(0, len(audio_array), sample_rate // 10):
        segment = audio_array[i:i+2048]
        if len(segment) < 2048:
            segment = np.pad(segment, (0, 2048 - len(segment)), mode='constant')
        spec = np.abs(np.fft.rfft(segment * window_function))
        max_log_volume = max(max_log_volume, np.max(np.log1p(spec)))

    # 2. Precompute smoothed loudness data with progress tracking
    loudness_data = precompute_loudness(
        audio_array, sample_rate, fps, audio_length, image_configs, max_log_volume, progress_callback
    )

    # 3. Load image assets as numpy arrays once (parallel loading)
    image_assets_np = []
    for entry in image_configs:
        img_pil = Image.open(entry['file']).convert("RGBA")
        img_np = np.array(img_pil)
        image_assets_np.append(img_np)
    
    if progress_callback:
        progress_callback(0, 100, 0, "Starting video render...")

    TRANSFORM_CACHE_SIZE = 64  # power of two

    def make_frame(t):
        frame_idx = int(t * fps)
        # Clamp index to bounds
        total_frames = len(loudness_data[0])
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1
        
        frame = bg_img_np.copy()

        for i, entry in enumerate(image_configs):
            loudness = loudness_data[i][frame_idx]
            # Look up the pre-calculated smoothed loudness
            # Quantize loudness to cache bucket
            bucket = int(loudness * (TRANSFORM_CACHE_SIZE - 1))
            bucket = max(0, min(bucket, TRANSFORM_CACHE_SIZE - 1))


            # Apply transform (numpy-based)
            if 'cache' not in entry:
                entry['cache'] = {}

            cache = entry['cache']

            if bucket not in cache:
                img = apply_transformations_numpy(
                    image_assets_np[i],
                    bucket / (TRANSFORM_CACHE_SIZE - 1),
                    entry
                )

                alpha = img[:, :, 3].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32) * alpha[..., None]

                cache[bucket] = (rgb, alpha)

            rgb, alpha = cache[bucket]


            h, w = rgb.shape[:2]

            x = int(entry['x'] - w / 2)
            y = int(entry['y'] - h / 2)
            
            # Composite images using numpy for better performance
            if 0 <= y < bg_h and 0 <= x < bg_w:
                # Determine region boundaries
                y1 = max(0, y)
                y2 = min(bg_h, y + h)
                x1 = max(0, x)
                x2 = min(bg_w, x + w)
                
                src_y1 = max(0, -y)
                src_y2 = src_y1 + (y2 - y1)
                src_x1 = max(0, -x)
                src_x2 = src_x1 + (x2 - x1)
                
                foreground = rgb[src_y1:src_y2, src_x1:src_x2]
                background = frame[y1:y2, x1:x2, :3].astype(np.float32)

                alpha_expanded = alpha[src_y1:src_y2, src_x1:src_x2][..., None]  # shape (H, W, 1)
                frame[y1:y2, x1:x2, :3] = (
                    foreground + background * (1 - alpha_expanded)
                ).astype(np.uint8)



        # Convert to RGB for video output
        return frame[..., :3]

    total_frames = int(audio_length * fps)
    start_time = time.time()
    frame_count = [0]  # Use list for mutable counter in nested function

    def make_frame_w_progress(t):
        current_frame = int(t * fps) + 1
        frame_count[0] = current_frame
        elapsed = time.time() - start_time
        
        if current_frame > 0 and current_frame <= total_frames:
            avg_t = elapsed / current_frame
            rem = (total_frames - current_frame) * avg_t
            if progress_callback and current_frame % max(1, total_frames // 20) == 0:
                pct = int((current_frame / total_frames) * 100)
                eta = time.strftime("%H:%M:%S", time.gmtime(rem))
                progress_callback(current_frame, total_frames, pct, eta)
        
        return make_frame(t)

    print(f"Rendering video with {NUM_WORKERS} parallel workers...")
    video_clip = VideoClip(make_frame_w_progress, duration=audio_length).with_fps(fps)
    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=custom_logger)

    final_clip.close()
    audio_clip.close()

    if on_complete:
        on_complete()