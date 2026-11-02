import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import glob
import subprocess
import soundfile as sf
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
audio_src = os.path.join(base_dir, 'Audio')
units_dir = os.path.join(base_dir, 'lab4_data', 'units')
output_dir = os.path.join(base_dir, 'audio_output')
figures_dir = os.path.join(base_dir, 'figures')

os.makedirs(units_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

word_map = {
    'ba': 'ba', 'bay': 'bảy', 'bon': 'bốn', 'chin': 'chín',
    'den': 'đến', 'hai': 'hai', 'hang': 'hàng', 'khach': 'khách',
    'khong': 'không', 'moi': 'mời', 'mot': 'một', 'nam': 'năm',
    'quay': 'quầy', 'sau': 'sáu', 'so': 'số', 'tam': 'tám'
}

def detect_speech_boundaries(d, sr, pad_ms=45):
    """
    Detect speech boundaries using combined RMS energy and Zero-Crossing Rate (ZCR).
    Expands around peak energy frame to preserve unvoiced fricatives and stops,
    while removing excessive silence and trailing mouse clicks.
    """
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = (len(d) - win) // hop
    if n_frames <= 2:
        return 0, len(d)
        
    rms = np.array([np.sqrt(np.mean(d[i*hop:i*hop+win]**2)) for i in range(n_frames)])
    zcr = np.array([np.mean(np.abs(np.diff(np.sign(d[i*hop:i*hop+win])))) / 2 for i in range(n_frames)])
    
    peak_idx = int(np.argmax(rms))
    peak_val = float(rms[peak_idx])
    
    noise_rms = float(np.mean(rms[:min(8, len(rms))]))
    rms_thresh = max(0.20 * peak_val, 0.022)
    
    # Expand left from peak (stop if low for 3 frames)
    left = peak_idx
    low_count = 0
    while left > 0:
        is_speech = (rms[left] > rms_thresh) or (rms[left] > noise_rms * 1.3 and zcr[left] > 0.12)
        if is_speech:
            low_count = 0
        else:
            low_count += 1
            if low_count >= 3:
                left += 3
                break
        left -= 1
    left = max(0, left)
        
    # Expand right from peak (stop if low for 3 frames)
    right = peak_idx
    low_count = 0
    while right < len(rms) - 1:
        is_speech = (rms[right] > rms_thresh) or (rms[right] > noise_rms * 1.3 and zcr[right] > 0.12)
        if is_speech:
            low_count = 0
        else:
            low_count += 1
            if low_count >= 3:
                right -= 3
                break
        right += 1
    right = min(len(rms) - 1, right)
        
    start_s = max(0.0, left * 0.01 - pad_ms/1000)
    end_s = min(len(d)/sr, (right * 0.01 + win/sr) + pad_ms/1000)
    
    start_idx = int(start_s * sr)
    end_idx = int(end_s * sr)
    return start_idx, end_idx

records = []
print("--- PREPARING INVENTORY UNITS ---")
for word_key, label in word_map.items():
    for cid in [1, 2]:
        src_file = os.path.join(audio_src, f'{word_key}{cid}.m4a')
        dst_name = f'{word_key}_{cid:02d}.wav'
        dst_path = os.path.join(units_dir, dst_name)
        
        # Convert m4a to wav 16kHz mono using ffmpeg
        tmp_wav = os.path.join(units_dir, f'tmp_{dst_name}')
        cmd = ['ffmpeg', '-y', '-i', src_file, '-ar', '16000', '-ac', '1', tmp_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        raw_data, sr = sf.read(tmp_wav)
        s_idx, e_idx = detect_speech_boundaries(raw_data, sr)
        trimmed = raw_data[s_idx:e_idx]
        
        # Mild normalization to avoid clipping
        peak = np.max(np.abs(trimmed))
        if peak > 0.95:
            trimmed = trimmed * (0.95 / peak)
            
        sf.write(dst_path, trimmed.astype(np.float32), sr, subtype='PCM_16')
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
            
        dur = len(trimmed) / sr
        rms_val = np.sqrt(np.mean(trimmed**2))
        records.append({
            'word': label,
            'token': word_key,
            'candidate_id': cid,
            'filename': dst_name,
            'filepath': os.path.relpath(dst_path, base_dir),
            'duration_s': round(dur, 3),
            'rms': round(float(rms_val), 4),
            'sample_rate': sr
        })
        print(f"Saved {dst_name:<12} (word='{label}'): dur={dur:.3f}s, rms={rms_val:.4f}")

df = pd.DataFrame(records)
meta_path = os.path.join(base_dir, 'metadata.csv')
df.to_csv(meta_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Successfully prepared 32 units! Metadata saved to {meta_path}")
