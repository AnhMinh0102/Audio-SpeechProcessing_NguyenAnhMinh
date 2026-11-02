import numpy as np
import scipy.signal as signal

def normalize_text_to_tokens(text):
    text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    digit_map = {
        "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
        "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
    }
    tokens = []
    for tok in text.split():
        if tok.isdigit():
            tokens.extend(digit_map[c] for c in tok)
        else:
            tokens.append(tok)
    return tokens

# Phonetic / acoustic models for queue domain words
# Formants: F1, F2, F3 (Hz), voiced ratio, base F0 contour shape
UNIT_ACOUSTIC_PROFILES = {
    'mời':   {'dur_base': 0.45, 'f1': 480, 'f2': 1800, 'f3': 2600, 'f0_mean': 165, 'f0_trend': -15, 'voiced': True},
    'khách': {'dur_base': 0.32, 'f1': 680, 'f2': 1400, 'f3': 2500, 'f0_mean': 190, 'f0_trend': +25, 'voiced': True},
    'hàng':  {'dur_base': 0.50, 'f1': 750, 'f2': 1250, 'f3': 2400, 'f0_mean': 150, 'f0_trend': -20, 'voiced': True},
    'số':    {'dur_base': 0.45, 'f1': 450, 'f2': 1950, 'f3': 2700, 'f0_mean': 185, 'f0_trend': +25, 'voiced': True},
    'không': {'dur_base': 0.42, 'f1': 400, 'f2': 1000, 'f3': 2300, 'f0_mean': 160, 'f0_trend': 0,   'voiced': True},
    'một':   {'dur_base': 0.32, 'f1': 520, 'f2': 1100, 'f3': 2350, 'f0_mean': 155, 'f0_trend': -30, 'voiced': True},
    'hai':   {'dur_base': 0.46, 'f1': 700, 'f2': 1850, 'f3': 2650, 'f0_mean': 165, 'f0_trend': 0,   'voiced': True},
    'ba':    {'dur_base': 0.44, 'f1': 750, 'f2': 1200, 'f3': 2450, 'f0_mean': 162, 'f0_trend': 0,   'voiced': True},
    'bốn':   {'dur_base': 0.46, 'f1': 480, 'f2': 1050, 'f3': 2400, 'f0_mean': 180, 'f0_trend': +20, 'voiced': True},
    'năm':   {'dur_base': 0.48, 'f1': 720, 'f2': 1300, 'f3': 2500, 'f0_mean': 160, 'f0_trend': 0,   'voiced': True},
    'sáu':   {'dur_base': 0.48, 'f1': 700, 'f2': 1450, 'f3': 2550, 'f0_mean': 185, 'f0_trend': +20, 'voiced': True},
    'bảy':   {'dur_base': 0.46, 'f1': 600, 'f2': 1750, 'f3': 2600, 'f0_mean': 155, 'f0_trend': -15, 'voiced': True},
    'tám':   {'dur_base': 0.45, 'f1': 740, 'f2': 1250, 'f3': 2450, 'f0_mean': 185, 'f0_trend': +20, 'voiced': True},
    'chín':  {'dur_base': 0.48, 'f1': 380, 'f2': 2100, 'f3': 2800, 'f0_mean': 185, 'f0_trend': +20, 'voiced': True},
    'đến':   {'dur_base': 0.40, 'f1': 520, 'f2': 1900, 'f3': 2650, 'f0_mean': 180, 'f0_trend': +20, 'voiced': True},
    'quầy':  {'dur_base': 0.48, 'f1': 550, 'f2': 1600, 'f3': 2500, 'f0_mean': 150, 'f0_trend': -20, 'voiced': True},
}

def synthesize_hmm(text, sr=16000, frame_shift_ms=10, duration_scale=1.0, f0_scale=1.0):
    """
    Parametric HMM Speech Synthesis simulation (CSE457 Lab 4).
    Simulates:
    1. Phone / word state sequence generation
    2. Duration model predicting frame duration per state
    3. Maximum Likelihood Parameter Generation (MLPG) with delta constraints
       yielding smooth continuous trajectories for F0, F1, F2, F3
    4. Source-Filter resynthesis using impulse train / noise excitation and all-pole resonator.
    """
    tokens = normalize_text_to_tokens(text)
    frame_shift_s = frame_shift_ms / 1000.0
    
    # 1. State sequence & duration generation
    total_frames = 0
    token_durations = []
    token_profiles = []
    for tok in tokens:
        prof = UNIT_ACOUSTIC_PROFILES.get(tok, UNIT_ACOUSTIC_PROFILES['mời'])
        dur_s = prof['dur_base'] * duration_scale
        n_frames = max(10, int(round(dur_s / frame_shift_s)))
        token_durations.append(n_frames)
        token_profiles.append(prof)
        total_frames += n_frames

    # 2. Parameter trajectory generation (F0, F1, F2, F3, Energy)
    f0_track = np.zeros(total_frames)
    f1_track = np.zeros(total_frames)
    f2_track = np.zeros(total_frames)
    f3_track = np.zeros(total_frames)
    energy_track = np.zeros(total_frames)
    
    current_frame = 0
    for tok, n_frames, prof in zip(tokens, token_durations, token_profiles):
        t_norm = np.linspace(0, 1, n_frames)
        # F0 contour with intonation trend
        base_f0 = prof['f0_mean'] * f0_scale
        trend = prof['f0_trend'] * f0_scale
        # parabolic tone curve
        f0_curve = base_f0 + trend * (t_norm - 0.5) + 5 * np.sin(np.pi * t_norm)
        
        # Formant contours with slight co-articulation drift
        f1_curve = prof['f1'] + 20 * np.sin(np.pi * t_norm)
        f2_curve = prof['f2'] + 40 * np.cos(np.pi * t_norm)
        f3_curve = prof['f3'] * np.ones(n_frames)
        
        # Energy contour: smooth attack and decay
        energy_curve = 0.08 * np.sin(np.pi * t_norm)**0.6
        
        f0_track[current_frame:current_frame + n_frames] = f0_curve
        f1_track[current_frame:current_frame + n_frames] = f1_curve
        f2_track[current_frame:current_frame + n_frames] = f2_curve
        f3_track[current_frame:current_frame + n_frames] = f3_curve
        energy_track[current_frame:current_frame + n_frames] = energy_curve
        current_frame += n_frames

    # Dynamic smoothing filter (simulating MLPG with delta features)
    smooth_win = int(7)
    kernel = signal.windows.gaussian(smooth_win, std=2.0)
    kernel /= np.sum(kernel)
    f0_smooth = signal.convolve(f0_track, kernel, mode='same')
    f1_smooth = signal.convolve(f1_track, kernel, mode='same')
    f2_smooth = signal.convolve(f2_track, kernel, mode='same')
    f3_smooth = signal.convolve(f3_track, kernel, mode='same')
    energy_smooth = signal.convolve(energy_track, kernel, mode='same')

    # 3. Source-Filter Resynthesis
    hop_length = int(sr * frame_shift_s)
    total_samples = total_frames * hop_length
    waveform = np.zeros(total_samples, dtype=np.float32)
    
    phase = 0.0
    bandwidths = [70, 90, 120] # Hz formant bandwidths
    
    # Generate excitation
    excitation = np.zeros(total_samples, dtype=np.float32)
    for i in range(total_samples):
        frame_idx = min(i // hop_length, total_frames - 1)
        cur_f0 = f0_smooth[frame_idx]
        cur_energy = energy_smooth[frame_idx]
        
        if cur_f0 > 40: # voiced frame
            phase += 2 * np.pi * cur_f0 / sr
            if phase >= 2 * np.pi:
                phase -= 2 * np.pi
                excitation[i] = cur_energy * 10.0 # impulse
            else:
                excitation[i] = 0.0
        else: # unvoiced frame
            excitation[i] = cur_energy * np.random.randn()
            
    # Apply time-varying vocal tract filter via cascaded second-order all-pole resonators
    y = np.copy(excitation)
    for formant_track, bw in zip([f1_smooth, f2_smooth, f3_smooth], bandwidths):
        r = np.exp(-np.pi * bw / sr)
        filtered = np.zeros_like(y)
        for i in range(2, total_samples):
            frame_idx = min(i // hop_length, total_frames - 1)
            theta = 2 * np.pi * formant_track[frame_idx] / sr
            a1 = -2 * r * np.cos(theta)
            a2 = r * r
            filtered[i] = y[i] - a1 * filtered[i-1] - a2 * filtered[i-2]
        y = filtered * (1 - r) # gain normalization

    # Lip radiation filter (first-order difference)
    y = np.diff(y, prepend=0)
    # Normalize peak
    peak = np.max(np.abs(y)) + 1e-8
    y = (y / peak * 0.75).astype(np.float32)

    frames_time = np.arange(total_frames) * frame_shift_s
    tracks = {
        'f0': f0_smooth,
        'f0_raw': f0_track,
        'f1': f1_smooth,
        'f2': f2_smooth,
        'f3': f3_smooth,
        'energy': energy_smooth,
        'frames_time': frames_time,
        'tokens': tokens,
        'token_durations': token_durations
    }
    return y, sr, tracks
