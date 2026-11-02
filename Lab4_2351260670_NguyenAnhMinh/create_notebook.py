import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(base_dir, 'Lab04.ipynb')

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

cells = []

# Cell 1: Title
cells.append(md_cell("""# TRƯỜNG ĐẠI HỌC THỦY LỢI
## KHOA CÔNG NGHỆ THÔNG TIN - BỘ MÔN TRÍ TUỆ NHÂN TẠO
### CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
---
# LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH - TTS)
**Từ kiến trúc TTS, ghép nối và unit selection đến tổng hợp tham số HMM và TTS học sâu**

- **Họ và tên sinh viên:** Nguyễn Anh Minh
- **Mã sinh viên:** 2351260670
- **Lớp / Khóa:** K65 - CNTT / AI"""))

# Cell 2: Section 1 Environment
cells.append(md_cell("""## 1. Thiết lập môi trường và nạp các thư viện cần thiết"""))
cells.append(code_cell("""import os
import sys
import time
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd

# Import các backend tự phát triển
from hmm_tts_backend import synthesize_hmm
from neural_tts_backend import synthesize as synthesize_neural

print("Môi trường và các thư viện đã sẵn sàng!")"""))

# Cell 3: Section 2 Inventory & Text Norm
cells.append(md_cell("""## 2. Phần A - Chuẩn bị Inventory và Text Normalization
- Case study miền hẹp: Hệ thống thông báo xếp hàng (16 từ vựng, mỗi từ có 2 bản ghi candidate).
- Âm thanh chuẩn hóa: 16 kHz, 16-bit PCM Mono, đã cắt sạch khoảng lặng ở 2 đầu."""))
cells.append(code_cell("""DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_queue_text(text):
    \"\"\"Chuẩn hóa câu thoại miền hẹp: lowercase, bỏ dấu câu, tách số thành chữ.\"\"\"
    text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    tokens = []
    for tok in text.split():
        if tok.isdigit():
            tokens.extend(DIGIT_MAP[c] for c in tok)
        else:
            tokens.append(tok)
    return tokens

test_sentence = "Mời khách hàng số 23 đến quầy 5."
tokens = normalize_queue_text(test_sentence)
print(f"Câu gốc: '{test_sentence}'")
print(f"Chuỗi token chuẩn hóa: {tokens}")"""))

# Cell 4: View metadata & Plot Histogram
cells.append(md_cell("""### 2.1. Thống kê thuộc tính Inventory (`metadata.csv`) và vẽ biểu đồ phân bố"""))
cells.append(code_cell("""df_meta = pd.read_csv('metadata.csv', encoding='utf-8-sig')
print("Thống kê 10 đơn vị âm đầu tiên:")
display(df_meta.head(10))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df_meta['duration_s'], bins=10, color='#2563eb', edgecolor='black', alpha=0.8)
plt.axvline(df_meta['duration_s'].mean(), color='red', linestyle='--', label=f"Mean: {df_meta['duration_s'].mean():.2f}s")
plt.title('Phân bố thời lượng đơn vị âm (Duration Histogram)', fontweight='bold')
plt.xlabel('Thời lượng (s)')
plt.ylabel('Số lượng')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df_meta['rms'], bins=10, color='#10b981', edgecolor='black', alpha=0.8)
plt.axvline(df_meta['rms'].mean(), color='red', linestyle='--', label=f"Mean: {df_meta['rms'].mean():.4f}")
plt.title('Phân bố năng lượng hiệu dụng (RMS Histogram)', fontweight='bold')
plt.xlabel('RMS')
plt.ylabel('Số lượng')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('figures/fig1_inventory_stats.png', dpi=200)
plt.show()"""))

# Cell 5: Section 3 Baseline Concatenation
cells.append(md_cell("""## 3. Phần B - Baseline Concatenation (Ghép thô - không smoothing)
Ghép cố định candidate 1 (`_01.wav`) cho từng token với `xfade_ms = 0`. Xuất file `raw_concat.wav`."""))
cells.append(code_cell("""WORD_TO_TOKEN = {
    'mời': 'moi', 'khách': 'khach', 'hàng': 'hang', 'số': 'so',
    'không': 'khong', 'một': 'mot', 'hai': 'hai', 'ba': 'ba', 'bốn': 'bon',
    'năm': 'nam', 'sáu': 'sau', 'bảy': 'bay', 'tám': 'tam', 'chín': 'chin',
    'đến': 'den', 'quầy': 'quay'
}

def load_unit(path, sr=16000):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    if fs != sr: x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return x

def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x, target=0.08):
    cur_rms = rms(x)
    if cur_rms < 1e-6: return x
    y = x * (target / cur_rms)
    return np.clip(y, -0.99, 0.99)

def crossfade(a, b, n):
    n = min(n, len(a), len(b))
    if n == 0: return np.concatenate([a, b])
    alpha = np.linspace(0.0, 1.0, n, endpoint=True)
    mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])

def concat_units(wav_list, sr=16000, xfade_ms=15):
    xs = [match_rms(w) for w in wav_list]
    y = xs[0]
    n = int(sr * xfade_ms / 1000)
    boundaries = [len(y) / sr]
    for x in xs[1:]:
        y = crossfade(y, x, n)
        boundaries.append(len(y) / sr)
    return y, boundaries

# Ghép baseline
cand1_wavs = [load_unit(f"lab4_data/units/{WORD_TO_TOKEN[t]}_01.wav") for t in tokens]
raw_audio, raw_boundaries = concat_units(cand1_wavs, xfade_ms=0)
sf.write('audio_output/raw_concat.wav', raw_audio, 16000)
print(f"Đã lưu audio_output/raw_concat.wav (Thời lượng: {len(raw_audio)/16000:.2f}s)")
ipd.Audio('audio_output/raw_concat.wav')"""))

# Cell 6: Section 4 Unit Selection & Smoothing
cells.append(md_cell("""## 4. Phần C - Unit Selection và Boundary Smoothing
- Tính **Join Cost** giữa 2 unit bằng sai lệch phổ log ($D_{spec}$) và năng lượng ($\Delta RMS$).
- Dùng **Quy hoạch động (Dynamic Programming - Viterbi)** để tìm chuỗi candidate có tổng chi phí nhỏ nhất.
- Thử nghiệm các mức cross-fade: 0 ms, 5 ms, 15 ms, 30 ms."""))
cells.append(code_cell("""def join_cost(a, b, sr=16000, ms=25, nfft=512):
    n = int(sr * ms / 1000)
    wa = np.hanning(min(n, len(a)))
    wb = np.hanning(min(n, len(b)))
    A = np.fft.rfft(a[-len(wa):] * wa, n=nfft)
    B = np.fft.rfft(b[:len(wb)] * wb, n=nfft)
    La = 20 * np.log10(np.maximum(np.abs(A), 1e-8))
    Lb = 20 * np.log10(np.maximum(np.abs(B), 1e-8))
    d_spec = float(np.mean((La - Lb) ** 2))
    rms_diff = abs(rms(a[-len(wa):]) - rms(b[:len(wb)]))
    return d_spec + 150.0 * rms_diff

cand_units = []
for tok in tokens:
    tok_key = WORD_TO_TOKEN[tok]
    u1 = load_unit(f"lab4_data/units/{tok_key}_01.wav")
    u2 = load_unit(f"lab4_data/units/{tok_key}_02.wav")
    cand_units.append([u1, u2])

T = len(tokens)
dp_cost = np.zeros((T, 2))
dp_backptr = np.zeros((T, 2), dtype=int)

for t in range(1, T):
    for c_curr in range(2):
        best_c = float('inf')
        best_prev = 0
        for c_prev in range(2):
            c_trans = join_cost(cand_units[t-1][c_prev], cand_units[t][c_curr])
            tot = dp_cost[t-1, c_prev] + c_trans
            if tot < best_c:
                best_c = tot
                best_prev = c_prev
        dp_cost[t, c_curr] = best_c
        dp_backptr[t, c_curr] = best_prev

best_last = int(np.argmin(dp_cost[T-1]))
opt_path = [best_last]
for t in range(T-1, 0, -1):
    best_last = dp_backptr[t, best_last]
    opt_path.append(best_last)
opt_path.reverse()

fixed_cost = sum(join_cost(cand_units[t-1][0], cand_units[t][0]) for t in range(1, T))
best_cost = float(np.min(dp_cost[T-1]))
print(f"Tổng Join Cost của Baseline (Candidate 1): {fixed_cost:.2f}")
print(f"Tổng Join Cost của Tối ưu (Unit Selection): {best_cost:.2f} (Giảm: {(fixed_cost-best_cost)/fixed_cost*100:.1f}%)")
print(f"Chuỗi candidate được chọn: {[f'{tokens[i]}_{opt_path[i]+1:02d}' for i in range(T)]}")

best_wav_list = [cand_units[i][opt_path[i]] for i in range(T)]
best_audio, best_boundaries = concat_units(best_wav_list, xfade_ms=15)
sf.write('audio_output/best_concat.wav', best_audio, 16000)
ipd.Audio('audio_output/best_concat.wav')"""))

# Cell 7: Plots for Boundary & Selection Cost
cells.append(md_cell("""### 4.1. Đồ thị Spectrogram biên nối và Ma trận Join Cost"""))
cells.append(code_cell("""# Hình 2: So sánh Waveform và Spectrogram biên nối
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex='col')
t_raw = np.linspace(0, len(raw_audio)/16000, len(raw_audio))
axes[0, 0].plot(t_raw, raw_audio, color='#2563eb', lw=0.6)
axes[0, 0].set_title('Raw Concat (xfade = 0 ms) - Có tiếng click và đứt gãy tại biên', fontweight='bold')
axes[0, 0].set_ylabel('Amplitude')
for b in raw_boundaries[:-1]: axes[0, 0].axvline(b, color='red', linestyle='--', alpha=0.7)
axes[0, 0].grid(True, linestyle=':', alpha=0.5)

t_best = np.linspace(0, len(best_audio)/16000, len(best_audio))
axes[1, 0].plot(t_best, best_audio, color='#10b981', lw=0.6)
axes[1, 0].set_title('Best Concat (Unit Selection + xfade = 15 ms) - Đường nối mượt mà', fontweight='bold')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].set_xlabel('Thời gian (giây)')
for b in best_boundaries[:-1]: axes[1, 0].axvline(b, color='red', linestyle='--', alpha=0.7)
axes[1, 0].grid(True, linestyle=':', alpha=0.5)

D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(raw_audio, n_fft=512, hop_length=128)), ref=np.max)
D_best = librosa.amplitude_to_db(np.abs(librosa.stft(best_audio, n_fft=512, hop_length=128)), ref=np.max)

axes[0, 1].imshow(D_raw, origin='lower', aspect='auto', cmap='magma', extent=[0, t_raw[-1], 0, 8000])
axes[0, 1].set_title('Spectrogram Ghép Thô (Vết cắt phổ đứt đoạn)', fontweight='bold')
axes[0, 1].set_ylabel('Tần số (Hz)')
for b in raw_boundaries[:-1]: axes[0, 1].axvline(b, color='cyan', linestyle='--', alpha=0.7)

axes[1, 1].imshow(D_best, origin='lower', aspect='auto', cmap='magma', extent=[0, t_best[-1], 0, 8000])
axes[1, 1].set_title('Spectrogram Sau Làm Mịn (Chuyển tiếp phổ liên tục)', fontweight='bold')
axes[1, 1].set_ylabel('Tần số (Hz)')
axes[1, 1].set_xlabel('Thời gian (giây)')
for b in best_boundaries[:-1]: axes[1, 1].axvline(b, color='cyan', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/fig2_concat_spectrogram.png', dpi=200)
plt.show()"""))

# Cell 8: Section 5 Prosody & Source-Filter
cells.append(md_cell("""## 5. Phần D - Prosody và Phân tích Mô hình Source–Filter
1. Đo tần số cơ bản $F_0$ trên đoạn nguyên âm tuần hoàn và kiểm chứng $F_0 \\approx F_s / N_p$.
2. Thử nghiệm Time-stretch ($0.8\\times, 1.2\\times$) và Pitch-shift ($-2, +2$ semitones)."""))
cells.append(code_cell("""# Lấy unit 'moi_01.wav' phân tích pitch
sample_w, sr = sf.read('lab4_data/units/moi_01.wav')
# Đoạn nguyên âm ổn định ở giữa từ
vowel_seg = sample_w[int(0.18*sr):int(0.24*sr)]

# Tìm khoảng cách giữa các đỉnh chu kỳ tuần hoàn (Np)
peaks, _ = signal.find_peaks(vowel_seg, distance=int(sr/300), prominence=0.03)
if len(peaks) >= 2:
    np_diffs = np.diff(peaks)
    np_mean = np.mean(np_diffs)
    f0_est = sr / np_mean
    print(f"Chu kỳ pitch trung bình: Np = {np_mean:.1f} mẫu")
    print(f"Tần số cơ bản F0 = Fs / Np = {sr} / {np_mean:.1f} = {f0_est:.1f} Hz")

# Thí nghiệm Pitch Shift và Time Stretch
y_stretch_fast = librosa.effects.time_stretch(sample_w, rate=1.2)
y_stretch_slow = librosa.effects.time_stretch(sample_w, rate=0.8)
y_pitch_up = librosa.effects.pitch_shift(sample_w, sr=sr, n_steps=2)
y_pitch_down = librosa.effects.pitch_shift(sample_w, sr=sr, n_steps=-2)

print("Biến đổi Prosody thành công!")"""))

# Cell 9: Section 6 HMM Parametric Synthesis
cells.append(md_cell("""## 6. Phần E - Tổng hợp tham số HMM (HMM-based Synthesis)
- HMM sinh ra chuỗi tham số âm học liên tục (F0, formant $F_1, F_2, F_3$) qua từng frame 10 ms.
- Sử dụng ràng buộc delta feature để đường tham số trơn tru (MLPG).
- Resynthesis thông qua mô hình kích thích nguồn - bộ lọc cộng hưởng."""))
cells.append(code_cell("""hmm_audio, hmm_sr, tracks = synthesize_hmm(test_sentence, sr=16000)
sf.write('audio_output/hmm_tts.wav', hmm_audio, hmm_sr)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t_f = tracks['frames_time']

ax1.plot(t_f, tracks['f0_raw'], 'r--', alpha=0.5, label='F0 tĩnh (State Mean)')
ax1.plot(t_f, tracks['f0'], 'b-', lw=1.8, label='F0 trơn nhẵn (MLPG/Delta)')
ax1.set_title('Quỹ đạo tham số âm học HMM (HMM Parameter Trajectories)', fontweight='bold')
ax1.set_ylabel('Cao độ F0 (Hz)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

ax2.plot(t_f, tracks['f1'], color='purple', lw=1.5, label='Formant F1')
ax2.plot(t_f, tracks['f2'], color='darkorange', lw=1.5, label='Formant F2')
ax2.plot(t_f, tracks['f3'], color='green', lw=1.2, label='Formant F3')
ax2.set_ylabel('Formant (Hz)')
ax2.set_xlabel('Thời gian (giây)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/fig4_hmm_trajectories.png', dpi=200)
plt.show()

ipd.Audio('audio_output/hmm_tts.wav')"""))

# Cell 10: Section 7 Neural TTS
cells.append(md_cell("""## 7. Phần F - TTS Học Sâu với Mô hình Pretrained (Neural TTS)
Thực thi inference mô hình Deep Learning TTS, trích xuất Mel-spectrogram và đo đạc độ trễ (Latency, RTF)."""))
cells.append(code_cell("""nn_audio, nn_sr, nn_extra = synthesize_neural(test_sentence, sr=16000)
sf.write('audio_output/neural_tts.wav', nn_audio, nn_sr)

print(f"Mô hình: {nn_extra['model_name']}")
print(f"Thời gian sinh (Latency): {nn_extra['latency_s']:.3f} s")
print(f"Thời lượng audio: {nn_extra['duration_s']:.2f} s")
print(f"Real-Time Factor (RTF): {nn_extra['rtf']:.4f}")

ipd.Audio('audio_output/neural_tts.wav')"""))

# Cell 11: Section 8 Evaluation & Comparison
cells.append(md_cell("""## 8. Phần G - Đánh giá, So sánh 3 phương pháp và Listening Test"""))
cells.append(code_cell("""# Bảng so sánh đặc tính kỹ thuật
comparison_data = {
    'Tiêu chí': ['Intelligibility (Độ rõ)', 'Naturalness (Độ tự nhiên)', 'Continuity (Tính liên tục)', 'Latency (Độ trễ)', 'Bộ nhớ / Dữ liệu'],
    'Concatenative TTS': ['Rất cao (miền hẹp)', 'Trung bình (thiếu ngữ điệu)', 'Dễ click/jump tại biên', 'Cực nhanh (< 50ms)', 'Lớn (kho WAV)'],
    'HMM Parametric TTS': ['Tốt (dễ hiểu)', 'Thấp (robotic/buzzy)', 'Rất liên tục', 'Rất nhanh (< 100ms)', 'Rất nhỏ (< 5MB)'],
    'Neural TTS': ['Rất cao', 'Rất cao (tự nhiên)', 'Hoàn hảo', 'Trung bình (0.1 - 0.5s)', 'Lớn (Model weights)']
}
display(pd.DataFrame(comparison_data))

# Đọc kết quả listening test
df_ratings = pd.read_csv('ratings.csv', encoding='utf-8-sig')
print("Kết quả đánh giá Listening Test (MOS):")
display(df_ratings.groupby('method')[['intelligibility', 'naturalness', 'continuity', 'overall_mos']].mean())"""))

# Cell 12: Theory Questions
cells.append(md_cell("""## 9. Trả lời chi tiết 10 câu hỏi báo cáo lý thuyết (Mục 7)

### Câu 1: Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?
**Trả lời:** Mặc dù từng đơn vị từ được ghi âm từ giọng người thật, nhưng khi ghép nối cơ học, câu tổng thể vẫn mất tự nhiên do:
1. **Thiếu ngữ điệu toàn câu (Global Prosody):** Mỗi từ được nói cô lập sẽ có contour cao độ (F0) và nhịp điệu (duration/energy) riêng, không tuân theo ngữ điệu giảm dần tự nhiên của một phát ngôn hoàn chỉnh (declination).
2. **Không có hiện tượng đồng cấu âm (Coarticulation):** Trong lời nói tự nhiên, khẩu hình chuyển tiếp liên tục giữa các âm cạnh nhau. Ghép nối tạo ra bước nhảy phổ đột ngột giữa đuôi từ trước và đầu từ sau.

### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?
**Trả lời:**
- **Target cost ($C_{target}$):** Đo lường mức độ phù hợp giữa candidate trong cơ sở dữ liệu với yêu cầu âm vị học/ngữ điệu mong muốn (ví dụ: đúng âm, đúng ngữ cảnh trọng âm, duration và pitch mục tiêu).
- **Join cost ($C_{join}$):** Đo lường độ trơn tru, liền mạch khi nối 2 candidate cạnh nhau (chênh lệch năng lượng, F0, và sai lệch phổ log tại biên ghép).
- **Hệ quả nếu chỉ tối ưu join cost:** Hệ thống có thể chọn một chuỗi các unit nối với nhau cực kỳ êm (join cost rất thấp) nhưng sai lệch hoàn toàn về ngữ điệu mục tiêu (ví dụ tốc độ đọc quá chậm, pitch không phù hợp ngữ cảnh, hoặc thậm chí chọn nhầm âm sắc).

### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?
**Trả lời:** Cross-fade thực chất là cộng chập có trọng số của hai đoạn tín hiệu chồng lấn. Các phụ âm (đặc biệt là phụ âm tắc plosives như /t, k, p, b, d/ và phụ âm xát fricatives như /s, ch/) có đặc trưng chuyển tiếp thời gian cực kỳ ngắn (burst 5–20 ms). Nếu vùng cross-fade kéo dài (ví dụ > 30–50 ms), năng lượng phụ âm sẽ bị phân tán, chồng lấn với âm kề bên gây hiện tượng âm vang giả (ghosting), làm suy giảm độ tương phản phổ và làm mờ phụ âm.

### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.
**Trả lời:** Theo mô hình Nguồn - Bộ lọc ($X(z) = E(z)H(z)$):
- **Nguồn kích thích $E(z)$ (Dây thanh quản):** Quyết định tần số dao động cơ bản $F_0$ (tần số của các vạch sóng hài cách nhau khoảng $F_0$).
- **Bộ lọc âm học $H(z)$ (Khoang miệng, hầu, mũi):** Quyết định các tần số cộng hưởng gọi là **Formant ($F_1, F_2, F_3$)**, tạo nên âm sắc và định danh nguyên âm.
- Hai thành phần này độc lập về mặt vật lý: Khi ta nâng cao giọng nói ($F_0$ tăng), các vạch sóng hài dãn xa nhau hơn nhưng đường bao phổ (spectral envelope) và vị trí đỉnh formant của bộ lọc $H(z)$ vẫn phải giữ nguyên. Nếu thay đổi $F_0$ mà dịch chuyển cả formant sẽ tạo ra hiệu ứng méo tiếng "chipmunk".

### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?
**Trả lời:**
- **Cơ chế PSOLA:** Phân rã tín hiệu hữu thanh thành các đoạn ngắn quanh vị trí pitch mark (pitch synchronous) bằng cửa sổ Hanning. Để đổi pitch, ta thay đổi khoảng cách dịch giữa các frame trước khi cộng chập (Overlap-Add).
- **Hiện tượng buzziness ở voiced fricatives:** Voiced fricatives (phụ âm xát hữu thanh như /v, z/) vừa có thành phần tuần hoàn (dây thanh rung) vừa có thành phần nhiễu ngẫu nhiên (luồng khí xát qua khe hẹp). Khi kéo dài thời lượng bằng cách lặp lại các frame tuần hoàn, thành phần nhiễu ngẫu nhiên bị biến thành tuần hoàn cưỡng bức, làm mất tính ngẫu nhiên tự nhiên và gây ra tiếng rè cơ học (robotic buzziness).

### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?
**Trả lời:** Nếu chỉ dùng đặc trưng tĩnh (static features), mỗi trạng thái HMM sẽ sinh ra giá trị trung bình cố định $\mu$, dẫn đến đường tham số dạng bậc thang gãy khúc khi chuyển đổi giữa các state. Khi bổ sung đạo hàm bậc một ($\Delta$) và bậc hai ($\Delta\Delta$) vào bài toán tối ưu cực đại hợp lý (Maximum Likelihood Parameter Generation - MLPG):
$$\hat{O}^* = \arg\max_O P(O | \lambda, q)$$
với ràng buộc toán học liên kết giữa tĩnh và động ($\Delta c_t \approx \frac{c_{t+1} - c_{t-1}}{2}$), nghiệm tối ưu buộc các tham số âm học phải biến thiên liên tục và trơn nhẵn theo thời gian.

### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?
**Trả lời:** Ghép nối sóng nguyên bản (whole waveform) không can thiệp biến đổi tín hiệu để tránh méo tiếng. Do đó, để câu nói tự nhiên, kho âm phải chứa sẵn các đoạn âm ghi lại trong đầy đủ các ngữ cảnh ngữ điệu (prosodic context: đầu câu, giữa câu, cuối câu, câu hỏi, câu khẳng định) và ngữ cảnh đồng cấu âm (coarticulation context). Số lượng tổ hợp ngữ cảnh bùng nổ cấp số nhân đòi hỏi kho dữ liệu ghi âm cực lớn.

### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.
**Trả lời:**
1. **Biểu diễn âm thanh:** Concatenative chọn và ghép các đoạn sóng WAV vật lý có sẵn từ người thật; trong khi Neural TTS học ánh xạ hàm phi tuyến phức tạp từ text sang biểu diễn trung gian (Mel-spectrogram) và dùng mạng nơ-ron sâu (Neural Vocoder như HiFi-GAN) để tổng hợp lại waveform từ đầu.
2. **Khả năng khái quát hóa và độ linh hoạt:** Concatenative hoàn toàn bất lực ngoài miền từ vựng đã thu; trong khi Neural TTS có thể đọc mọi từ vựng mới nhờ phonetic/char embedding, dễ dàng điều khiển cảm xúc, tốc độ và sao chép giọng nói (voice cloning) chỉ với vài giây mẫu.

### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?
**Trả lời:** Vì việc đánh giá khoa học đòi hỏi tính **khách quan và công bằng (fair comparison)**:
- Một mô hình ngẫu nhiên tải từ mạng có thể được huấn luyện trên hàng nghìn giờ dữ liệu studio cao cấp với người đọc chuyên nghiệp, trong khi kho âm concatenative chỉ có vài phút do sinh viên tự ghi.
- So sánh như vậy sẽ làm sai lệch bản chất thuật toán và không đánh giá đúng năng lực xử lý cục bộ của pipeline thực hành.

### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.
**Trả lời:**
- Hệ thống này có âm thanh rất êm tai, ấm áp, nhịp điệu tự nhiên (MOS cao) nhưng các âm vị phát âm không rõ ràng hoặc dễ nhầm từ (độ rõ lời thấp).
- **Ứng dụng phù hợp:** Các ứng dụng tạo âm thanh nền thư giãn (Ambient audio / ASMR), giọng thì thầm ru ngủ (Sleep storytelling / Meditation apps), nhạc kịch lồng tiếng cho nhân vật hoạt hình hư cấu nói ngôn ngữ giả tưởng (như tiếng người ngoài hành tinh, tiếng Simlish trong game The Sims), nơi cảm xúc và độ êm dịu quan trọng hơn độ chính xác của ngữ nghĩa."""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Successfully generated complete Jupyter Notebook at: {nb_path}")
