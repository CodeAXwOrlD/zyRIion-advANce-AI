"""
Shared speaker-embedding module using MFCCs (Mel-Frequency Cepstral Coefficients).

Computes a rich speaker fingerprint using ONLY numpy — no PyTorch, no extra
dependencies. MFCCs capture vocal tract shape, pitch patterns, and speaking
dynamics, producing ~110 features that are far more discriminative than raw
spectral features.

Both enroll_voice.py and speaker_verify.py import from here.
"""

import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40
FRAME_SIZE = 512
HOP_SIZE = 256
F_MIN = 80.0
F_MAX = 7600.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max):
    """Build a Mel-scale triangular filterbank (pure numpy)."""
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def _compute_mfccs(audio):
    """Compute MFCCs frame-by-frame from raw audio using DCT-II."""
    # Pre-emphasis to boost high frequencies (speaker-discriminative)
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    n_frames = 1 + (len(emphasized) - FRAME_SIZE) // HOP_SIZE
    if n_frames < 1:
        return np.zeros((1, N_MFCC))

    frames = np.zeros((n_frames, FRAME_SIZE))
    for i in range(n_frames):
        start = i * HOP_SIZE
        frames[i] = emphasized[start:start + FRAME_SIZE]

    # Apply Hamming window
    window = np.hamming(FRAME_SIZE)
    frames *= window

    # FFT → power spectrum
    spectra = np.abs(np.fft.rfft(frames, n=FRAME_SIZE)) ** 2

    # Mel filterbank
    mel_fb = _mel_filterbank(FRAME_SIZE, N_MELS, SAMPLE_RATE, F_MIN, F_MAX)
    mel_energies = np.dot(spectra, mel_fb.T)
    mel_energies = np.maximum(mel_energies, 1e-10)  # avoid log(0)
    log_mel = np.log(mel_energies)

    # DCT-II to get MFCCs (using numpy — equivalent to scipy.fft.dct)
    n = log_mel.shape[1]
    dct_matrix = np.zeros((N_MFCC, n))
    for k in range(N_MFCC):
        for i in range(n):
            dct_matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    dct_matrix *= np.sqrt(2.0 / n)
    dct_matrix[0] *= 1.0 / np.sqrt(2.0)

    mfccs = np.dot(log_mel, dct_matrix.T)
    return mfccs


def _compute_deltas(features, width=2):
    """Compute delta (velocity) features for temporal dynamics."""
    n_frames, n_features = features.shape
    if n_frames < 2 * width + 1:
        return np.zeros_like(features)

    padded = np.pad(features, ((width, width), (0, 0)), mode='edge')
    deltas = np.zeros_like(features)
    denom = 2 * sum(t ** 2 for t in range(1, width + 1))

    for t in range(n_frames):
        for w in range(1, width + 1):
            deltas[t] += w * (padded[t + width + w] - padded[t + width - w])
        deltas[t] /= denom

    return deltas


def _pitch_features(audio):
    """Estimate F0 (fundamental frequency) statistics via autocorrelation."""
    frame_size = 1024
    hop = 512
    f0_values = []

    for i in range(0, len(audio) - frame_size, hop):
        frame = audio[i:i + frame_size]
        if np.max(np.abs(frame)) < 0.01:
            continue  # skip silence

        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-9)

        # Find first peak after the initial drop (min lag = 20 → max F0 = 800Hz)
        min_lag = SAMPLE_RATE // 800
        max_lag = SAMPLE_RATE // 60  # min F0 = 60Hz

        if max_lag >= len(corr):
            max_lag = len(corr) - 1

        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue

        peak_idx = np.argmax(search) + min_lag
        if corr[peak_idx] > 0.3:  # confidence threshold
            f0 = SAMPLE_RATE / peak_idx
            f0_values.append(f0)

    if len(f0_values) < 3:
        return np.zeros(4, dtype=np.float32)

    f0_arr = np.array(f0_values)
    return np.array([
        np.mean(f0_arr),
        np.std(f0_arr),
        np.median(f0_arr),
        np.percentile(f0_arr, 90) - np.percentile(f0_arr, 10),  # range
    ], dtype=np.float32)


def _spectral_contrast(audio):
    """Compute spectral contrast across 7 frequency bands."""
    n_frames = 1 + (len(audio) - FRAME_SIZE) // HOP_SIZE
    if n_frames < 1:
        return np.zeros(7, dtype=np.float32)

    contrasts = []
    for i in range(n_frames):
        start = i * HOP_SIZE
        frame = audio[start:start + FRAME_SIZE]
        spectrum = np.abs(np.fft.rfft(frame * np.hamming(FRAME_SIZE)))

        # Split into 7 bands
        band_size = len(spectrum) // 7
        band_contrasts = []
        for b in range(7):
            band = np.sort(spectrum[b * band_size:(b + 1) * band_size])
            if len(band) < 4:
                band_contrasts.append(0.0)
                continue
            n_top = max(1, len(band) // 4)
            peak = np.mean(band[-n_top:])
            valley = np.mean(band[:n_top])
            band_contrasts.append(np.log1p(peak) - np.log1p(valley))
        contrasts.append(band_contrasts)

    return np.mean(contrasts, axis=0).astype(np.float32)


def extract_embedding(audio_np: np.ndarray) -> np.ndarray:
    """
    Extract a ~110-dimensional speaker embedding from raw audio.

    Features computed:
    - 13 MFCCs × 4 statistics (mean, std, skew, kurtosis) = 52
    - 13 delta-MFCCs × 4 statistics = 52
    - Pitch/F0 statistics = 4
    - Spectral contrast (7 bands) = 7
    Total: ~115 features

    Parameters
    ----------
    audio_np : np.ndarray, shape (N,), dtype float32, sample-rate 16 kHz

    Returns
    -------
    np.ndarray, shape (~115,)
    """
    if len(audio_np) == 0:
        return np.zeros(115, dtype=np.float32)

    # 1. MFCCs: captures vocal tract shape (formants)
    mfccs = _compute_mfccs(audio_np)

    # Statistics over all frames
    mfcc_mean = np.mean(mfccs, axis=0)
    mfcc_std = np.std(mfccs, axis=0)

    # Skewness and kurtosis (capture asymmetry and peakiness of distributions)
    mfcc_centered = mfccs - mfcc_mean
    n = mfccs.shape[0]
    if n > 2:
        mfcc_skew = np.mean(mfcc_centered ** 3, axis=0) / (mfcc_std ** 3 + 1e-9)
        mfcc_kurt = np.mean(mfcc_centered ** 4, axis=0) / (mfcc_std ** 4 + 1e-9) - 3
    else:
        mfcc_skew = np.zeros(N_MFCC)
        mfcc_kurt = np.zeros(N_MFCC)

    # 2. Delta-MFCCs: captures speaking dynamics / temporal patterns
    deltas = _compute_deltas(mfccs)
    delta_mean = np.mean(deltas, axis=0)
    delta_std = np.std(deltas, axis=0)
    delta_centered = deltas - delta_mean
    if n > 2:
        delta_skew = np.mean(delta_centered ** 3, axis=0) / (delta_std ** 3 + 1e-9)
        delta_kurt = np.mean(delta_centered ** 4, axis=0) / (delta_std ** 4 + 1e-9) - 3
    else:
        delta_skew = np.zeros(N_MFCC)
        delta_kurt = np.zeros(N_MFCC)

    # 3. Pitch (F0): captures voice pitch characteristics
    pitch_feat = _pitch_features(audio_np)

    # 4. Spectral contrast: captures voice texture / timbre
    contrast_feat = _spectral_contrast(audio_np)

    # Combine all features
    embedding = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurt,          # 52
        delta_mean, delta_std, delta_skew, delta_kurt,        # 52
        pitch_feat,                                            # 4
        contrast_feat,                                         # 7
    ])

    return embedding.astype(np.float32)
