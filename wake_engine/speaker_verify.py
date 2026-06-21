import sounddevice as sd
import numpy as np
import os

SAMPLE_RATE = 16000
DURATION = 2
THRESHOLD = 0.90
VOICE_PATH = os.path.expanduser("~/Documents/zyrion/wake_engine/akhil_voice.npy")

def record_voice():
    print("🎤 Speak now...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def extract_features(audio):
    """
    Multi-feature voice fingerprint:
    1. MFCC-like spectral features (frequency signature)
    2. Zero Crossing Rate (how voice vibrates)
    3. Energy envelope (volume pattern)
    4. Spectral centroid (voice brightness/tone)
    5. Pitch estimation (fundamental frequency)
    """
    features = []
    frame_size = 512
    hop_size = 256

    zcr_list = []
    energy_list = []
    centroid_list = []
    spectral_list = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i+frame_size]

        # 1. Zero Crossing Rate — unique per voice
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
        zcr_list.append(zcr)

        # 2. Energy
        energy = np.mean(frame**2)
        energy_list.append(energy)

        # 3. FFT spectrum
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
        freqs = np.fft.rfftfreq(frame_size, 1/SAMPLE_RATE)

        # 4. Spectral centroid — voice tone/brightness
        if spectrum.sum() > 0:
            centroid = np.sum(freqs * spectrum) / spectrum.sum()
        else:
            centroid = 0
        centroid_list.append(centroid)

        # 5. Spectral bands — voice frequency fingerprint
        # Split into 8 bands (like mini MFCC)
        band_size = len(spectrum) // 8
        bands = [np.mean(spectrum[j*band_size:(j+1)*band_size]) 
                 for j in range(8)]
        spectral_list.append(bands)

    # Combine all features into one fingerprint vector
    zcr_feat = np.array([np.mean(zcr_list), np.std(zcr_list)])
    energy_feat = np.array([np.mean(energy_list), np.std(energy_list)])
    centroid_feat = np.array([np.mean(centroid_list), np.std(centroid_list)])
    spectral_feat = np.mean(spectral_list, axis=0)

    # Normalize spectral
    if spectral_feat.max() > 0:
        spectral_feat = spectral_feat / spectral_feat.max()

    fingerprint = np.concatenate([zcr_feat, energy_feat, 
                                   centroid_feat, spectral_feat])
    return fingerprint

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def verify_speaker():
    if not os.path.exists(VOICE_PATH):
        print("❌ No voice enrolled! Run enroll_voice.py first.")
        return False

    saved = np.load(VOICE_PATH, allow_pickle=True).item()
    saved_features = saved['features']

    audio = record_voice()
    current_features = extract_features(audio)

    similarity = cosine_similarity(saved_features, current_features)
    print(f"🔍 Voice match score: {similarity:.4f}")

    if similarity >= THRESHOLD:
        print("✅ Welcome Akhil!")
        return True
    else:
        print("❌ Access Denied!")
        return False

if __name__ == "__main__":
    verify_speaker()
