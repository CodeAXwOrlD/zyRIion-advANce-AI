import sounddevice as sd
import numpy as np
import os

SAMPLE_RATE = 16000
DURATION = 5
SAVE_PATH = os.path.expanduser("~/Documents/zyrion/wake_engine/akhil_voice.npy")

def record_sample(i):
    print(f"\n🎤 Recording sample {i+1}/7 — speak for 5 seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ Done!")
    return audio.flatten()

def extract_features(audio):
    features = []
    frame_size = 512
    hop_size = 256

    zcr_list = []
    energy_list = []
    centroid_list = []
    spectral_list = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i+frame_size]

        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
        zcr_list.append(zcr)

        energy = np.mean(frame**2)
        energy_list.append(energy)

        spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
        freqs = np.fft.rfftfreq(frame_size, 1/SAMPLE_RATE)

        if spectrum.sum() > 0:
            centroid = np.sum(freqs * spectrum) / spectrum.sum()
        else:
            centroid = 0
        centroid_list.append(centroid)

        band_size = len(spectrum) // 8
        bands = [np.mean(spectrum[j*band_size:(j+1)*band_size]) 
                 for j in range(8)]
        spectral_list.append(bands)

    zcr_feat = np.array([np.mean(zcr_list), np.std(zcr_list)])
    energy_feat = np.array([np.mean(energy_list), np.std(energy_list)])
    centroid_feat = np.array([np.mean(centroid_list), np.std(centroid_list)])
    spectral_feat = np.mean(spectral_list, axis=0)

    if spectral_feat.max() > 0:
        spectral_feat = spectral_feat / spectral_feat.max()

    fingerprint = np.concatenate([zcr_feat, energy_feat,
                                   centroid_feat, spectral_feat])
    return fingerprint

def enroll():
    print("🔐 ZYRION Voice Enrollment!")
    print("Speak naturally for 5 seconds each time\n")
    all_features = []

    for i in range(7):
        input("Press ENTER when ready...")
        audio = record_sample(i)
        features = extract_features(audio)
        all_features.append(features)
        print(f"✅ Sample {i+1} saved!")

    min_len = min(len(f) for f in all_features)
    all_features = [f[:min_len] for f in all_features]
    avg_features = np.mean(all_features, axis=0)

    np.save(SAVE_PATH, {'features': avg_features})
    print(f"\n🔥 Voice fingerprint saved!")
    print("✅ Enrollment complete!")

if __name__ == "__main__":
    enroll()