import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
CHUNK = 4096

MIN_CLAP_GAP = 0.15        # ignore re-triggers closer than this (same clap's decay/ringing)
MAX_CLAP_GAP = 1.0         # max gap between 2 claps to count as "double clap"
HIGH_FREQ_CUTOFF_HZ = 1500
HIGH_FREQ_RATIO_MIN = 0.30 # claps are broadband/percussive — require real high-freq energy

def is_clap_like(chunk):
    """Distinguishes an actual clap (sharp, broadband, high-frequency) from generic
    loud noise like a fan spike, hum, or voice — only runs when volume already crossed
    threshold, so this stays cheap on CPU."""
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1 / SAMPLE_RATE)

    total_energy = np.sum(spectrum) + 1e-9
    high_energy = np.sum(spectrum[freqs > HIGH_FREQ_CUTOFF_HZ])

    return (high_energy / total_energy) > HIGH_FREQ_RATIO_MIN

def listen_for_claps(on_double_clap):
    print("👂 ZYRION listening for claps...")

    while True:
        # Calibrate ambient noise floor fresh each time (room conditions change)
        calib_stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK)
        calib_stream.start()
        levels = []
        for _ in range(5):
            chunk, _ = calib_stream.read(CHUNK)
            levels.append(np.abs(chunk).max())
        calib_stream.stop()
        calib_stream.close()

        noise_floor = float(np.median(levels))
        threshold = max(noise_floor * 4, 0.18)
        print(f"🔧 Room noise floor: {noise_floor:.3f} → clap threshold: {threshold:.3f}")

        clap_times = []
        last_clap_time = 0
        detected = {"flag": False}

        def callback(indata, frames, time_info, status):
            nonlocal last_clap_time, clap_times

            volume = np.abs(indata).max()
            now = time.time()

            if volume > threshold and (now - last_clap_time) > MIN_CLAP_GAP:
                if not is_clap_like(indata.flatten()):
                    return  # loud, but not clap-shaped — ignore

                last_clap_time = now
                clap_times.append(now)
                print(f"👏 Clap! volume={volume:.3f} total={len(clap_times)}")

                if len(clap_times) >= 2:
                    gap = clap_times[-1] - clap_times[-2]
                    if MIN_CLAP_GAP < gap < MAX_CLAP_GAP:
                        print("✅ DOUBLE CLAP DETECTED!")
                        detected["flag"] = True
                        raise sd.CallbackStop()
                    else:
                        clap_times = [clap_times[-1]]

        with sd.InputStream(callback=callback, channels=1,
                            samplerate=SAMPLE_RATE, blocksize=CHUNK):
            while not detected["flag"]:
                time.sleep(0.05)

        on_double_clap()

if __name__ == "__main__":
    def test():
        print("🔥 ZYRION WAKING UP!")
    listen_for_claps(test)
