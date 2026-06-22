import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
CHUNK = 4096

MIN_CLAP_GAP = 0.15        # ignore re-triggers closer than this (same clap's decay/ringing)
MAX_CLAP_GAP = 1.0         # max gap between 2 claps to count as "double clap"
HIGH_FREQ_CUTOFF_HZ = 1500
HIGH_FREQ_RATIO_MIN = 0.30 # claps are broadband/percussive — require real high-freq energy

# Pre-compute frequency mask once (same for every chunk of the same size)
_freqs = np.fft.rfftfreq(CHUNK, 1 / SAMPLE_RATE)
_high_freq_mask = _freqs > HIGH_FREQ_CUTOFF_HZ
_window = np.hanning(CHUNK)


def is_clap_like(chunk):
    """Distinguishes an actual clap (sharp, broadband, high-frequency) from generic
    loud noise like a fan spike, hum, or voice — only runs when volume already crossed
    threshold, so this stays cheap on CPU."""
    spectrum = np.abs(np.fft.rfft(chunk * _window))
    total_energy = np.sum(spectrum) + 1e-9
    high_energy = np.sum(spectrum[_high_freq_mask])
    return (high_energy / total_energy) > HIGH_FREQ_RATIO_MIN


def _calibrate_noise():
    """One-time ambient noise calibration at startup."""
    calib_stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK)
    calib_stream.start()
    levels = []
    for _ in range(5):
        chunk, _ = calib_stream.read(CHUNK)
        levels.append(np.abs(chunk).max())
    calib_stream.stop()
    calib_stream.close()

    noise_floor = float(np.median(levels))
    threshold = max(noise_floor * 3, 0.18)
    print(f"[CLAP] noise floor: {noise_floor:.3f} | threshold: {threshold:.3f}")
    return threshold


def listen_for_claps(on_double_clap):
    print("[CLAP] listening for double clap activation")

    # Calibrate once at startup — room noise doesn't change enough to re-measure every loop
    threshold = _calibrate_noise()

    while True:
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

                if len(clap_times) >= 2:
                    gap = clap_times[-1] - clap_times[-2]
                    if MIN_CLAP_GAP < gap < MAX_CLAP_GAP:
                        print("[CLAP] double clap detected — activating")
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
        print("[WAKE] activated")
    listen_for_claps(test)
