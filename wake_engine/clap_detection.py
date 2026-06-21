import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
THRESHOLD = 0.25
MIN_CLAP_GAP = 0.1
MAX_CLAP_GAP = 1.0
CHUNK = 4096

def listen_for_claps(on_double_clap):
    print("👂 ZYRION listening for claps...")
    print(f"Threshold set to: {THRESHOLD}")
    
    clap_times = []
    last_clap_time = 0
    cooldown = 0.1

    def callback(indata, frames, time_info, status):
        nonlocal last_clap_time, clap_times
        
        volume = np.abs(indata).max()  # peak volume not average
        now = time.time()

        if volume > THRESHOLD and (now - last_clap_time) > cooldown:
            last_clap_time = now
            clap_times.append(now)
            print(f"👏 Clap! volume={volume:.3f} total={len(clap_times)}")

            if len(clap_times) >= 2:
                gap = clap_times[-1] - clap_times[-2]
                if MIN_CLAP_GAP < gap < MAX_CLAP_GAP:
                    print("✅ DOUBLE CLAP DETECTED!")
                    clap_times.clear()
                    on_double_clap()
                elif gap >= MAX_CLAP_GAP:
                    clap_times = [clap_times[-1]]

    with sd.InputStream(callback=callback, channels=1,
                        samplerate=SAMPLE_RATE, blocksize=CHUNK):
        while True:
            time.sleep(0.05)

if __name__ == "__main__":
    def test():
        print("🔥 ZYRION WAKING UP!")
    listen_for_claps(test)
