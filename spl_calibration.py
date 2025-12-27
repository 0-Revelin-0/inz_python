import numpy as np
import sounddevice as sd



def generate_pink_noise(duration, fs, level_db=-1):
    """
    Generacja różowego szumu metodą filtracji białego szumu
    przez filtr 1/f.
    """
    n = int(duration * fs)

    # biały szum
    white = np.random.normal(0.0, 1.0, n)

    # FFT białego szumu
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1/fs)

    # filtr 1/f (pink noise)
    pink_filter = 1 / np.maximum(freqs, 1.0)  # unika dzielenia przez 0
    spectrum *= pink_filter

    pink = np.fft.irfft(spectrum)
    pink = pink / np.max(np.abs(pink) + 1e-12)

    # Ustaw poziom w dBFS
    scale = 10 ** (level_db / 20)
    pink *= scale

    return pink.astype(np.float32)



# =============================================================
# 2. ODTWARZANIE CIĄGŁEGO RÓŻOWEGO SZUMU
# =============================================================

class PinkNoisePlayer:
    def __init__(self):
        self.stream = None
        self.running = False
        self.noise = None
        self.idx = 0
        self.fs = 48000

    def start(self, audio_cfg, level_db=0):
        if self.running:
            return

        self.running = True
        self.fs = audio_cfg["sample_rate"]

        # generujemy duży (np. 10 s) bufor różowego szumu
        self.noise = generate_pink_noise(10.0, self.fs, level_db)
        self.idx = 0

        def callback(outdata, frames, time, status):
            if not self.running:
                outdata[:] = np.zeros((frames, 1), dtype=np.float32)
                return

            end = self.idx + frames

            # jeśli dojdziemy do końca → regenerujemy szum
            if end >= len(self.noise):
                self.noise = generate_pink_noise(10.0, self.fs, level_db)
                self.idx = 0
                end = frames

            chunk = self.noise[self.idx:end]
            outdata[:] = chunk.reshape(-1, 1)

            self.idx = end

        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=1,
            device=audio_cfg["output_device"],
            blocksize=audio_cfg["buffer_size"],
            dtype="float32",
            callback=callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.running = False
            self.stream.stop()
            self.stream.close()
        self.stream = None
        self.noise = None
        self.idx = 0



# =============================================================
# 3. POMIAR RMS / PEAK W CZASIE RZECZYWISTYM
# =============================================================

def measure_input_level(audio_cfg, duration=0.5):
    """
    Nagrywa sygnał przez 'duration' sekund i liczy:
      RMS, RMS[dBFS], PEAK, PEAK[dBFS]
    """
    fs = audio_cfg["sample_rate"]

    rec = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        device=audio_cfg["input_device"],
        dtype="float32",
    )
    sd.wait()
    x = rec[:, 0]

    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))

    rms_db = 20 * np.log10(max(rms, 1e-12))
    peak_db = 20 * np.log10(max(peak, 1e-12))

    return {
        "rms": float(rms),
        "peak": float(peak),
        "rms_db": float(rms_db),
        "peak_db": float(peak_db),
    }


class InputLevelMonitor:
    """
    Ciągły monitor RMS/Peak wejścia audio.
    Używany podczas kalibracji SPL na różowym szumie.
    """
    def __init__(self, audio_cfg, level_callback, block_duration=0.1):
        self.audio_cfg = audio_cfg
        self.callback = level_callback
        self.block_duration = block_duration
        self.running = False
        self.stream = None

    def start(self):
        if self.running:
            return

        self.running = True
        fs = self.audio_cfg["sample_rate"]
        blocksize = int(fs * self.block_duration)

        def audio_callback(indata, frames, time, status):
            if not self.running:
                return

            x = indata[:, 0]
            rms = float(np.sqrt(np.mean(x**2)))
            peak = float(np.max(np.abs(x)))

            rms_db = 20 * np.log10(max(rms, 1e-12))
            peak_db = 20 * np.log10(max(peak, 1e-12))

            self.callback(rms_db, peak_db)

        self.stream = sd.InputStream(
            samplerate=fs,
            channels=1,
            device=self.audio_cfg["input_device"],
            blocksize=blocksize,
            dtype="float32",
            callback=audio_callback
        )

        self.stream.start()

    def stop(self):
        if not self.running:
            return

        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stream = None


