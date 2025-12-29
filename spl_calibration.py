import numpy as np
import sounddevice as sd


def set_rms_dbfs(x: np.ndarray, target_dbfs: float, eps: float = 1e-12) -> np.ndarray:
    """Skalowanie sygnału do zadanego poziomu RMS w dBFS (FS=1.0)."""
    rms = np.sqrt(np.mean(x*x) + eps)
    target_rms = 10 ** (target_dbfs / 20.0)
    return x * (target_rms / rms)

def generate_pink_noise(duration: float,
                            fs: int,
                            level_dbfs_rms: float = -12.0,
                            f_low: float = 20.0,
                            f_high: float | None = None,
                            seed: int | None = None) -> np.ndarray:

    if seed is not None:
        rng = np.random.default_rng(seed)
        white = rng.normal(0.0, 1.0, int(duration * fs))
    else:
        white = np.random.normal(0.0, 1.0, int(duration * fs))

    n = white.size
    if n < 4:
        raise ValueError("Za krótki sygnał.")

    # RFFT
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # Ograniczenia pasma
    if f_high is None:
        f_high = fs / 2.0

    # Filtr amplitudowy: 1/sqrt(f) -> PSD ~ 1/f
    f = np.maximum(freqs, 1e-9)           # zabezpieczenie
    H = 1.0 / np.sqrt(f)

    # Usuń DC (żeby nie było offsetu)
    H[0] = 0.0

    # Utnij pasmo poza [f_low, f_high] jeśli podane
    H[freqs < f_low] = 0.0
    H[freqs > f_high] = 0.0

    # Zastosuj filtr
    X *= H

    pink = np.fft.irfft(X, n=n)

    #usuń ewentualny bardzo mały offset numeryczny
    pink -= np.mean(pink)

    # Ustaw poziom RMS w dBFS
    pink = set_rms_dbfs(pink, level_dbfs_rms)

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

    def start(self, audio_cfg, level_db=-6):
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


