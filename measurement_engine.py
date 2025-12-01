# measurement_engine.py

import numpy as np
import sounddevice as sd


def generate_exponential_sweep(fs, duration, f_start, f_end):
    """
    Eksponencjalny sine sweep wg Fariny.
    fs        - częstotliwość próbkowania [Hz]
    duration  - długość sweepa [s]
    f_start   - częstotliwość startowa [Hz]
    f_end     - częstotliwość końcowa [Hz]
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # klasyczne wzory na ESS
    K = duration * 2.0 * np.pi * f_start / np.log(f_end / f_start)
    L = np.log(f_end / f_start) / duration

    sweep = np.sin(K * (np.exp(L * t) - 1.0))

    # normalizacja, żeby nie przesterować wyjścia
    sweep /= np.max(np.abs(sweep) + 1e-12)
    return sweep.astype(np.float32)


def generate_inverse_filter(sweep, fs, f_start, f_end):
    """
    Inverse filter do ESS:
    - odwrócenie w czasie
    - korekcja amplitudy malejąca eksponencjalnie
      (tak jak w klasycznej metodzie Fariny).
    """
    n = len(sweep)
    duration = n / fs
    t = np.linspace(0, duration, n, endpoint=False)

    L = np.log(f_end / f_start) / duration

    # odwrócony sweep + ważenie
    sweep_rev = sweep[::-1]
    w = np.exp(-L * t)  # kompensacja gęstości energii
    inv = sweep_rev * w

    inv /= np.max(np.abs(inv) + 1e-12)
    return inv.astype(np.float32)


def playrec_sweep(sweep, fs, audio_cfg, extra_silence=1.0):
    """
    Odtwarza sweep na wyjściu i nagrywa na wejściu.
    audio_cfg: dict z kluczami:
        - input_device
        - output_device
        - sample_rate
        - buffer_size
    extra_silence: dodatkowy czas ciszy po sweepie [s],
                   żeby odpowiedź pomieszczenia zdążyła wybrzmieć.
    """
    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([sweep, silence])

    rec = sd.playrec(
        play_sig,
        samplerate=fs,
        channels=1,
        device=(audio_cfg["output_device"], audio_cfg["input_device"]),
        blocksize=audio_cfg["buffer_size"],
        dtype="float32",
    )
    sd.wait()

    return rec[:, 0].copy()  # mono


def deconvolve_ir(recorded, inverse_filter, fs, ir_length_s, fade_time_s):
    """
    Dekonwolucja (ESS * inverse filter) → IR.
    Przycinamy do ir_length_s i robimy fade na końcu.
    """
    n_conv = len(recorded) + len(inverse_filter) - 1

    # najbliższa potęga 2 do FFT
    nfft = 1
    while nfft < n_conv:
        nfft *= 2

    R = np.fft.rfft(recorded, nfft)
    I = np.fft.rfft(inverse_filter, nfft)
    ir_full = np.fft.irfft(R * I, nfft)

    ir_samples = int(ir_length_s * fs)
    ir = ir_full[:ir_samples]

    # fade na końcu IR
    fade_samples = int(fade_time_s * fs)
    if fade_samples > 0 and fade_samples < ir_samples:
        window = np.ones(ir_samples, dtype=np.float32)
        tail = np.linspace(1.0, 0.0, fade_samples, endpoint=True)
        window[-fade_samples:] = tail
        ir *= window

    # normalizacja, żeby nie wyskoczyć z zakresu przy zapisie
    max_val = np.max(np.abs(ir) + 1e-12)
    ir = ir / max_val

    return ir.astype(np.float32)


def compute_mag_response(ir, fs):
    """
    Liczy charakterystykę amplitudową [dB] z IR.
    Zwraca:
        freqs [Hz], mag_db [dB]
    """
    nfft = 1
    n = len(ir)
    while nfft < n:
        nfft *= 2

    F = np.fft.rfft(ir, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(F), 1e-12))

    return freqs, mag_db


def measure_ir(params, audio_cfg):
    """
    Główna funkcja pomiaru IR.

    params: dict
        {
          "sweep_length": float [s],
          "start_freq":   float [Hz],
          "end_freq":     float [Hz],
          "ir_length":    float [s],
          "fade_time":    float [s]
        }

    audio_cfg: dict
        {
          "input_device": int,
          "output_device": int,
          "sample_rate": int,
          "buffer_size": int
        }

    Zwraca:
        ir        - numpy array (float32)
        freqs     - numpy array [Hz]
        mag_db    - numpy array [dB]
    """
    fs = int(audio_cfg["sample_rate"])

    sweep = generate_exponential_sweep(
        fs,
        params["sweep_length"],
        params["start_freq"],
        params["end_freq"],
    )
    inv = generate_inverse_filter(
        sweep,
        fs,
        params["start_freq"],
        params["end_freq"],
    )

    recorded = playrec_sweep(
        sweep,
        fs,
        audio_cfg,
        extra_silence=params["ir_length"]
    )

    ir = deconvolve_ir(
        recorded,
        inv,
        fs,
        params["ir_length"],
        params["fade_time"],
    )

    freqs, mag_db = compute_mag_response(ir, fs)

    return ir, freqs, mag_db
