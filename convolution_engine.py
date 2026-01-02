import os
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

from hrtf_engine import build_binaural_ir_from_mono_ir
from hrtf_engine import apply_hrtf_to_audio


# ============================== HELPERS ===============================

def _load_wav_mono_stereo(path: str) -> Tuple[np.ndarray, int]:

    if not os.path.isfile(path):
        raise ValueError(f"Plik nie istnieje:\n{path}")

    data, fs = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] == 0:
        raise ValueError("Plik audio ma 0 kanałów.")
    if data.shape[1] > 2:
        raise ValueError(f"Plik {path} ma {data.shape[1]} kanałów, obsługuję tylko mono/stereo.")

    return data, int(fs)


def _downmix_to_mono(data: np.ndarray) -> np.ndarray:

    if data.shape[1] == 1:
        return data
    mono = data.mean(axis=1, keepdims=True)
    return mono.astype(np.float32)


def _fft_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)

    n_out = len(x) + len(h) - 1
    nfft = 1
    while nfft < n_out:
        nfft <<= 1

    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    y = np.fft.irfft(X * H, nfft)

    return y[:n_out].astype(np.float32)


def _normalize_ir_mono(ir: np.ndarray) -> np.ndarray:

    if ir.ndim == 2:
        ir = ir[:, 0]
    max_abs = float(np.max(np.abs(ir)) + 1e-12)
    return (ir / max_abs).astype(np.float32)


def _normalize_ir_stereo(ir_L: np.ndarray, ir_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    if ir_L.ndim == 2:
        ir_L = ir_L[:, 0]
    if ir_R.ndim == 2:
        ir_R = ir_R[:, 0]

    stacked = np.stack([ir_L, ir_R], axis=1)
    max_abs = float(np.max(np.abs(stacked)) + 1e-12)

    return (ir_L / max_abs).astype(np.float32), (ir_R / max_abs).astype(np.float32)


def _rms(x: np.ndarray) -> float:

    return float(np.sqrt(np.mean(x**2)) + 1e-12)

def normalize_to_dbfs(signal: np.ndarray, target_dbfs: float = -6.0) -> np.ndarray:

    target_peak = 10 ** (target_dbfs / 20.0)
    peak = float(np.max(np.abs(signal)))

    if peak > 0:
        gain = target_peak / peak
        signal = signal * gain

    return signal


# ============================ MAIN ENGINE ==============================

def convolve_audio_files(
    audio_path: str,
    mode: str,
    ir_mono_path: Optional[str] = None,
    ir_left_path: Optional[str] = None,
    ir_right_path: Optional[str] = None,
    wet: float = 1.0,
    output_path: Optional[str] = None,
    use_hrtf: bool = False,
    hrtf_db_path: Optional[str] = None,
    hrtf_az_deg: float = 0.0,
    hrtf_el_deg: float = 0.0,
    hrtf_direct_tail_ms: float = 5.0,
    hrtf_early_ms: float = 80.0,
    hrtf_crossfade_ms: float = 10.0,
    hrtf_early_spread_deg: float = 15.0,
) -> str:



    """
    Główna funkcja silnika
    Silnik splotu audio z RMS matching + logarytmicznym wet/dry.
    """

    # -------------------- WET/DRY (wejście) --------------------

    wet_frac = np.clip(float(wet), 0.0, 1.0)

    # logarytmiczna krzywa (bardziej naturalna)
    wet_adj = wet_frac ** 1.5
    dry_adj = 1.0 - wet_adj

    # ---------------------- Wczytanie audio ----------------------

    audio, fs_audio = _load_wav_mono_stereo(audio_path)
    N, C = audio.shape

    audio_L = audio[:, 0]
    audio_R = audio[:, 1] if C == 2 else None


    # ---------------------- IR ----------------------

    if mode == "Mono":

        if not ir_mono_path:
            raise ValueError("Brak pliku IR w trybie Mono.")

        ir, fs_ir = _load_wav_mono_stereo(ir_mono_path)
        if fs_ir != fs_audio:
            raise ValueError("Audio i IR mają różne sample rate.")

        ir = _downmix_to_mono(ir)[:, 0]
        ir = _normalize_ir_mono(ir)

        binaural_ir = None
        if use_hrtf:
            if not hrtf_db_path:
                raise ValueError("Nie wybrano pliku bazy HRTF (.mat) w Settings.")

            # Budujemy binauralną IR stereo z IR mono
            binaural_ir = build_binaural_ir_from_mono_ir(
                ir_mono=ir,
                fs=fs_audio,
                mat_path=hrtf_db_path,
                az_deg=float(hrtf_az_deg),
                el_deg=float(hrtf_el_deg),
                direct_tail_ms=float(hrtf_direct_tail_ms),
                early_ms=float(hrtf_early_ms),
                crossfade_ms=float(hrtf_crossfade_ms),
                early_spread_deg=float(hrtf_early_spread_deg),
                rng_seed=None,
            )


    else:  # Stereo

        if not ir_left_path or not ir_right_path:
            raise ValueError("Brak IR Left/Right w trybie Stereo.")

        ir_L, fs_L = _load_wav_mono_stereo(ir_left_path)
        ir_R, fs_R = _load_wav_mono_stereo(ir_right_path)

        if fs_L != fs_audio or fs_R != fs_audio:
            raise ValueError("Sample rate IR i audio muszą być takie same.")

        if ir_L.shape[1] != 1 or ir_R.shape[1] != 1:
            raise ValueError(
                "W trybie Stereo IR_L i IR_R muszą być plikami mono "
                "(np. binauralny pomiar L/R)."
            )

        ir_L, ir_R = _normalize_ir_stereo(ir_L, ir_R)

    # ---------------------- KONWOLUCJA ----------------------

    if mode == "Mono":

        if use_hrtf:
            # HRTF => zawsze robimy stereo binaural
            if binaural_ir is None:
                raise ValueError("Błąd: binaural_ir jest None mimo use_hrtf=True.")

            # wejście do binauralizacji powinno być mono

            audio_mono = audio_L if C == 1 else 0.5 * (audio_L + audio_R)

            # DRY = audio już po HRTF (bez IR)
            dry_binaural = apply_hrtf_to_audio(
                audio=audio_mono[:, None],
                fs_audio=fs_audio,
                mat_path=hrtf_db_path,
                az_deg=hrtf_az_deg,
                el_deg=hrtf_el_deg,
                downmix_stereo_to_mono=True,
            )

            audio_L = dry_binaural[:, 0]
            audio_R = dry_binaural[:, 1]

            # WET = audio mono * binauralna IR
            y_L = _fft_convolve_1d(audio_mono, binaural_ir[:, 0])
            y_R = _fft_convolve_1d(audio_mono, binaural_ir[:, 1])

            out_channels = 2

        else:
            # bez HRTF: dotychczasowa logika
            if C == 1:
                y_L = _fft_convolve_1d(audio_L, ir)
                out_channels = 1
            else:
                y_L = _fft_convolve_1d(audio_L, ir)
                y_R = _fft_convolve_1d(audio_R, ir)
                out_channels = 2

    if mode == "Stereo":
        # wejście do splotu stereo-IR robimy jako mono (typowe: źródło w środku)
        audio_mono = audio_L if C == 1 else 0.5 * (audio_L + audio_R)

        y_L = _fft_convolve_1d(audio_mono, ir_L)
        y_R = _fft_convolve_1d(audio_mono, ir_R)
        out_channels = 2

        # WAŻNE: dry też musi być stereo dla miksu wet/dry + RMS matching
        if C == 1:
            audio_L = audio_mono
            audio_R = audio_mono

    # -------------------- RMS MATCHING -----------------------
    # Wspólny gain dla stereo (żeby nie psuć balansu L/R)
    if out_channels == 1:
        gain = _rms(audio_L) / _rms(y_L)
        y_L *= gain
    else:
        dry_stack = np.stack([audio_L, audio_R], axis=1)
        wet_stack = np.stack([y_L, y_R], axis=1)
        gain = _rms(dry_stack) / _rms(wet_stack)
        y_L *= gain
        y_R *= gain

    # -------------------- MIKSOWANIE -----------------------

    def _mix(dry_sig: np.ndarray, wet_sig: np.ndarray):
        # dopasuj długość
        M = len(wet_sig)
        if len(dry_sig) < M:
            dry_sig = np.pad(dry_sig, (0, M - len(dry_sig)))
        else:
            dry_sig = dry_sig[:M]

        return dry_adj * dry_sig + wet_adj * wet_sig

    if out_channels == 1:
        out = _mix(audio_L, y_L)[:, None]
    else:
        outL = _mix(audio_L, y_L)
        outR = _mix(audio_R, y_R)
        out = np.stack([outL, outR], axis=1)

    # -------------------- NORMALIZACJA DO -6 dBFS -----------------------

    out = normalize_to_dbfs(out, target_dbfs=-6.0)

    # -------------------- Limiter -----------------------

    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak

    # -------------------- Zapis -----------------------

    if output_path is None:
        base, _ = os.path.splitext(audio_path)
        suffix = "_conv.wav"
        output_path = base + suffix

    sf.write(output_path, out.astype(np.float32), fs_audio)
    return output_path
