# ============================================================
#   convolution_engine.py – WERSJA POPRAWIONA
#   – RMS matching
#   – logarytmiczny slider wet/dry
#   – downmix, FFT convolution
# ============================================================

import os
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

from hrtf_engine import apply_hrtf_to_audio


# ============================== HELPERS ===============================

def _load_wav_mono_stereo(path: str) -> Tuple[np.ndarray, int]:
    """
    Wczytuje WAV → (N, C), gdzie C=1 lub 2. float32.
    """
    if not os.path.isfile(path):
        raise ValueError(f"Plik nie istnieje:\n{path}")

    data, fs = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] == 0:
        raise ValueError("Plik audio ma 0 kanałów.")
    if data.shape[1] > 2:
        raise ValueError(f"Plik {path} ma {data.shape[1]} kanałów, obsługuję tylko mono/stereo.")

    return data, int(fs)


def _downmix_to_mono(data: np.ndarray) -> np.ndarray:
    """Z (N, C) → (N,1) uśredniając kanały. Jeśli C==1 → bez zmian."""
    if data.shape[1] == 1:
        return data
    mono = data.mean(axis=1, keepdims=True)
    return mono.astype(np.float32)


def _fft_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Konwolucja FFT (full). Zwraca length = len(x)+len(h)-1."""
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
    """Normalizacja IR mono do max=1."""
    if ir.ndim == 2:
        ir = ir[:, 0]
    max_abs = float(np.max(np.abs(ir)) + 1e-12)
    return (ir / max_abs).astype(np.float32)


def _normalize_ir_stereo(ir_L: np.ndarray, ir_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wspólna normalizacja IR stereo."""
    if ir_L.ndim == 2:
        ir_L = ir_L[:, 0]
    if ir_R.ndim == 2:
        ir_R = ir_R[:, 0]

    stacked = np.stack([ir_L, ir_R], axis=1)
    max_abs = float(np.max(np.abs(stacked)) + 1e-12)

    return (ir_L / max_abs).astype(np.float32), (ir_R / max_abs).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    """Root Mean Square sygnału."""
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


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
) -> str:

    """
    Silnik splotu audio z RMS matching + logarytmicznym wet/dry.
    """

    # -------------------- WET/DRY (wejście) --------------------

    wet_frac = np.clip(float(wet), 0.0, 1.0)

    # logarytmiczna krzywa (bardziej naturalna)
    # 0 → bardzo subtelny
    # 1 → pełny pogłos
    wet_adj = wet_frac ** 1.5
    dry_adj = 1.0 - wet_adj

    # ---------------------- Wczytanie audio ----------------------

    audio, fs_audio = _load_wav_mono_stereo(audio_path)
    N, C = audio.shape

    audio_L = audio[:, 0]
    audio_R = audio[:, 1] if C == 2 else None


    # ---------------------- HRTF (PRE-CONV) ----------------------
    if use_hrtf:
        if mode != "Mono":
            raise ValueError("HRTF jest dostępne tylko w trybie Mono.")

        if not hrtf_db_path:
            raise ValueError("Nie wybrano pliku bazy HRTF (.mat) w Settings.")

        audio = apply_hrtf_to_audio(
            audio=audio,
            fs_audio=fs_audio,
            mat_path=hrtf_db_path,
            az_deg=float(hrtf_az_deg),
            el_deg=float(hrtf_el_deg),
            downmix_stereo_to_mono=True,  # Twoje: opcja 2
        )

        # odśwież wymiary po HRTF
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

        if C == 1:
            y_L = _fft_convolve_1d(audio_L, ir)
            out_channels = 1
        else:
            y_L = _fft_convolve_1d(audio_L, ir)
            y_R = _fft_convolve_1d(audio_R, ir)
            out_channels = 2

    if mode == "Stereo":
        audio_mono = audio_L if C == 1 else 0.5 * (audio_L + audio_R)

        y_L = _fft_convolve_1d(audio_mono, ir_L)

        y_R = _fft_convolve_1d(audio_mono, ir_R)

        out_channels = 2

    # -------------------- RMS MATCHING -----------------------
    # Wspólny gain dla stereo (żeby nie psuć balansu L/R).
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
