# convolution_engine.py

import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def _load_wav_mono_stereo(path: str) -> Tuple[np.ndarray, int]:
    """
    Wczytuje WAV i zwraca:
        data  – shape (N, C), C=1 lub 2, typ float32
        fs    – sample rate

    Jeśli plik ma >2 kanały → rzuca ValueError.
    """
    if not os.path.isfile(path):
        raise ValueError(f"Plik nie istnieje:\n{path}")

    data, fs = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] == 0:
        raise ValueError(f"Plik audio ma 0 kanałów:\n{path}")
    if data.shape[1] > 2:
        raise ValueError(
            f"Plik {os.path.basename(path)} ma {data.shape[1]} kanałów.\n"
            f"Aktualnie obsługuję tylko mono / stereo."
        )
    return data, int(fs)


def _downmix_to_mono(data: np.ndarray) -> np.ndarray:
    """
    Zamienia (N, C) na (N, 1) uśredniając kanały,
    jeśli C > 1. Jeśli C==1 zwraca bez zmian.
    """
    if data.ndim != 2:
        raise ValueError("Data musi mieć kształt (N, C).")
    if data.shape[1] == 1:
        return data
    mono = data.mean(axis=1, keepdims=True)
    return mono.astype(np.float32)


def _fft_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Konwolucja FFT dwóch sygnałów 1D (full).
    Zwraca sygnał o długości len(x) + len(h) - 1.
    """
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
    """
    Normalizuje IR mono (N,) lub (N,1) do max=1.
    """
    if ir.ndim == 2:
        ir = ir[:, 0]
    max_abs = float(np.max(np.abs(ir)) + 1e-12)
    return (ir / max_abs).astype(np.float32)


def _normalize_ir_stereo(ir_L: np.ndarray, ir_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wspólna normalizacja IR L/R: globalny max liczony
    po obu kanałach, aby zachować relację L/R.
    """
    if ir_L.ndim == 2:
        ir_L = ir_L[:, 0]
    if ir_R.ndim == 2:
        ir_R = ir_R[:, 0]

    stacked = np.stack([ir_L, ir_R], axis=1)
    max_abs = float(np.max(np.abs(stacked)) + 1e-12)
    ir_Ln = (ir_L / max_abs).astype(np.float32)
    ir_Rn = (ir_R / max_abs).astype(np.float32)
    return ir_Ln, ir_Rn


def convolve_audio_files(
    audio_path: str,
    mode: str,
    ir_mono_path: Optional[str] = None,
    ir_left_path: Optional[str] = None,
    ir_right_path: Optional[str] = None,
    wet: float = 1.0,
    output_path: Optional[str] = None,
) -> str:
    """
    Główny silnik splotu.

    Parametry
    ---------
    audio_path : str
        Ścieżka do pliku WAV z sygnałem wejściowym (mono lub stereo).
    mode : str
        "Mono" lub "Stereo" — wybór z GUI.
    ir_mono_path : str, optional
        Ścieżka do IR w trybie Mono.
    ir_left_path, ir_right_path : str, optional
        Ścieżki do IR Left / Right w trybie Stereo.
    wet : float
        Udział pogłosu 0..1 (0=tylko dry, 1=tylko wet).
    output_path : str, optional
        Ścieżka do pliku wynikowego; jeśli None, zostanie wygenerowana
        na podstawie audio_path.

    Zwraca
    ------
    str : ścieżka do zapisanego pliku wynikowego.
    """

    if mode not in ("Mono", "Stereo"):
        raise ValueError(f"Nieznany tryb splotu: {mode}")

    if wet < 0.0:
        wet = 0.0
    if wet > 1.0:
        wet = 1.0
    dry = 1.0 - wet

    # 1) Wczytanie audio
    audio, fs_audio = _load_wav_mono_stereo(audio_path)  # (N, C)
    n_samples, n_channels = audio.shape  # C=1 lub 2

    # 2) Wczytanie IR w zależności od trybu
    if mode == "Mono":
        if not ir_mono_path:
            raise ValueError("W trybie Mono musisz wybrać plik IR.")
        ir_mono, fs_ir = _load_wav_mono_stereo(ir_mono_path)
        if fs_ir != fs_audio:
            raise ValueError(
                f"Sample rate audio ({fs_audio} Hz) i IR ({fs_ir} Hz) są różne.\n"
                f"Na razie wymagane jest takie samo fs."
            )
        # downmix na wszelki wypadek (np. gdy ktoś poda IR stereo)
        ir_mono = _downmix_to_mono(ir_mono)[:, 0]
        ir_mono = _normalize_ir_mono(ir_mono)

    else:  # mode == "Stereo"
        if not ir_left_path or not ir_right_path:
            raise ValueError("W trybie Stereo musisz wybrać IR Left oraz IR Right.")

        ir_L, fs_L = _load_wav_mono_stereo(ir_left_path)
        ir_R, fs_R = _load_wav_mono_stereo(ir_right_path)

        if fs_L != fs_audio or fs_R != fs_audio or fs_L != fs_R:
            raise ValueError(
                "Sample rate audio i IR L/R są różne.\n"
                "Na razie wymagane jest takie samo fs dla wszystkich plików."
            )

        ir_L = _downmix_to_mono(ir_L)[:, 0]
        ir_R = _downmix_to_mono(ir_R)[:, 0]
        ir_L, ir_R = _normalize_ir_stereo(ir_L, ir_R)

    # 3) Konwolucja według scenariuszy

    # Ujednolicenie audio: (N,C)
    audio_L = audio[:, 0]
    audio_R = audio[:, 1] if n_channels == 2 else None

    if mode == "Mono":
        if n_channels == 1:
            # MONO tryb + audio mono → mono wynik
            y_L = _fft_convolve_1d(audio_L, ir_mono)
            out_channels = 1
        else:
            # MONO tryb + audio stereo → L, R z tym samym IR
            y_L = _fft_convolve_1d(audio_L, ir_mono)
            y_R = _fft_convolve_1d(audio_R, ir_mono)
            out_channels = 2

    else:  # mode == "Stereo"
        if n_channels == 1:
            # STEREO tryb + audio mono → mono ⊗ IR L, IR R
            y_L = _fft_convolve_1d(audio_L, ir_L)
            y_R = _fft_convolve_1d(audio_L, ir_R)
            out_channels = 2
        else:
            # STEREO tryb + audio stereo → L⊗IR L, R⊗IR R
            y_L = _fft_convolve_1d(audio_L, ir_L)
            y_R = _fft_convolve_1d(audio_R, ir_R)
            out_channels = 2

    # 4) Wet/Dry – dopasowanie długości
    def _mix_channel(dry_sig: np.ndarray, wet_sig: np.ndarray) -> np.ndarray:
        """
        dry_sig – oryginalny kanał (N,)
        wet_sig – wynik splotu (M,)
        Zwraca kanał (M,) – mieszanka wet/dry.
        """
        N = len(dry_sig)
        M = len(wet_sig)
        if M < N:
            # teoretycznie prawie się nie zdarzy, ale zabezpieczenie:
            dry_use = dry_sig[:M]
        else:
            pad = M - N
            dry_use = np.pad(dry_sig, (0, pad), mode="constant")
        return (dry * dry_use + wet * wet_sig).astype(np.float32)

    if out_channels == 1:
        mixed_L = _mix_channel(audio_L, y_L)
        out = mixed_L[:, np.newaxis]  # (M,1)
    else:
        mixed_L = _mix_channel(audio_L, y_L)
        mixed_R = _mix_channel(audio_R, y_R)
        out = np.stack([mixed_L, mixed_R], axis=1)  # (M,2)

    # 5) Ochrona przed clippingiem
    max_abs = float(np.max(np.abs(out)) + 1e-12)
    if max_abs > 1.0:
        out /= max_abs

    # 6) Ścieżka wyjściowa
    if output_path is None or output_path.strip() == "":
        base, _ = os.path.splitext(audio_path)
        suffix = "_conv_mono" if mode == "Mono" and out_channels == 1 else "_conv"
        output_path = base + suffix + ".wav"

    sf.write(output_path, out, fs_audio)
    return output_path
