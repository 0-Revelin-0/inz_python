# hrtf_engine.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve


def _to_az_360(az_deg: float) -> float:
    """Mapuje azymut z [-180..180] albo dowolnego zakresu na [0..360)."""
    a = float(az_deg) % 360.0
    if a < 0:
        a += 360.0
    return a


def _circular_abs_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Kołowa różnica kątowa w stopniach dla azymutu."""
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


@lru_cache(maxsize=8)
def load_hrtf_database(mat_path: str) -> dict:
    """
    Ładuje bazę HRTF .mat (pojedyncza osoba) i zwraca słownik z:
    - hM: (hrir_len, n_pos, 2)
    - az_deg: (n_pos,)
    - el_deg: (n_pos,)
    - fs: int
    """
    if not mat_path or not os.path.isfile(mat_path):
        raise ValueError(f"Nie znaleziono pliku bazy HRTF:\n{mat_path}")

    mat = loadmat(mat_path)

    if "hM" not in mat or "meta" not in mat or "stimPar" not in mat:
        raise ValueError(
            "Plik .mat nie ma oczekiwanej struktury (wymagane: hM, meta, stimPar)."
        )

    hM = mat["hM"]
    if not (isinstance(hM, np.ndarray) and hM.ndim == 3 and hM.shape[2] == 2):
        raise ValueError("hM ma nieoczekiwany kształt. Oczekiwano (len, n, 2).")

    meta = mat["meta"]
    stimPar = mat["stimPar"]

    # meta jest MATLAB struct (1,1)
    try:
        meta0 = meta[0, 0]
        pos = meta0["pos"]  # (n_pos, 7)
    except Exception:
        raise ValueError("Nie udało się odczytać meta.pos z pliku HRTF .mat.")

    if not (isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[0] == hM.shape[1]):
        raise ValueError("meta.pos ma niezgodny rozmiar z hM (liczba pozycji).")

    # Przyjęcie: pos[:,0] = azymut 0..360, pos[:,1] = elewacja
    az = pos[:, 0].astype(np.float64, copy=False)
    el = pos[:, 1].astype(np.float64, copy=False)

    # stimPar.SamplingRate
    try:
        sp0 = stimPar[0, 0]
        fs = int(sp0["SamplingRate"][0, 0])
    except Exception:
        raise ValueError("Nie udało się odczytać stimPar.SamplingRate z pliku HRTF .mat.")

    return {
        "hM": hM.astype(np.float32, copy=False),
        "az_deg": az,
        "el_deg": el,
        "fs": fs,
    }


def find_nearest_hrir_index(mat_data: dict, az_deg: float, el_deg: float) -> int:
    """
    Zwraca indeks najbliższej dostępnej pozycji (az, el).
    Azymut traktujemy kołowo (0..360), elewację liniowo.
    """
    az360 = _to_az_360(az_deg)

    az_all = mat_data["az_deg"]
    el_all = mat_data["el_deg"]

    daz = _circular_abs_diff_deg(az_all, az360)
    delv = np.abs(el_all - float(el_deg))

    # Proste kryterium: suma różnic w stopniach.
    # (Możemy to później ulepszyć wagami, jeśli zechcesz.)
    score = daz + delv
    return int(np.argmin(score))


def apply_hrtf_to_audio(
    audio: np.ndarray,
    fs_audio: int,
    mat_path: str,
    az_deg: float,
    el_deg: float,
    downmix_stereo_to_mono: bool = True,
) -> np.ndarray:
    """
    Nakłada HRTF na audio:
    - jeśli audio jest stereo: downmix do mono (średnia kanałów) -> Twoja "Opcja 2"
    - konwolucja mono z HRIR L i HRIR R
    - zwraca stereo (N', 2)
    NIC nie normalizujemy.
    """
    if audio is None or not isinstance(audio, np.ndarray):
        raise ValueError("apply_hrtf_to_audio: audio ma zły typ.")

    if audio.ndim != 2 or audio.shape[1] not in (1, 2):
        raise ValueError("apply_hrtf_to_audio: audio musi mieć kształt (N,1) lub (N,2).")

    mat_data = load_hrtf_database(mat_path)

    fs_hrtf = int(mat_data["fs"])
    if int(fs_audio) != fs_hrtf:
        raise ValueError(
            f"Niezgodny samplerate: audio={fs_audio} Hz, HRTF={fs_hrtf} Hz. "
            "Wymagamy zgodnego samplerate."
        )

    x = audio.astype(np.float32, copy=False)

    if x.shape[1] == 2:
        if downmix_stereo_to_mono:
            x_mono = 0.5 * (x[:, 0] + x[:, 1])
        else:
            # gdybyś kiedyś chciał alternatywę: np. tylko L
            x_mono = x[:, 0]
    else:
        x_mono = x[:, 0]

    # Korekta konwencji osi:
    # GUI: +az = prawo
    # Baza HRTF: +az = lewo (CCW)
    az_deg = -az_deg

    idx = find_nearest_hrir_index(mat_data, az_deg=az_deg, el_deg=el_deg)

    hM = mat_data["hM"]  # (hrir_len, n_pos, 2)
    hrir_L = hM[:, idx, 0].astype(np.float32, copy=False)
    hrir_R = hM[:, idx, 1].astype(np.float32, copy=False)

    # FFT convolution (szybko i stabilnie)
    yL = fftconvolve(x_mono, hrir_L, mode="full").astype(np.float32, copy=False)
    yR = fftconvolve(x_mono, hrir_R, mode="full").astype(np.float32, copy=False)

    out = np.column_stack([yL, yR]).astype(np.float32, copy=False)
    return out
