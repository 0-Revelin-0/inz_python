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


def build_binaural_ir_from_mono_ir(
    ir_mono: np.ndarray,
    fs: int,
    mat_path: str,
    az_deg: float,
    el_deg: float,
    direct_tail_ms: float = 5.0,
    early_ms: float = 80.0,
    crossfade_ms: float = 10.0,
    early_spread_deg: float = 15.0,
    early_sources: int = 3,
    late_sources: int = 12,
    rng_seed: int | None = None,
) -> np.ndarray:
    """
    Finalna logika:
    - DIRECT: 0 .. (peak + direct_tail_ms) -> 1 HRIR z GUI
    - EARLY:  (peak + direct_tail_ms) .. (peak + early_ms) -> suma kilku HRIR blisko GUI (spread)
    - LATE:   (peak + early_ms) .. koniec -> suma wielu HRIR losowo po azymucie (wszechkierunkowo)
    Crossfade Hann na granicach, żeby nie było “dziur”.
    """

    if ir_mono.ndim != 1:
        ir_mono = np.asarray(ir_mono).reshape(-1)

    ir_mono = ir_mono.astype(np.float32, copy=False)
    N = len(ir_mono)
    if N < 4:
        raise ValueError("IR jest za krótka do binauralizacji.")

    rng = np.random.default_rng(rng_seed)

    # --- Peak (direct arrival) ---
    peak_idx = int(np.argmax(np.abs(ir_mono)))

    direct_end = peak_idx + int(round(float(direct_tail_ms) * fs / 1000.0))
    direct_end = int(np.clip(direct_end, 0, N))

    early_end = peak_idx + int(round(float(early_ms) * fs / 1000.0))
    early_end = int(np.clip(early_end, direct_end, N))

    # Crossfade (Hann) na granicach fragmentów
    xfade = int(round(float(crossfade_ms) * fs / 1000.0))
    xfade = max(0, xfade)
    half = xfade // 2  # pół po obu stronach granicy

    def _fade_in(L: int) -> np.ndarray:
        if L <= 0:
            return np.ones(0, dtype=np.float32)
        t = np.linspace(0.0, np.pi, L, endpoint=True, dtype=np.float32)
        return 0.5 * (1.0 - np.cos(t))  # 0 -> 1 (raised cosine)

    def _fade_out(L: int) -> np.ndarray:
        fi = _fade_in(L)
        return (1.0 - fi).astype(np.float32, copy=False)  # 1 -> 0

    direct = np.zeros(N, dtype=np.float32)
    early = np.zeros(N, dtype=np.float32)
    late = np.zeros(N, dtype=np.float32)

    # ---------- Granica 1: direct <-> early ----------
    s1 = max(0, direct_end - half)
    e1 = min(N, direct_end + half)
    L1 = max(0, e1 - s1)

    d0_end = max(0, s1)
    direct[:d0_end] += ir_mono[:d0_end]

    if L1 > 0:
        w_in = _fade_in(L1)
        w_out = _fade_out(L1)
        direct[s1:e1] += ir_mono[s1:e1] * w_out
        early[s1:e1] += ir_mono[s1:e1] * w_in

    # early środek (bez okna)
    e_mid_start = e1
    e_mid_end = max(e_mid_start, max(0, early_end - half))
    if e_mid_end > e_mid_start:
        early[e_mid_start:e_mid_end] += ir_mono[e_mid_start:e_mid_end]

    # ---------- Granica 2: early <-> late ----------
    s2 = max(0, early_end - half)
    e2 = min(N, early_end + half)
    L2 = max(0, e2 - s2)

    if L2 > 0:
        w_in2 = _fade_in(L2)
        w_out2 = _fade_out(L2)
        early[s2:e2] += ir_mono[s2:e2] * w_out2
        late[s2:e2] += ir_mono[s2:e2] * w_in2

    l0_start = e2
    if l0_start < N:
        late[l0_start:] += ir_mono[l0_start:]

    # ------------------------------------------------------------------
    # Kierunki (DIRECT / EARLY / LATE)
    # ------------------------------------------------------------------
    az_gui = float(az_deg)
    el_gui = float(el_deg)

    # EARLY: GUI + kilka losowych odchyleń (spread)
    early_sources = int(max(1, early_sources))
    early_dirs = [(az_gui, el_gui)]
    for _ in range(early_sources - 1):
        az_e = az_gui + float(rng.uniform(-early_spread_deg, early_spread_deg))
        el_e = el_gui
        early_dirs.append((az_e, el_e))

    # LATE: wiele losowych kierunków (wszechkierunkowo)
    late_sources = int(max(1, late_sources))
    late_dirs = []
    for _ in range(late_sources):
        az_l = float(rng.uniform(-180.0, 180.0))
        el_l = 0.0
        late_dirs.append((az_l, el_l))

    # ------------------------------------------------------------------
    # Suma równoległa HRIR z normalizacją energii (1/sqrt(K))
    # ------------------------------------------------------------------
    def _sum_hrtf(audio_mono_1d: np.ndarray, dirs: list[tuple[float, float]]) -> np.ndarray:
        K = len(dirs)
        if K == 1:
            az, el = dirs[0]
            return apply_hrtf_to_audio(
                audio=audio_mono_1d.reshape(-1, 1),
                fs_audio=fs,
                mat_path=mat_path,
                az_deg=az,
                el_deg=el,
                downmix_stereo_to_mono=True,
            )
        w = 1.0 / np.sqrt(float(K))
        acc = None
        for (az, el) in dirs:
            bi = apply_hrtf_to_audio(
                audio=audio_mono_1d.reshape(-1, 1),
                fs_audio=fs,
                mat_path=mat_path,
                az_deg=az,
                el_deg=el,
                downmix_stereo_to_mono=True,
            ).astype(np.float32, copy=False)
            acc = (w * bi) if acc is None else (acc + w * bi)
        return acc

    d_bi = _sum_hrtf(direct, [(az_gui, el_gui)])
    e_bi = _sum_hrtf(early, early_dirs)
    l_bi = _sum_hrtf(late, late_dirs)

    binaural_ir = (d_bi + e_bi + l_bi).astype(np.float32, copy=False)
    return binaural_ir

