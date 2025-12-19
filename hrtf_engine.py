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
    az = np.real(pos[:, 0]).astype(np.float64, copy=False)
    el = np.real(pos[:, 1]).astype(np.float64, copy=False)

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
    late_sources: int = 14,
    late_pairing: bool = True,
    late_time_jitter_ms: float = 6,
    rng_seed: int | None = None,
) -> np.ndarray:
    """
    FINALNA LOGIKA (zgodna z literaturą):

    DIRECT:
      - 1 HRIR, kierunek z GUI

    EARLY:
      - kilka HRIR (early_sources)
      - kierunki: GUI ± early_spread_deg
      - suma z normalizacją 1/sqrt(N)

    LATE:
      - wiele HRIR (late_sources)
      - kierunki w parach ±az (diffuse field)
      - mikro-jitter czasowy (late_time_jitter_ms)
      - suma z normalizacją 1/sqrt(N)

    Podział IR:
      direct → early → late
      z crossfade Hann-like na granicach
    """

    if ir_mono.ndim != 1:
        ir_mono = np.asarray(ir_mono).reshape(-1)

    ir_mono = ir_mono.astype(np.float32, copy=False)
    N = len(ir_mono)
    if N < 8:
        raise ValueError("IR jest za krótka do binauralizacji.")

    rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------
    # PODZIAŁ IR
    # ------------------------------------------------------------
    peak_idx = int(np.argmax(np.abs(ir_mono)))

    direct_end = int(np.clip(
        peak_idx + round(direct_tail_ms * fs / 1000.0),
        0, N
    ))

    early_end = int(np.clip(
        peak_idx + round(early_ms * fs / 1000.0),
        direct_end, N
    ))

    xfade = int(round(crossfade_ms * fs / 1000.0))
    half = max(1, xfade // 2)

    def fade_in(L):
        t = np.linspace(0, np.pi, L, dtype=np.float32)
        return 0.5 * (1 - np.cos(t))

    def fade_out(L):
        return 1.0 - fade_in(L)

    direct = np.zeros(N, dtype=np.float32)
    early = np.zeros(N, dtype=np.float32)
    late = np.zeros(N, dtype=np.float32)

    # direct → early
    s1 = max(0, direct_end - half)
    e1 = min(N, direct_end + half)
    direct[:s1] = ir_mono[:s1]
    direct[s1:e1] = ir_mono[s1:e1] * fade_out(e1 - s1)
    early[s1:e1] = ir_mono[s1:e1] * fade_in(e1 - s1)

    # early środek
    early[e1:early_end - half] = ir_mono[e1:early_end - half]

    # early → late
    s2 = max(0, early_end - half)
    e2 = min(N, early_end + half)
    early[s2:e2] += ir_mono[s2:e2] * fade_out(e2 - s2)
    late[s2:e2] = ir_mono[s2:e2] * fade_in(e2 - s2)
    late[e2:] = ir_mono[e2:]

    # ------------------------------------------------------------
    # FUNKCJE POMOCNICZE
    # ------------------------------------------------------------
    def shift_stereo(bi: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return bi
        N = bi.shape[0]
        if shift > 0:
            return np.pad(bi, ((shift, 0), (0, 0)))[:N]
        else:
            s = -shift
            return np.pad(bi, ((0, s), (0, 0)))[s:s+N]

    def sum_hrtf(audio: np.ndarray, dirs: list, jitter_ms: float = 0.0) -> np.ndarray:
        K = len(dirs)
        w = 1.0 / np.sqrt(K)
        jitter_samp = int(jitter_ms * fs / 1000.0)
        acc = None

        for az, el in dirs:
            bi = apply_hrtf_to_audio(
                audio=audio.reshape(-1, 1),
                fs_audio=fs,
                mat_path=mat_path,
                az_deg=az,
                el_deg=el,
                downmix_stereo_to_mono=True,
            ).astype(np.float32, copy=False)

            if jitter_samp > 0:
                sh = rng.integers(-jitter_samp, jitter_samp + 1)
                bi = shift_stereo(bi, sh)

            acc = w * bi if acc is None else acc + w * bi

        return acc

    # ------------------------------------------------------------
    # KIERUNKI
    # ------------------------------------------------------------
    az0, el0 = float(az_deg), float(el_deg)

    # EARLY
    early_dirs = [(az0, el0)]
    for _ in range(max(1, early_sources) - 1):
        early_dirs.append((
            az0 + rng.uniform(-early_spread_deg, early_spread_deg),
            el0
        ))

    # LATE – totalnie losowe kierunki z siatki HRIR, ale symetryczne (mirror)
    # Elewacje ograniczamy do [-30..90] (Twoje wymaganie), ale tylko jeśli takie są w bazie.

    mat_data = load_hrtf_database(mat_path)
    az_all = mat_data["az_deg"].astype(np.float64, copy=False)
    el_all = mat_data["el_deg"].astype(np.float64, copy=False)

    EL_MIN, EL_MAX = -30.0, 90.0
    valid_idx = np.where((el_all >= EL_MIN) & (el_all <= EL_MAX))[0]

    if valid_idx.size == 0:
        # awaryjnie: jeśli baza nie ma takich elewacji, bierzemy wszystko
        valid_idx = np.arange(len(az_all), dtype=int)

    # mapowanie (az, el) -> index (pozycje w tej bazie są zwykle dokładnymi wartościami siatki)
    pos_to_idx: dict[tuple[float, float], int] = {}
    for i in valid_idx:
        pos_to_idx[(float(az_all[i]), float(el_all[i]))] = int(i)

    def mirror_az_360(az_deg: float) -> float:
        az360 = _to_az_360(az_deg)
        return (360.0 - az360) % 360.0

    # lista par indeksów, które mają swoje lustrzane odbicie w bazie na tej samej elewacji
    pair_indices: list[tuple[int, int]] = []
    for (az, el), i in pos_to_idx.items():
        az_m = mirror_az_360(az)
        key_m = (float(az_m), float(el))
        j = pos_to_idx.get(key_m, None)
        if j is None:
            continue
        # żeby nie dublować par (i,j) i (j,i)
        if i <= j:
            pair_indices.append((i, j))

    late_dirs: list[tuple[float, float]] = []

    if late_pairing and len(pair_indices) > 0:
        pairs_needed = late_sources // 2
        replace = len(pair_indices) < pairs_needed

        chosen = rng.choice(len(pair_indices), size=pairs_needed, replace=replace)
        for k in chosen:
            i, j = pair_indices[int(k)]
            late_dirs.append((float(az_all[i]), float(el_all[i])))
            late_dirs.append((float(az_all[j]), float(el_all[j])))

        # jeśli nieparzysta liczba źródeł – dobierz jeszcze jeden losowy kierunek
        if late_sources % 2 == 1:
            i = int(rng.choice(valid_idx))
            late_dirs.append((float(az_all[i]), float(el_all[i])))

    else:
        # jeśli nie chcemy parowania albo nie da się sparować – losujemy po prostu z siatki
        chosen = rng.choice(valid_idx, size=late_sources, replace=(valid_idx.size < late_sources))
        for i in chosen:
            late_dirs.append((float(az_all[int(i)]), float(el_all[int(i)])))

    rng.shuffle(late_dirs)

    # ------------------------------------------------------------
    # BINAURALIZACJA
    # ------------------------------------------------------------
    d_bi = sum_hrtf(direct, [(az0, el0)], 0.0)
    e_bi = sum_hrtf(early, early_dirs, 0.0)
    l_bi = sum_hrtf(late, late_dirs, late_time_jitter_ms)

    binaural_ir = (d_bi + e_bi + l_bi).astype(np.float32, copy=False)
    return binaural_ir


