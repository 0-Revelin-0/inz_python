import numpy as np

SPEED_OF_SOUND = 343.0  # [m/s]
CENTER_FREQS = np.array(
    [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0], dtype=float)



def compute_room_and_t60(room_dims, alpha_walls, alpha_ceiling, alpha_floor):

    W, L, H = room_dims
    W = float(W)
    L = float(L)
    H = float(H)

    # Geometria
    V = W * L * H
    S_su = W * L                   # powierzchnia sufitu
    S_po = W * L                   # powierzchnia podłogi
    S_sc = 2.0 * (W * H + L * H)   # powierzchnia wszystkich ściany
    S = S_su + S_po + S_sc

    # Konwersja list współczynników pochłaniania na tablice
    alpha_walls = np.asarray(alpha_walls, dtype=float)
    alpha_ceiling = np.asarray(alpha_ceiling, dtype=float)
    alpha_floor = np.asarray(alpha_floor, dtype=float)

    # Zakładamy, że wszystkie trzy mają tę samą długość, czyli 8 pasm
    n_bands = len(alpha_walls)

    if not (len(alpha_ceiling) == n_bands == len(alpha_floor)):
        raise ValueError(
            f"Niezgodne długości wektorów alfa: "
            f"walls={len(alpha_walls)}, ceil={len(alpha_ceiling)}, floor={len(alpha_floor)}"
        )

    alpha_mean = np.zeros(n_bands, dtype=float) # tablica średnich współczynników pochłaniania dla każdego pasma
    t60_bands = np.zeros(n_bands, dtype=float)  # tablica czasów pogłosu T60 dla każdego pasma

    #Pochłananie powierzchni w pasmach i czyli alfa * powierzchnia
    for i in range(n_bands):
        A_su = alpha_ceiling[i] * S_su
        A_po = alpha_floor[i] * S_po
        A_sc = alpha_walls[i] * S_sc
        A = A_su + A_po + A_sc   # Całkowita powierzchnia pochłaniająca w paśmie i

        if A <= 1e-6 or S <= 0:
            alpha_mean[i] = 0.0
            t60_bands[i] = 3   # Domyślna wartość T60 = 3 s, gdy brak pochłaniania (ograniczenie przed nieskończonym T60)
        else:
            alpha_mean[i] = A / S       # Średni współczynnik pochłaniania w paśmie i (alpha_mean = A / S)
            t60_bands[i] = 0.161 * V / A  # Czas pogłosu T60 w paśmie i wg wzoru Sabine'a

    # # Mean free path (średnia droga swobodna dźwięku między odbiciami)
    if S <= 0:
        mfp = 1.0
    else:
        mfp = 4.0 * V / S # Średnia droga swobodna [m] (lambda = 4V/S dla pomieszczenia zamkniętego)

    geom = {
        "V": V,
        "S": S,
        "S_su": S_su,
        "S_po": S_po,
        "S_sc": S_sc,
    }

    return alpha_mean, t60_bands, mfp, geom



def _design_fir_bandpass(fs, f_center, num_taps=513):
    """
    Prosty filtr FIR pasmowo-przepustowy (1 oktawa wokół f_center),
    projektowany metodą okien (Hamming).
    """
    fs = float(fs)
    f_center = float(f_center)

    # Granice pasma 1 oktawa (±1/2 oktawy)
    f1 = f_center / np.sqrt(2.0)
    f2 = f_center * np.sqrt(2.0)

    #Zabezpieczenie przed za wysoką i za niską częstotliwością
    nyq = fs / 2.0
    f2 = min(f2, 0.99 * nyq)
    f1 = max(f1, 1.0)  # nie schodzimy do 0 Hz

    n = np.arange(num_taps) - (num_taps // 2)

    def h_lp(fc):
        x = 2.0 * fc / fs * n
        return 2.0 * fc / fs * np.sinc(x)  # Impuls h dla filtru dolnoprzepustowego o częstotliwości odcięcia fc (idealny sinc)

    h2 = h_lp(f2)   # współczynniki idealnego filtru dolnoprzepustowego o odcięciu f2
    h1 = h_lp(f1)   # współczynniki idealnego filtru dolnoprzepustowego o odcięciu f1

    h = h2 - h1     # odejmując dwa filtry dolnoprzepustowe, otrzymujemy filtr pasmowy (przepuszcza zakres [f1, f2])

    # Okno Hamminga
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * (np.arange(num_taps)) / (num_taps - 1))
    h *= window     # aplikacja okna do impulsu filtra (wygładzenie odpowiedzi, redukcja efektów sinc-a)

    # # Normalizacja energii (skalowanie sumy modułów impulsu do 1)
    norm = np.sum(np.abs(h))
    if norm > 0:
        h /= norm

    return h.astype(np.float32)


def generate_late_reverb(fs, duration_s, t60_bands, mfp):
    """
    Generuje późny pogłos:
      - biały szum
      - filtracja w 8 pasmach oktawowych
      - obwiednia 10^(-3 * t / T60) w każdym paśmie
      - fade-in na początku zależny od mean free path.
    """
    fs = int(fs)
    n_samples = int(fs * duration_s) + 1     # liczba próbek odpowiadająca czasowi trwania (zaokrąglamy w górę +1)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=True)  # oś czasu od 0 do duration_s (w sekundach)

    # Biały szum (Gaussowski)
    white = np.random.normal(0.0, 1.0, size=n_samples).astype(np.float32)

    # Bank filtrów
    num_taps = 513
    band_signals = []

    for f0 in CENTER_FREQS: #Dla każdego filtra w liście częstotliwości środkowych
        h = _design_fir_bandpass(fs, f0, num_taps=num_taps) # zaprojektuj filtr band dla tej oktawy
        band = np.convolve(white, h, mode="same")      # przefiltruj biały szum tym filtrem (szum ograniczony do pasma tej oktawy)
        band_signals.append(band)

    band_signals = np.stack(band_signals, axis=0)  # łączy listę sygnałów w macierz [8 x n_samples]

    # Obwiednie w pasmach
    t60_bands = np.asarray(t60_bands, dtype=float)
    t60_bands = np.maximum(t60_bands, 0.05)  # ograniczenie, że minimalny T60 = 0.05 s dla każdego pasma

    # Sprawdzenie zgodności liczby pasm T60 z bankiem filtrów
    if t60_bands.shape[0] != CENTER_FREQS.shape[0]:
        raise ValueError(
            f"t60_bands ma {t60_bands.shape[0]} pasm, a CENTER_FREQS {CENTER_FREQS.shape[0]}.\n"
            "W GUI nie ma tyle samo pasm pochłaniania, co w CENTER_FREQS."
        )

    # env to macierz, gdzie każdy wiersz to obwiednia eksp. 10^(-3 t / T60_i) dla pasma i
    env = 10.0 ** (-3.0 * t[np.newaxis, :] / t60_bands[:, np.newaxis])

    # Zastosowanie obwiedni i suma pasm
    shaped = band_signals * env
    nL = np.sum(shaped, axis=0)

    # Fade-in pogłosu
    t_fade = 3.0 * mfp / SPEED_OF_SOUND # czas trwania fade-in [s], proporcjonalny do podwójnej średniej drogi swobodnej
    fade_len = int(t_fade * fs) # liczba próbek fade-in

    if fade_len > 1 and fade_len < n_samples:
        fade = np.linspace(0.0, 1.0, fade_len)
        mask = np.ones_like(nL)
        mask[:fade_len] = fade
        nL *= mask  # zastosowanie maski: początkowe próbki są skalowane od 0 do 1 (reszta *1)

    return nL.astype(np.float32)


def generate_early_reflections(fs, duration_s,
                               alpha_for_fdn,
                               mfp,
                               reflections_no,
                               rays_no,
                               first_refl_percent,
                               mfp_dev_percent,
                               rng=None):
    """
    Implementacja FDN (wczesnych odbić)

    """
    fs = int(fs)
    n_samples = int(fs * duration_s) + 1

    if rng is None:
        rng = np.random.default_rng()

    # Parametry odległości w metrach
    first_refl_dist = 0.5 * mfp  # dystans pierwszego odbicia
    first_refl_dev = (first_refl_percent / 100.0) * mfp # odchylenie std pierwszego odbicia (procent mfp)
    mean_free_path = mfp
    mean_free_path_dev = (mfp_dev_percent / 100.0) * mfp # odchylenie std kolejnych odcinków drogi (procent mfp)

    # Zamiana odległości [m] na liczbę próbek
    def dist_to_samples(dist_m):
        return int(np.floor((dist_m / SPEED_OF_SOUND) * fs))

    # Powyżej wyliczamy odpowiadające parametry w próbkach
    first_refl_samples = max(1, dist_to_samples(first_refl_dist))
    first_refl_dev_samples = max(0, dist_to_samples(first_refl_dev))
    mfp_samples = max(1, dist_to_samples(mean_free_path))
    mfp_dev_samples = max(0, dist_to_samples(mean_free_path_dev))

    nE = np.zeros(n_samples, dtype=np.float32)

    alpha = float(alpha_for_fdn)
    alpha_dev = alpha / 2.0 # Odchylenie standardowe dla losowej absorpcji = połowa wartości średniej

    for _ in range(int(rays_no)):
        a = float(rng.normal(alpha, alpha_dev)) # losowy współczynnik absorpcji dla tego promienia (normalnie wokół alpha)
        a = float(np.clip(a, 0.0, 0.99))    # ograniczenie od 0 do 0,99

        # losowe opóźnienie pierwszego odbicia
        r_first = int(np.floor(rng.normal(first_refl_samples, first_refl_dev_samples)))
        if r_first < 1:
            r_first = 1

        k = np.zeros(int(reflections_no), dtype=np.float32)
        M = np.zeros(int(reflections_no), dtype=np.int64)

        M[0] = r_first
        # losowy znak dla pierwszego odbicia (+1 lub -1)
        sign0 = -1.0 if rng.random() < 0.5 else 1.0
        k[0] = sign0 * (1.0 - a)

        for i in range(1, int(reflections_no)):
            r = int(np.floor(rng.normal(mfp_samples, mfp_dev_samples))) # długość kolejnej drogi w próbkach, losowana wokół mfp
            if r < 1:
                r = 1
            # losowy współczynnik pochłaniania przy tym odbiciu
            a = float(rng.normal(alpha, alpha_dev))
            a = float(np.clip(a, 0.0, 0.99)) # ograniczenie od 0 do 0,99

            # losowy znak dla każdego kolejnego odbicia
            signi = -1.0 if rng.random() < 0.5 else 1.0
            k[i] = signi * k[i - 1] * (1.0 - a) # amplituda kolejnego odbicia = poprzednia amplituda * współczynnik odbicia
            M[i] = M[i - 1] + r  # czas w próbkach kolejnego odbicia, czyli poprzedni czas + czas kolejnej droga

        # Dodanie wygenerowanej linii odbić do sumarycznego sygnału
        valid = (M >= 0) & (M < n_samples)   # maska dla indeksów, które mieszczą się w długości sygnału
        nE[M[valid]] += k[valid]    # sumujemy amplitudy odbić na tych pozycjach (zamiast nadpisywać)

    return nE


def generate_synthetic_ir_from_config(config, early_fraction, ir_duration_s, rng=None):
    """
    Główna funkcja silnika

    Parameters

    config : dict
          - fs                  : int
          - room_dims (W, L, H) : float
          - alpha_walls         : list(8)
          - alpha_ceiling       : list(8)
          - alpha_floor         : list(8)
          - rays_no             : int
          - reflections_no      : int
          - first_dev_percent   : float
          - mfp_dev_percent     : float
          - early_fraction      : float
          - ir_duration_s       : float
    """
    if rng is None:
        rng = np.random.default_rng()

    fs = int(config["fs"])
    room_dims = config["room_dims"]
    alpha_walls = np.asarray(config["alpha_walls"], dtype=float)
    alpha_ceiling = np.asarray(config["alpha_ceiling"], dtype=float)
    alpha_floor = np.asarray(config["alpha_floor"], dtype=float)

    rays_no = int(config.get("rays_no", 8))
    reflections_no = int(config.get("reflections_no", 20))
    first_dev_percent = float(config.get("first_dev_percent", 20.0))
    mfp_dev_percent = float(config.get("mfp_dev_percent", 50.0))

    # 1) Geometria, T60, mean free path
    alpha_mean, t60_bands, mfp, geom = compute_room_and_t60(
        room_dims,
        alpha_walls,
        alpha_ceiling,
        alpha_floor,
    )

    # Do FDN używamy alfa z pasma 1 kHz (indeks 3)
    alpha_for_fdn = float(alpha_mean[3]) if alpha_mean.size >= 4 else float(alpha_mean.mean())

    # Późny pogłos
    nL = generate_late_reverb(fs, ir_duration_s, t60_bands, mfp)

    # Wczesne odbicia
    nE = generate_early_reflections(
        fs=fs,
        duration_s=ir_duration_s,
        alpha_for_fdn=alpha_for_fdn,
        mfp=mfp,
        reflections_no=reflections_no,
        rays_no=rays_no,
        first_refl_percent=first_dev_percent,
        mfp_dev_percent=mfp_dev_percent,
        rng=rng,
    )

    # Dźwięk bezpośredni
    n_samples = len(nL)
    nD = np.zeros(n_samples, dtype=np.float32)
    nD[0] = 1.0

    # Mieszanie Early / Late
    p = float(np.clip(early_fraction, 0.0, 1.0))
    ir = nD + p * nE + (1.0 - p) * nL

    # # 6) Normalizacja globalna
    # max_abs = float(np.max(np.abs(ir)))
    # if max_abs > 0:
    #     ir = ir / max_abs

    return ir.astype(np.float32), fs
