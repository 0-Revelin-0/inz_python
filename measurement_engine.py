# measurement_engine.py
import queue
import numpy as np
import sounddevice as sd



def generate_exponential_sweep(fs, duration, f_start, f_end):
    if f_start <= 0:
        raise ValueError("f_start musi być > 0 Hz")

    R = f_end / f_start
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    K = duration * 2 * np.pi * f_start / np.log(R)
    L = np.log(R) / duration

    sweep = np.sin(K * (np.exp(L * t) - 1.0))

    # --- Normalizacja (0 dBFS) ---
    sweep /= (np.max(np.abs(sweep)) + 1e-12)

    # --- Obniżenie poziomu do -3 dbfs ---
    sweep *= 10 ** (-3 / 20)   # ≈ 0.70794578

    # --- Szukanie ostatniego przejścia przez zero ---
    # patrzymy na znak kolejnych próbek
    signs = np.sign(sweep)
    zero_cross = None

    for i in range(len(signs) - 2, 0, -1):
        if signs[i] == 0:
            zero_cross = i
            break
        if signs[i] != signs[i+1]:
            zero_cross = i+1
            break

    if zero_cross is None:
        zero_cross = len(sweep) - 1

    # --- Ucięcie sweepa w miejscu zero-cross ---
    sweep = sweep[:zero_cross]
    # Dodanie próbki 0, aby zakończyć sygnał perfekcyjnie
    sweep = np.concatenate([sweep, np.zeros(1, dtype=np.float32)])

    # --- Wymuszenie pierwszej próbki 0 — zgodnie z wymaganiami metody Fariny ---
    sweep[0] = 0.0

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


def _play_record(play_sig, fs, audio_cfg):
    play_sig = np.asarray(play_sig, dtype=np.float32)
    channels_in = int(audio_cfg["input_channels"])
    in_dev = audio_cfg["input_device"]
    out_dev = audio_cfg["output_device"]
    blocksize = audio_cfg["buffer_size"]

    q = queue.Queue()

    def input_callback(indata, frames, time, status):
        if status:
            print("Input status:", status)
        q.put(indata.copy())

    play_pos = 0

    def output_callback(outdata, frames, time, status):
        nonlocal play_pos
        if status:
            print("Output status:", status)

        end = min(play_pos + frames, len(play_sig))
        chunk = play_sig[play_pos:end]

        outdata[:, 0] = 0.0
        outdata[:len(chunk), 0] = chunk
        play_pos = end

    in_stream = sd.InputStream(
        samplerate=fs,
        channels=channels_in,
        device=in_dev,
        blocksize=blocksize,
        dtype="float32",
        callback=input_callback
    )

    out_stream = sd.OutputStream(
        samplerate=fs,
        channels=1,
        device=out_dev,
        blocksize=blocksize,
        dtype="float32",
        callback=output_callback
    )

    in_stream.start()
    out_stream.start()

    while play_pos < len(play_sig):
        sd.sleep(10)

    out_stream.stop(); out_stream.close()
    in_stream.stop(); in_stream.close()

    # Zebranie nagrania
    blocks = []
    while not q.empty():
        blocks.append(q.get())

    recorded = np.concatenate(blocks, axis=0)   # mono: (N,1) stereo:(N,2)
    if channels_in == 1:
        return recorded[:,0]
    return recorded



def playrec_sweep(sweep, fs, audio_cfg, extra_silence=1.0):
    """
    Klasyczny pomiar – pojedynczy sweep + cisza na ogon IR.
    Zachowanie jak dotychczas.
    """
    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([sweep, silence]).astype(np.float32)
    return _play_record(play_sig, fs, audio_cfg)


def playrec_sweeps_concat(sweep, fs, audio_cfg, repeats, extra_silence=1.0):
    """
    Odtwarza kilka sweepów sklejonych PRÓBKA DO PRÓBKI oraz końcową ciszę.

    repeats – liczba sweepów w szeregu (np. averages + 1)
    extra_silence – cisza po OSTATNIM sweepie (żeby złapać ogon IR ostatniego).
    """
    sweep = np.asarray(sweep, dtype=np.float32)
    concat = np.tile(sweep, repeats).astype(np.float32)

    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([concat, silence]).astype(np.float32)

    return _play_record(play_sig, fs, audio_cfg)






def deconvolve_ir(recorded, inverse_filter, fs, ir_length_s, fade_time_s):
    """
    Dekonwolucja (ESS * inverse filter) → IR.
    Przycinamy do ir_length_s i robimy fade na końcu.

    Uwaga: TA FUNKCJA NIE NORMALIZUJE już IR.
    Normalizacja jest wykonywana na wyższym poziomie (w measure_ir),
    aby dla stereo móc zastosować wspólny współczynnik dla L/R.
    """
    recorded = np.asarray(recorded, dtype=np.float32)
    inverse_filter = np.asarray(inverse_filter, dtype=np.float32)

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

    # 1. Przesunięcie impulsu do t=0 (direct sound)
    peak = np.argmax(np.abs(ir))
    ir = np.roll(ir, -peak)

    return ir.astype(np.float32)


def trim_ir(ir, fs, pre_ms=50, post_ms=300, tail_drop_db=60):
    """
    Automatyczne przycinanie IR:
      - pre_ms: ile ms zostawić PRZED głównym pikiem
      - post_ms: ile ms zostawić PO końcu ogona (wykrytego po spadku tail_drop_db)
    """

    ir = np.asarray(ir)
    n = len(ir)

    # 1) znajdź pik (direct sound)
    peak_idx = np.argmax(np.abs(ir))

    # przelicz ms na próbki
    pre_samp = int((pre_ms / 1000) * fs)
    post_samp = int((post_ms / 1000) * fs)

    # 2) początek IR = 50 ms przed pikiem
    start = max(0, peak_idx - pre_samp)

    # 3) znajdź punkt w którym IR spadło np. o 60 dB
    peak_val = np.max(np.abs(ir))
    threshold = peak_val * 10**(-tail_drop_db / 20)

    # idziemy od końca aż IR > próg
    end_idx = n - 1
    for i in range(n - 1, peak_idx, -1):
        if abs(ir[i]) > threshold:
            end_idx = i
            break

    # 4) koniec IR = 300 ms po końcu ogona
    end = min(n, end_idx + post_samp)

    return ir[start:end]



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


# def smooth_mag_response(freqs, mag_db, fraction=6):
#     """
#     Fractional-octave smoothing — domyślnie 1/6 oktawy.
#     """
#     smoothed = np.zeros_like(mag_db)
#     for i, f in enumerate(freqs):
#         if f <= 0:
#             smoothed[i] = mag_db[i]
#             continue
#
#         # szerokość okna w oktawach
#         f_low = f / (2 ** (1/(2*fraction)))
#         f_high = f * (2 ** (1/(2*fraction)))
#
#         idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
#         smoothed[i] = np.mean(mag_db[idx])
#
#
#     return smoothed


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
          "input_device":    int,
          "output_device":   int,
          "sample_rate":     int,
          "buffer_size":     int,
          "input_channels":  int (1 = mono, 2 = stereo)  [opcjonalne, domyślnie 1]
        }

    Zwraca:
        ir        - numpy array:
                    mono:  (N,)
                    stereo:(N, 2)
        freqs     - numpy array [Hz]
        mag_db    - numpy array [dB]:
                    mono:  (F,)
                    stereo:(F, 2)
        recorded  - surowe nagranie z wejścia:
                    mono:  (N,)
                    stereo:(N, 2)
    """
    fs = int(audio_cfg["sample_rate"])
    channels_in = int(audio_cfg.get("input_channels", 1))

    # 1) Sweep concat-safe wg Fariny (0 na początku i końcu)
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

    # 2) Liczba uśrednień (synchroniczne uśrednianie z okien)
    avg_count = int(params.get("averages", 1) or 1)
    if avg_count < 1:
        avg_count = 1

    if avg_count == 1:
        # Klasyczny pomiar – pojedynczy sweep + cisza
        recorded = playrec_sweep(
            sweep,
            fs,
            audio_cfg,
            extra_silence=params["ir_length"]
        )
    else:
        # Concatenated sweeps: repeats = avg_count + 1 (pierwsze okno wyrzucamy)
        repeats = avg_count + 1

        recorded_full = playrec_sweeps_concat(
            sweep,
            fs,
            audio_cfg,
            repeats=repeats,
            extra_silence=params["ir_length"]
        )
        recorded_full = np.asarray(recorded_full, dtype=np.float32)

        n_sweep = len(sweep)
        total_needed = repeats * n_sweep

        if recorded_full.ndim == 1:
            # MONO
            if len(recorded_full) < total_needed:
                raise RuntimeError("Nagranie jest krótsze niż oczekiwana liczba sweepów (mono).")

            rec_trim = recorded_full[:total_needed]
            windows = rec_trim.reshape(repeats, n_sweep)  # (repeats, N)
            useful = windows[1:, :]  # wyrzucamy pierwsze okno
            recorded = useful.mean(axis=0).astype(np.float32)  # (N,)
        else:
            # STEREO / wielokanałowe
            if recorded_full.shape[0] < total_needed:
                raise RuntimeError("Nagranie jest krótsze niż oczekiwana liczba sweepów (stereo).")

            n_ch = recorded_full.shape[1]
            rec_trim = recorded_full[:total_needed, :]  # (repeats*N, n_ch)
            windows = rec_trim.reshape(repeats, n_sweep, n_ch)  # (repeats, N, n_ch)
            useful = windows[1:, :, :]  # wyrzucamy pierwsze okno
            recorded = useful.mean(axis=0).astype(np.float32)  # (N, n_ch)

    # --- MONO ---
    if recorded.ndim == 1 or channels_in == 1:
        if recorded.ndim > 1:
            recorded_mono = recorded[:, 0]
        else:
            recorded_mono = recorded

        ir = deconvolve_ir(
            recorded_mono,
            inv,
            fs,
            params["ir_length"],
            params["fade_time"],
        )

        # normalizacja pojedynczego kanału
        max_val = np.max(np.abs(ir)) + 1e-12
        ir = (ir / max_val).astype(np.float32)

        freqs, mag_db = compute_mag_response(ir, fs)
        return ir, freqs, mag_db, recorded_mono

    # --- STEREO (lub więcej kanałów) ---
    if recorded.ndim == 1:
        # awaryjnie traktujemy to jako 1 kanał, choć konfiguracja mówi co innego
        recorded = recorded.reshape(-1, 1)

    n_ch = recorded.shape[1]

    ir_list = []
    for ch in range(n_ch):
        ir_ch = deconvolve_ir(
            recorded[:, ch],
            inv,
            fs,
            params["ir_length"],
            params["fade_time"],
        )
        ir_list.append(ir_ch)

    # zakładamy tę samą długość wszystkich kanałów
    min_len = min(len(x) for x in ir_list)
    ir_array = np.stack([x[:min_len] for x in ir_list], axis=1)  # (N, n_ch)

    # WSPÓLNA NORMALIZACJA DLA WSZYSTKICH KANAŁÓW (L/R itd.)
    max_val = np.max(np.abs(ir_array)) + 1e-12
    ir_array = (ir_array / max_val).astype(np.float32)

    # Charakterystyki amplitudowe dla każdego kanału
    freqs, mag0 = compute_mag_response(ir_array[:, 0], fs)
    mag_db = np.empty((len(freqs), n_ch), dtype=np.float32)
    mag_db[:, 0] = mag0
    for ch in range(1, n_ch):
        _, mag_c = compute_mag_response(ir_array[:, ch], fs)
        mag_db[:, ch] = mag_c

    return ir_array, freqs, mag_db, recorded



def smooth_mag_response(freqs, mag_db, fraction=6):
    """
    Fractional-octave smoothing – szybka implementacja O(N)
    z oknem szerokości 1/fraction oktawy w skali log2(f).
    """
    freqs = np.asarray(freqs)
    mag_db = np.asarray(mag_db)
    smoothed = np.empty_like(mag_db)

    n = len(freqs)
    if n == 0:
        return smoothed

    # logarytm częstotliwości (log2, bo wygodnie w oktawach)
    logf = np.log2(np.maximum(freqs, 1e-12))
    half = 1.0 / (2.0 * fraction)  # połowa szerokości okna w oktawach

    left = 0
    right = 0

    for i in range(n):
        f = freqs[i]
        if f <= 0:
            smoothed[i] = mag_db[i]
            continue

        center = logf[i]
        low = center - half
        high = center + half

        # przesuwamy lewy wskaźnik, aż wejdziemy w okno
        while left < n and logf[left] < low:
            left += 1
        # przesuwamy prawy wskaźnik, dopóki jesteśmy w oknie
        while right < n and logf[right] <= high:
            right += 1

        if right > left:
            smoothed[i] = mag_db[left:right].mean()
        else:
            smoothed[i] = mag_db[i]

    return smoothed


