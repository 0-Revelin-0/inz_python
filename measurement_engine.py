import queue
import numpy as np
import sounddevice as sd



def generate_exponential_sweep(fs, duration, f_start, f_end):
    if f_start <= 0:
        raise ValueError("f_start musi być > 0 Hz")

    if f_end > fs/2:
        raise ValueError("f_end musi być < połowy częstotliwości Nyquista")

    # generowanie parametrów

    # Od od kilku Herz do połowy Nyq f_start i f_stop (dodać)
    f_start = 10
    f_end = fs/2

    R = float(f_end) / float(f_start)
    n_samples = int(fs * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    K = duration * 2.0 * np.pi * f_start / np.log(R)
    L = np.log(R) / duration

    #core generowania sweepa
    sweep = np.sin(K * (np.exp(L * t) - 1.0))

    # minimalny fade-in + fade-out, aby zniwelować strzały
    fade_len = int(fs * 0.005)  # 5 ms
    if fade_len > 1:
        window = np.ones_like(sweep)
        fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))

        window[:fade_len] *= fade
        window[-fade_len:] *= fade[::-1]

        sweep *= window

    # normalizacja
    sweep /= np.max(np.abs(sweep)) + 1e-12 #Zabezpieczenie przed dzieleniem przez 0

    return sweep.astype(np.float32)



def generate_inverse_filter(sweep, fs, f_start, f_end):
    """
    Inverse filter do metody ESS (Farina – Time Reversal Mirror).
    """

    if f_start <= 0:
        raise ValueError("f_start musi być > 0 Hz")

    if f_end > fs / 2:
        raise ValueError("f_end musi być < połowy częstotliwości Nyquista")

    # Od od kilku Herz do połowy Nyq f_start i f_stop (dodać)
    f_start = 10
    f_end = fs / 2

    sweep = np.asarray(sweep, dtype=np.float64)

    # parametr ESS
    n = len(sweep)
    T = n / fs
    t = np.linspace(0.0, T, n, endpoint=False)


    L = np.log(f_end / f_start) / T

    # Time Reversal Mirror oraz ważenie amplitudowe (+3 dB / okt)
    inv = sweep[::-1] * np.exp(-L * t)

    # Normalizacja (stabilna numerycznie)
    inv /= np.max(np.abs(inv)) + 1e-12

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



def playrec_sweep(sweep, fs, audio_cfg, extra_silence=5.0):

    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([sweep, silence]).astype(np.float32)
    return _play_record(play_sig, fs, audio_cfg)


def playrec_sweeps_concat(sweep, fs, audio_cfg, repeats, extra_silence=5.0):

    sweep = np.asarray(sweep, dtype=np.float32)
    concat = np.tile(sweep, repeats).astype(np.float32)

    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([concat, silence]).astype(np.float32)

    return _play_record(play_sig, fs, audio_cfg)



def deconvolve_full(recorded, inverse_filter):

    recorded = np.asarray(recorded, dtype=np.float32)
    inverse_filter = np.asarray(inverse_filter, dtype=np.float32)

    recorded = recorded - np.mean(recorded)

    n_conv = len(recorded) + len(inverse_filter) - 1

    # najbliższa potęga 2 do FFT (wydajność)
    nfft = 1
    while nfft < n_conv:
        nfft *= 2

    R = np.fft.rfft(recorded, nfft)
    I = np.fft.rfft(inverse_filter, nfft)
    ir_full = np.fft.irfft(R * I, nfft)

    # ważne: obcięcie do długości splotu liniowego
    return ir_full[:n_conv].astype(np.float32)


def _extract_segment_from_peak(ir_full, fs, length_s, pre_ms=50):


    ir_full = np.asarray(ir_full, dtype=np.float32)

    peak = int(np.argmax(np.abs(ir_full)))
    pre_samples = int((pre_ms / 1000.0) * fs)

    start = max(0, peak - pre_samples)
    seg_len = int(length_s * fs)
    end = start + seg_len

    if end <= len(ir_full):
        return ir_full[start:end]

    # jeśli brakuje próbek – dopełnij zerami
    out = np.zeros(seg_len, dtype=np.float32)
    available = len(ir_full) - start
    if available > 0:
        out[:available] = ir_full[start:]
    return out


def _extract_segment_from_index(x, start_idx, fs, length_s, pre_ms=50):


    x = np.asarray(x, dtype=np.float32)

    pre_samples = int((pre_ms / 1000.0) * fs)
    start = int(max(0, start_idx - pre_samples))

    seg_len = int(length_s * fs)
    end = start + seg_len

    if end <= len(x):
        return x[start:end]

    # jeśli brakuje próbek – dopełnij zerami
    out = np.zeros(seg_len, dtype=np.float32)
    available = len(x) - start
    if available > 0:
        out[:available] = x[start:]
    return out




def _normalize_signal(x):
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.max(np.abs(x)) + 1e-12)).astype(np.float32)




def compute_mag_response(ir, fs):

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
          "averages":     int   [liczba uśrednień]
        }

    audio_cfg: dict
        {
          "input_device":    int,
          "output_device":   int,
          "sample_rate":     int,
          "buffer_size":     int,
          "input_channels":  int (opcjonalne, ale domyślnie 1)
        }

    Zwraca:
        ir        - IR (mono: N, stereo: N×2)
        freqs     - częstotliwości FFT
        mag_db    - charakterystyka magnitude
        recorded  - surowe nagranie
    """

    fs = int(audio_cfg["sample_rate"])
    channels_in = int(audio_cfg.get("input_channels", 1))
    mode = params.get("mode", "single")


    # Generacja sweepa i inverse filter
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

    sweep_len_s = float(params["sweep_length"])
    sweep_samp = len(sweep)
    ir_length_s = float(params["ir_length"])
    ir_block_samp = int(ir_length_s * fs)
    avg_count = max(1, int(params.get("averages", 1)))

    # Pojedyńczy pomiar - sanity check
    if mode == "single":
        avg_count = 1  # zawsze 1

    # Uśredniony pomiar - sanity check
    elif mode == "average":
        # IR length musi być równe sweep length
        ir_length_s = sweep_len_s
        # wymuszenie poprawnej wartości
        params["ir_length"] = sweep_len_s
        # Averaging musi być >= 2
        avg_count = max(2, avg_count)


    # Pojedyńczy pomiar worker
    if avg_count == 1:
        recorded = playrec_sweep(
            sweep,
            fs,
            audio_cfg,
            extra_silence=ir_length_s,
        )

        # MONO
        if channels_in == 1 or (np.ndim(recorded) == 1):
            rec_mono = recorded[:, 0] if np.ndim(recorded) > 1 else recorded

            ir_full = deconvolve_full(rec_mono, inv)
            ir = _extract_segment_from_peak(ir_full, fs, ir_length_s)
            ir = _normalize_signal(ir)

            freqs, mag_db = compute_mag_response(ir, fs)
            return ir.astype(np.float32), freqs, mag_db, rec_mono

        # STEREO
        rec_arr = np.asarray(recorded, dtype=np.float32)
        if rec_arr.ndim == 1:
            rec_arr = rec_arr.reshape(-1, 1)

        # Dekonwolucja pełna dla każdego kanału
        ir_full_list = []
        for ch in range(rec_arr.shape[1]):
            ir_full_ch = deconvolve_full(rec_arr[:, ch], inv)
            ir_full_list.append(ir_full_ch)

        # znalezienie peak w każdym kanale
        peaks = [int(np.argmax(np.abs(ir))) for ir in ir_full_list]

        # wybór najwcześniejszego peaku (minimum indeksu)
        peak0 = min(peaks)

        # Wycięcie tego samego okna z każdego kanału
        ir_list = []
        for ch in range(len(ir_full_list)):
            ir_ch = _extract_segment_from_index(ir_full_list[ch], peak0, fs, ir_length_s)
            ir_list.append(ir_ch)

        # Złóż kanały do wspólnej długości (wydłuża uzupełniając zerami)
        max_len = max(map(len, ir_list))
        ir_array = np.stack(
            [np.pad(x, (0, max_len - len(x))) for x in ir_list],
            axis=1
        )

        # Normalizacja wspólnym współczynnikiem (zachowuje relacje L/R)
        ir_array = ir_array / (np.max(np.abs(ir_array)) + 1e-12)

        # magnitude
        freqs, mag0 = compute_mag_response(ir_array[:, 0], fs)
        mag_db = np.zeros((len(freqs), ir_array.shape[1]), dtype=np.float32)
        mag_db[:, 0] = mag0
        for ch in range(1, ir_array.shape[1]):
            _, mag_db[:, ch] = compute_mag_response(ir_array[:, ch], fs)

        return ir_array.astype(np.float32), freqs, mag_db, rec_arr

    # Uśredniony pomiar - worker
    if mode == "average":
        ir_block_samp = sweep_samp
    else:
        ir_block_samp = int(ir_length_s * fs)

    repeats = avg_count + 1   # Powtórzeń o jeden więcej bo pierwsze niekompletne okienko wyrzucamy

    recorded_full = playrec_sweeps_concat(
        sweep,
        fs,
        audio_cfg,
        repeats=repeats,
        extra_silence=0.0,
    )

    recorded_full = np.asarray(recorded_full, dtype=np.float32)

    # ------------------ MONO ------------------
    if channels_in == 1 or recorded_full.ndim == 1:
        rec_mono = recorded_full[:, 0] if recorded_full.ndim > 1 else recorded_full

        ir_full = deconvolve_full(rec_mono, inv)

        n_total = len(ir_full)
        max_blocks = (n_total - ir_block_samp) // sweep_samp
        num_blocks = min(max_blocks, repeats)

        #Wycinanie okien i uśrednianie
        ir_blocks = []
        for k in range(1, num_blocks):
            start = k * sweep_samp
            end = start + ir_block_samp
            if end > n_total:
                break
            ir_blocks.append(ir_full[start:end].astype(np.float32))

        if not ir_blocks:
            # fallback: wyciągnij pojedynczą IR z całego przebiegu
            ir = _extract_segment_from_peak(ir_full, fs, ir_length_s)
        else:
            ir_stack = np.stack(ir_blocks, axis=0)
            ir = ir_stack.mean(axis=0).astype(np.float32)

        #Wycinanie okna od peaku
        ir = _extract_segment_from_peak(ir, fs, ir_length_s)
        ir = _normalize_signal(ir)

        freqs, mag_db = compute_mag_response(ir, fs)
        return ir.astype(np.float32), freqs, mag_db, rec_mono

    # ------------------ STEREO ------------------
    rec_arr = recorded_full
    if rec_arr.ndim == 1:
        rec_arr = rec_arr.reshape(-1, 1)

    n_ch = rec_arr.shape[1]
    ir_channels = []

    for ch in range(n_ch):
        rec_ch = rec_arr[:, ch]
        ir_full_ch = deconvolve_full(rec_ch, inv)

        n_total = len(ir_full_ch)
        max_blocks = (n_total - ir_block_samp) // sweep_samp
        num_blocks = min(max_blocks, repeats)

        blocks = []
        for k in range(1, num_blocks):
            start = k * sweep_samp
            end = start + ir_block_samp
            if end > n_total:
                break
            blocks.append(ir_full_ch[start:end].astype(np.float32))

        if not blocks:
            # fallback: wyciągnij pojedynczą IR z całego przebiegu
            ir_ch = _extract_segment_from_peak(ir_full_ch, fs, ir_length_s)
        else:
            stack_ch = np.stack(blocks, axis=0)
            ir_ch = stack_ch.mean(axis=0).astype(np.float32)

        ir_channels.append(ir_ch)

    # Wydłuża kanały do wspólnej długości (wydłuża uzupełniając zerami)
    max_len = max(map(len, ir_channels))
    ir_array = np.stack(
        [np.pad(x, (0, max_len - len(x))) for x in ir_channels],
        axis=1
    )

    # znalezienie peak w każdym kanale
    peaks = [int(np.argmax(np.abs(ir))) for ir in ir_array]

    # wybór najwcześniejszego peaku (minimum indeksu)
    peak0 = min(peaks)

    ir_list = []
    for ch in range(n_ch):
        ir_list.append(_extract_segment_from_index(ir_array[:, ch], peak0, fs, ir_length_s))

    # Normalizacja wspólnym współczynnikiem
    ir_array = ir_array / (np.max(np.abs(ir_array)) + 1e-12)

    # magnitude
    freqs, mag0 = compute_mag_response(ir_array[:, 0], fs)
    mag_db = np.zeros((len(freqs), n_ch), dtype=np.float32)
    mag_db[:, 0] = mag0

    for ch in range(1, n_ch):
        _, mag_db[:, ch] = compute_mag_response(ir_array[:, ch], fs)

    return ir_array.astype(np.float32), freqs, mag_db, rec_arr


def smooth_mag_response(freqs, mag_db, fraction=6):

    # Konwersja wejść do tablic
    freqs = np.asarray(freqs)
    mag_db = np.asarray(mag_db)

    # Tablica wyjściowa – ten sam kształt co mag_db
    smoothed = np.empty_like(mag_db)

    # Liczba punktów charakterystyki
    n = len(freqs)
    if n == 0:
        # Brak danych
        return smoothed

    # Logarytm częstotliwości w podstawie 2:
    # +1 w log2 = wzrost o jedną oktawę
    logf = np.log2(np.maximum(freqs, 1e-12))

    # Połowa szerokości okna wygładzania w oktawach
    # Całe okno ma szerokość 1/fraction oktawy
    half = 1.0 / (2.0 * fraction)

    # Wskaźniki lewego i prawego brzegu okna
    # Dzięki temu nie liczymy zakresu od nowa dla każdego punktu
    left = 0
    right = 0

    # Iteracja po wszystkich punktach charakterystyki
    for i in range(n):
        f = freqs[i]

        # Punkt 0 Hz (DC) – brak sensownego okna oktawowego
        # Przepisujemy wartość bez wygładzania
        if f <= 0:
            smoothed[i] = mag_db[i]
            continue

        # Pozycja bieżącej częstotliwości w skali log2
        center = logf[i]

        # Dolna i górna granica okna wygładzania
        low = center - half
        high = center + half

        # Przesuwanie lewego wskaźnika:
        # pomijamy częstotliwości poniżej dolnej granicy okna
        while left < n and logf[left] < low:
            left += 1

        # Przesuwanie prawego wskaźnika:
        # włączamy częstotliwości aż do górnej granicy okna
        while right < n and logf[right] <= high:
            right += 1

        # Jeśli okno zawiera co najmniej jeden punkt
        if right > left:

            # # Konwersja dB → amplituda liniowa
            # lin = 10.0 ** (mag_db[left:right] / 20.0)
            #
            # # Moc to amplituda^2
            # power = lin ** 2
            #
            # # Średnia mocy w oknie oktawowym
            # mean_power = power.mean()
            #
            # # Powrót do skali dB
            # smoothed[i] = 10.0 * np.log10(mean_power + 1e-30)

            smoothed[i] = mag_db[left:right].mean()

        else:
            # Gdyby okno było puste
            smoothed[i] = mag_db[i]

    # Zwrócenie wygładzonej charakterystyki
    return smoothed



