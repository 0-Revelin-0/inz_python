# measurement_engine.py
import queue
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
    Odtwarza sweep na wyjściu i jednocześnie nagrywa wejście.

    Obsługuje:
    - MONO:  1 kanał z input_device
    - STEREO wariant A: 2 kanały z JEDNEGO device'a (input_device_L == input_device_R)
    - STEREO wariant B: 2 osobne device'y (input_device_L != input_device_R)
    """
    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([sweep, silence]).astype(np.float32)

    channels_in = int(audio_cfg.get("input_channels", 1))
    mode = str(audio_cfg.get("measurement_mode", "Mono")).lower()

    # Aliasy / domyślne wartości
    in_L = audio_cfg.get("input_device_L", audio_cfg.get("input_device"))
    in_R = audio_cfg.get("input_device_R", in_L)
    out_dev = audio_cfg["output_device"]
    blocksize = audio_cfg["buffer_size"]

    # ------------------------------------------------------
    # 1) MONO: klasyczny przypadek – 1 input stream, 1 kanał
    # ------------------------------------------------------
    if channels_in == 1 or mode.startswith("mono"):
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
            channels=1,
            device=in_L,
            blocksize=blocksize,
            dtype="float32",
            callback=input_callback,
        )

        out_stream = sd.OutputStream(
            samplerate=fs,
            channels=1,
            device=out_dev,
            blocksize=blocksize,
            dtype="float32",
            callback=output_callback,
        )

        in_stream.start()
        out_stream.start()

        while play_pos < len(play_sig):
            sd.sleep(10)

        out_stream.stop(); out_stream.close()
        in_stream.stop(); in_stream.close()

        blocks = []
        while not q.empty():
            blocks.append(q.get())

        if not blocks:
            raise RuntimeError("Brak nagranego sygnału podczas pomiaru IR.")

        recorded = np.concatenate(blocks, axis=0)  # (N, 1)
        return recorded[:, 0].copy()

    # ------------------------------------------------------
    # 2) STEREO z jednego device'a (2 kanały)
    # ------------------------------------------------------
    if in_L == in_R:
        q = queue.Queue()

        def input_callback(indata, frames, time, status):
            if status:
                print("Input status (stereo 1 dev):", status)
            q.put(indata.copy())   # (frames, 2)

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
            channels=2,
            device=in_L,
            blocksize=blocksize,
            dtype="float32",
            callback=input_callback,
        )

        out_stream = sd.OutputStream(
            samplerate=fs,
            channels=1,
            device=out_dev,
            blocksize=blocksize,
            dtype="float32",
            callback=output_callback,
        )

        in_stream.start()
        out_stream.start()

        while play_pos < len(play_sig):
            sd.sleep(10)

        out_stream.stop(); out_stream.close()
        in_stream.stop(); in_stream.close()

        blocks = []
        while not q.empty():
            blocks.append(q.get())
        if not blocks:
            raise RuntimeError("Brak nagranego sygnału podczas pomiaru IR (stereo 1 dev).")

        recorded = np.concatenate(blocks, axis=0)  # (N, 2)
        return recorded.copy()

    # ------------------------------------------------------
    # 3) STEREO z dwóch device'ów (L i R osobno)
    # ------------------------------------------------------
    qL = queue.Queue()
    qR = queue.Queue()

    def input_callback_L(indata, frames, time, status):
        if status:
            print("Input L status:", status)
        qL.put(indata.copy())   # (frames,1)

    def input_callback_R(indata, frames, time, status):
        if status:
            print("Input R status:", status)
        qR.put(indata.copy())   # (frames,1)

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

    in_stream_L = sd.InputStream(
        samplerate=fs,
        channels=1,
        device=in_L,
        blocksize=blocksize,
        dtype="float32",
        callback=input_callback_L,
    )

    in_stream_R = sd.InputStream(
        samplerate=fs,
        channels=1,
        device=in_R,
        blocksize=blocksize,
        dtype="float32",
        callback=input_callback_R,
    )

    out_stream = sd.OutputStream(
        samplerate=fs,
        channels=1,
        device=out_dev,
        blocksize=blocksize,
        dtype="float32",
        callback=output_callback,
    )

    in_stream_L.start()
    in_stream_R.start()
    out_stream.start()

    while play_pos < len(play_sig):
        sd.sleep(10)

    out_stream.stop(); out_stream.close()
    in_stream_L.stop(); in_stream_L.close()
    in_stream_R.stop(); in_stream_R.close()

    # Zbierz bloki
    blocks_L, blocks_R = [], []
    while not qL.empty():
        blocks_L.append(qL.get())
    while not qR.empty():
        blocks_R.append(qR.get())

    if not blocks_L or not blocks_R:
        raise RuntimeError("Brak nagranego sygnału na jednym z kanałów L/R.")

    recL = np.concatenate(blocks_L, axis=0)  # (N_L,1)
    recR = np.concatenate(blocks_R, axis=0)  # (N_R,1)

    # przytnij do wspólnej długości
    min_len = min(len(recL), len(recR))
    recL = recL[:min_len, 0]
    recR = recR[:min_len, 0]

    recorded = np.column_stack([recL, recR])  # (N, 2)
    return recorded.copy()





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

    # 2. Fade na końcu IR
    ir_samples = len(ir)
    fade_samples = int(fade_time_s * fs)
    if 0 < fade_samples < ir_samples:
        window = np.ones(ir_samples, dtype=np.float32)
        tail = np.linspace(1.0, 0.0, fade_samples, endpoint=True)
        window[-fade_samples:] = tail
        ir *= window

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


