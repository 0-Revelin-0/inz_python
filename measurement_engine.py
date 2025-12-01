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
    Odtwarza sweep na wyjściu i JEDNOCZEŚNIE nagrywa wejście,
    używając osobnego OutputStream i InputStream (manualny full-duplex).

    To omija problemy sd.playrec() z PaError -9998 ("Invalid number of channels")
    na wielu sterownikach (Realtek, WASAPI itd.).
    """
    silence = np.zeros(int(fs * extra_silence), dtype=np.float32)
    play_sig = np.concatenate([sweep, silence]).astype(np.float32)

    # Kolejka na zarejestrowane próbki z callbacku wejściowego
    q = queue.Queue()

    # --- CALLBACK WEJŚCIA ---
    def input_callback(indata, frames, time, status):
        if status:
            print("Input status:", status)
        q.put(indata.copy())

    # --- CALLBACK WYJŚCIA ---
    play_pos = 0

    def output_callback(outdata, frames, time, status):
        nonlocal play_pos
        if status:
            print("Output status:", status)

        end = min(play_pos + frames, len(play_sig))
        chunk = play_sig[play_pos:end]

        # wypełnij to, co mamy
        outdata[:, 0] = 0.0
        outdata[:len(chunk), 0] = chunk
        play_pos = end

    # Tworzymy dwa osobne streamy
    in_stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        device=audio_cfg["input_device"],
        blocksize=audio_cfg["buffer_size"],
        dtype="float32",
        callback=input_callback,
    )

    out_stream = sd.OutputStream(
        samplerate=fs,
        channels=1,
        device=audio_cfg["output_device"],
        blocksize=audio_cfg["buffer_size"],
        dtype="float32",
        callback=output_callback,
    )

    # Startujemy oba
    in_stream.start()
    out_stream.start()

    # Czekamy, aż cały sweep zostanie odtworzony
    while True:
        if play_pos >= len(play_sig):
            break
        sd.sleep(10)  # krótkie "drzemki", żeby nie blokować CPU

    # Zatrzymujemy i zamykamy strumienie
    out_stream.stop()
    in_stream.stop()
    out_stream.close()
    in_stream.close()

    # Zbieramy wszystko z kolejki
    recorded_blocks = []
    while not q.empty():
        recorded_blocks.append(q.get())

    if not recorded_blocks:
        raise RuntimeError("Brak nagranego sygnału podczas pomiaru IR.")

    recorded = np.concatenate(recorded_blocks, axis=0).flatten()

    return recorded.copy()




def deconvolve_ir(recorded, inverse_filter, fs, ir_length_s, fade_time_s):
    """
    Dekonwolucja (ESS * inverse filter) → IR.
    Przycinamy do ir_length_s i robimy fade na końcu.
    """
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

    # ================================
    # 1. Przesunięcie impulsu do t=0
    # ================================
    peak = np.argmax(np.abs(ir))
    ir = np.roll(ir, -peak)

    # --- Auto trimming IR ---
    # ir = trim_ir(ir, fs, pre_ms=50, post_ms=100, tail_drop_db=60)

    # ================================
    # 2. Usunięcie harmonicznych
    # (zostawiamy tylko pierwsze 0.5 s IR)
    # ================================
    # cut = int(0.5 * fs)
    # if cut < len(ir):
    #     ir = ir[:cut]

    ir_samples = len(ir)

    # fade na końcu IR
    fade_samples = int(fade_time_s * fs)
    if fade_samples > 0 and fade_samples < ir_samples:
        window = np.ones(ir_samples, dtype=np.float32)
        tail = np.linspace(1.0, 0.0, fade_samples, endpoint=True)
        window[-fade_samples:] = tail
        ir *= window

    # normalizacja, żeby nie wyskoczyć z zakresu przy zapisie
    max_val = np.max(np.abs(ir) + 1e-12)
    ir = ir / max_val

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
          "input_device": int,
          "output_device": int,
          "sample_rate": int,
          "buffer_size": int
        }

    Zwraca:
        ir        - numpy array (float32)
        freqs     - numpy array [Hz]
        mag_db    - numpy array [dB]
    """
    fs = int(audio_cfg["sample_rate"])

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


    ir = deconvolve_ir(
        recorded,
        inv,
        fs,
        params["ir_length"],
        params["fade_time"],
    )

    freqs, mag_db = compute_mag_response(ir, fs)

    return ir, freqs, mag_db, recorded


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


