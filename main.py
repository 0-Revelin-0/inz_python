import customtkinter as ctk

try:
    import customtkinter.windows.ctk_tk
    customtkinter.windows.ctk_tk.CTk._check_dpi_scaling = lambda *a, **k: None
    customtkinter.windows.ctk_tk.CTk._update_checks = lambda *a, **k: None
except:
    pass

import sounddevice as sd
import threading
import numpy as np
import os
import time
import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Fix: wyłączenie wewnętrznych after() CustomTkinter ---



# --------------------------------------------------
# KONFIGURACJA GLOBALNA
# --------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-red.json")  # plik obok main.py



def show_error(message: str):
    dlg = ctk.CTkToplevel()
    dlg.title("Błąd audio")
    dlg.geometry("350x140")

    label = ctk.CTkLabel(dlg, text=message, wraplength=300)
    label.pack(pady=15)

    btn = ctk.CTkButton(dlg, text="OK", command=dlg.destroy)
    btn.pack(pady=10)

# --------------------------------------------------
# PODSTRONY
# --------------------------------------------------
class MeasurementPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.is_measuring = False
        self.output_folder = os.getcwd()

        self.ir_data = None
        self.ir_samplerate = None

        # --- globalny styl matplotlib (nowoczesne dark UI) ---
        import matplotlib
        matplotlib.use("TkAgg")
        plt.style.use("dark_background")

        matplotlib.rcParams.update({
            "axes.facecolor": "#111111",
            "figure.facecolor": "#111111",
            "axes.edgecolor": "#bbbbbb",
            "axes.labelcolor": "#ffffff",
            "xtick.color": "#cccccc",
            "ytick.color": "#cccccc",
            "grid.color": "#555555",
            "grid.alpha": 0.3,
            "lines.linewidth": 1.4,
            "font.size": 10,
        })

        # --- układ strony ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        title = ctk.CTkLabel(
            self,
            text="🎤 Pomiar odpowiedzi impulsowej",
            font=("Roboto", 24, "bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        subtitle = ctk.CTkLabel(
            self,
            text="Ustaw urządzenia w zakładce Ustawienia.\n"
                 "Skalibruj SPL, wybierz folder i rozpocznij pomiar.",
            justify="left"
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # ------------------- główna ramka ---------------------
        main_frame = ctk.CTkFrame(self, corner_radius=12)
        main_frame.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")

        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # ======================================================
        # LEWA RAMKA — USTAWIENIA
        # ======================================================
        config = ctk.CTkFrame(main_frame, corner_radius=12)
        config.grid(row=0, column=0, padx=15, pady=15, sticky="nsw")

        for r in range(20):
            config.grid_rowconfigure(r, weight=0)
        config.grid_columnconfigure(1, weight=1)

        # --- Calibrate SPL ---
        ctk.CTkLabel(config, text="Kalibracja SPL:", font=("Roboto", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 5), sticky="w"
        )

        self.calib_status = ctk.CTkLabel(config, text="Poziom niezmierzony.", wraplength=220)
        self.calib_status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))

        self.calib_btn = ctk.CTkButton(config, text="Calibrate SPL", command=self._on_calibrate_spl)
        self.calib_btn.grid(row=2, column=0, columnspan=2, pady=(0, 15), sticky="we")

        # --- Ustawienia pomiaru ---
        ctk.CTkLabel(config, text="Parametry pomiaru:", font=("Roboto", 14, "bold")).grid(
            row=3, column=0, columnspan=2, pady=(5, 5), sticky="w"
        )

        # Sweep duration
        ctk.CTkLabel(config, text="Długość sweepa [s]:").grid(row=4, column=0, sticky="w", pady=3)
        self.sweep_entry = ctk.CTkEntry(config, width=80)
        self.sweep_entry.insert(0, "5")
        self.sweep_entry.grid(row=4, column=1, sticky="w")

        # Start freq
        ctk.CTkLabel(config, text="Start freq [Hz]:").grid(row=5, column=0, sticky="w", pady=3)
        self.fstart_entry = ctk.CTkEntry(config, width=80)
        self.fstart_entry.insert(0, "20")
        self.fstart_entry.grid(row=5, column=1, sticky="w")

        # End freq
        ctk.CTkLabel(config, text="End freq [Hz]:").grid(row=6, column=0, sticky="w", pady=3)
        self.fend_entry = ctk.CTkEntry(config, width=80)
        self.fend_entry.insert(0, "20000")
        self.fend_entry.grid(row=6, column=1, sticky="w")

        # IR length
        ctk.CTkLabel(config, text="Długość IR [s]:").grid(row=7, column=0, sticky="w", pady=3)
        self.irlen_entry = ctk.CTkEntry(config, width=80)
        self.irlen_entry.insert(0, "3")
        self.irlen_entry.grid(row=7, column=1, sticky="w")

        # Fade
        ctk.CTkLabel(config, text="Fade [s]:").grid(row=8, column=0, sticky="w", pady=3)
        self.fade_entry = ctk.CTkEntry(config, width=80)
        self.fade_entry.insert(0, "0.05")
        self.fade_entry.grid(row=8, column=1, sticky="w")

        # --- folder outputu ---
        ctk.CTkLabel(config, text="Folder zapisu IR:", font=("Roboto", 14, "bold")).grid(
            row=10, column=0, columnspan=2, pady=(10, 5), sticky="w"
        )

        self.folder_var = ctk.StringVar(value=self.output_folder)
        ctk.CTkEntry(config, textvariable=self.folder_var, width=220).grid(
            row=11, column=0, pady=2, sticky="w"
        )

        ctk.CTkButton(config, text="Wybierz folder…", command=self._choose_folder).grid(
            row=11, column=1, pady=2, sticky="w"
        )

        # --- Start measurement ---
        self.measure_btn = ctk.CTkButton(
            config, text="Start measurement", command=self._on_start_measurement
        )
        self.measure_btn.grid(row=15, column=0, columnspan=2, pady=(20, 5), sticky="we")

        self.status_label = ctk.CTkLabel(config, text="Gotowy do pomiaru.", wraplength=220)
        self.status_label.grid(row=16, column=0, columnspan=2, pady=5, sticky="w")

        # ======================================================
        # PRAWA RAMKA — WYKRESY
        # ======================================================
        plot = ctk.CTkFrame(main_frame, corner_radius=12)
        plot.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        plot.grid_columnconfigure(0, weight=1)
        plot.grid_rowconfigure(0, weight=1)

        # --- Matplotlib figure ---
        self.fig, (self.ax_ir, self.ax_mag) = plt.subplots(2, 1, figsize=(6, 5), dpi=100)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self._clear_plots()

    # ============================================================
    # FUNKCJE POMOCNICZE
    # ============================================================
    def _get_audio_settings(self):
        settings = self.controller.pages["settings"]
        out_idx = settings.get_selected_output_index()
        in_idx = settings.get_selected_input_index()
        sr = int(settings.sample_rate_combo.get())
        return {"out": out_idx, "inp": in_idx, "sr": sr}

    def _choose_folder(self):
        folder = fd.askdirectory(initialdir=self.output_folder)
        if folder:
            self.output_folder = folder
            self.folder_var.set(folder)

    # ============================================================
    # KALIBRACJA SPL
    # ============================================================
    def _on_calibrate_spl(self):
        if self.is_measuring:
            return

        audio = self._get_audio_settings()
        if audio["out"] is None or audio["inp"] is None:
            self.status_label.configure(text="Błąd: brak poprawnie ustawionych urządzeń.")
            return

        self.status_label.configure(text="Kalibracja SPL…")
        threading.Thread(target=self._run_calibration, args=(audio,), daemon=True).start()

    def _run_calibration(self, audio):
        sr = audio["sr"]
        t = np.linspace(0, 2.0, int(sr * 2), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

        try:
            rec = sd.playrec(
                tone[:, np.newaxis],
                samplerate=sr,
                device=(audio["out"], audio["inp"]),
                channels=1,
                dtype="float32",
                blocking=True,
            )[:, 0]
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Błąd SPL: {e}"))
            return

        rms = np.sqrt(np.mean(rec ** 2)) + 1e-12
        peak = np.max(np.abs(rec))
        dbfs = 20 * np.log10(rms)

        if peak > 0.999:
            msg = "⚠️ CLIPPING – obniż głośność!"
        elif dbfs < -40:
            msg = f"🔈 Za cicho (~{dbfs:.1f} dBFS)"
        elif -40 <= dbfs < -20:
            msg = f"ℹ️ OK, ale trochę cicho (~{dbfs:.1f} dBFS)"
        else:
            msg = f"✅ Poziom dobry (~{dbfs:.1f} dBFS)"

        self.after(0, lambda: self.calib_status.configure(text=msg))
        self.after(0, lambda: self.status_label.configure(text="Kalibracja zakończona."))

    # ============================================================
    # START MEASUREMENT
    # ============================================================
    def _on_start_measurement(self):
        if self.is_measuring:
            return

        audio = self._get_audio_settings()

        # pobranie parametrów
        try:
            sweep_dur = float(self.sweep_entry.get())
            ir_len = float(self.irlen_entry.get())
            f1 = float(self.fstart_entry.get())
            f2 = float(self.fend_entry.get())
            fade = float(self.fade_entry.get())
        except:
            self.status_label.configure(text="Błędne wartości w ustawieniach!")
            return

        self.status_label.configure(text="Trwa pomiar…")
        self._clear_plots()

        self.is_measuring = True
        threading.Thread(
            target=self._run_measurement,
            args=(audio, sweep_dur, ir_len, f1, f2, fade),
            daemon=True
        ).start()

    def _run_measurement(self, audio, T, ir_len, f1, f2, fade):
        sr = audio["sr"]

        # --- Generacja sweepa logarytmicznego ---
        t = np.linspace(0, T, int(sr * T), endpoint=False)
        K = T / np.log(f2 / f1)
        sweep = np.sin(2 * np.pi * f1 * K * (np.exp(t / K) - 1)).astype(np.float32)

        # --- Fade in/out ---
        fade_n = int(fade * sr)
        sweep[:fade_n] *= np.linspace(0, 1, fade_n)
        sweep[-fade_n:] *= np.linspace(1, 0, fade_n)

        # --- Filtr odwrotny ---
        w = np.exp(-t / K)
        inv = (sweep[::-1] * w).astype(np.float32)

        # --- nagranie ---
        try:
            rec = sd.playrec(
                sweep[:, None],
                samplerate=sr,
                device=(audio["out"], audio["inp"]),
                channels=1,
                dtype="float32",
                blocking=True,
            )[:, 0]
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Błąd pomiaru: {e}"))
            self.is_measuring = False
            return

        # --- Dekonwolucja ---
        N = len(rec) + len(inv)
        nfft = 1 << (N - 1).bit_length()
        IR = np.fft.irfft(
            np.fft.rfft(rec, nfft) * np.fft.rfft(inv, nfft),
            n=nfft
        )[: int(ir_len * sr)]

        IR = IR / (np.max(np.abs(IR)) + 1e-12)
        IR = IR.astype(np.float32)

        # zapis
        ts = time.strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(self.output_folder, f"IR_{ts}.wav")
        self._save_wav(outpath, sr, IR)

        self.ir_data = IR
        self.ir_samplerate = sr

        self.after(0, lambda: self.status_label.configure(text=f"Pomiar zakończony.\nZapisano: {outpath}"))
        self.after(0, lambda: self._update_plots(IR, sr))

        self.is_measuring = False

    # ============================================================
    # Zapis WAV
    # ============================================================
    def _save_wav(self, path, sr, data):
        import wave
        data16 = (np.clip(data, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(data16.tobytes())

    # ============================================================
    # WYKRESY
    # ============================================================
    def _clear_plots(self):
        self.ax_ir.clear()
        self.ax_ir.set_title("Impulse Response")
        self.ax_ir.set_xlabel("Czas [s]")
        self.ax_ir.set_ylabel("Amplituda")
        self.ax_ir.grid(True, alpha=0.25)

        self.ax_mag.clear()
        self.ax_mag.set_title("Magnitude Response")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]")
        self.ax_mag.set_ylabel("Poziom [dB]")
        self.ax_mag.grid(True, which="both", alpha=0.25)

        self.canvas.draw()

    def _update_plots(self, ir, sr):
        # IR
        t = np.arange(len(ir)) / sr
        self.ax_ir.clear()
        self.ax_ir.plot(t, ir, color="#ff4444")
        self.ax_ir.grid(True, alpha=0.25)
        self.ax_ir.set_title("Impulse Response")
        self.ax_ir.set_xlabel("Czas [s]")
        self.ax_ir.set_ylabel("Amplituda")

        # FFT
        freq = np.fft.rfftfreq(len(ir), 1 / sr)
        mag = 20 * np.log10(np.abs(np.fft.rfft(ir)) + 1e-12)

        self.ax_mag.clear()
        self.ax_mag.semilogx(freq, mag, color="#55aaff")
        self.ax_mag.set_xlim(20, sr / 2)
        self.ax_mag.set_ylim(np.max(mag) - 60, np.max(mag) + 3)
        self.ax_mag.grid(True, which="both", alpha=0.25)
        self.ax_mag.set_title("Magnitude Response")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]")
        self.ax_mag.set_ylabel("Poziom [dB]")

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()




class GeneratorPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        title = ctk.CTkLabel(
            self,
            text="🌊 Generator odpowiedzi impulsowej",
            font=("Roboto", 24, "bold")
        )
        title.pack(pady=(20, 10), anchor="w", padx=20)

        subtitle = ctk.CTkLabel(
            self,
            text="Tutaj będziesz generował własne odpowiedzi impulsowe\n"
                 "(parametry filtra, czas trwania, częstotliwości itd.).",
            justify="left"
        )
        subtitle.pack(pady=(0, 20), anchor="w", padx=20)

        gen_button = ctk.CTkButton(self, text="Wygeneruj IR")
        gen_button.pack(pady=10, padx=20, anchor="w")

class InputMonitor:
    def __init__(self, progress_bar, input_device_index: int, samplerate: int = 48000):
        self.progress_bar = progress_bar
        self.running = False
        self.device = input_device_index
        self.samplerate = samplerate
        self.stream = None

    def start(self):
        if self.running:
            return
        self.running = True

        def audio_callback(indata, frames, time, status):
            if status:
                print("status:", status)
            if not self.running:
                return
            rms = float(np.sqrt(np.mean(indata ** 2)))
            level = min(rms * 8, 1.0)
            self.progress_bar.set(level)

        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.samplerate,
                callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            show_error(f"Nie udało się otworzyć wejścia audio.\n\nPowód:\n{e}")
            self.running = False
            self.progress_bar.set(0)

    def stop(self):
        if self.running:
            self.running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.progress_bar.set(0)


def play_test_tone(output_device_index: int, samplerate: int = 48000, duration: float = 3.0):
    """Plays 1 kHz test sine wave."""
    def _worker():
        t = np.linspace(0, duration, int(duration * samplerate), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 1000 * t)
        sd.play(tone, samplerate=samplerate, device=output_device_index)
        sd.wait()

    threading.Thread(target=_worker, daemon=True).start()



class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.audio_devices = []
        self.input_monitor = None

        # ---------------- Header ----------------
        title = ctk.CTkLabel(self, text="⚙️ Ustawienia", font=("Roboto", 28, "bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        subtitle = ctk.CTkLabel(self, text="Skonfiguruj urządzenia audio dla pomiaru i generowania IR.")
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # ---------------- TAB VIEW ----------------
        self.tabs = ctk.CTkTabview(self, width=700, height=500)
        self.tabs.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")

        self.tabs.add("Ustawienia pomiaru")
        self.tabs.add("Ustawienia generowania")

        # ---------------------------------------------------------
        # TAB 1 — USTAWIENIA POMIARU (WSZYSTKO CO JUŻ MAMY)
        # ---------------------------------------------------------
        measure_tab = self.tabs.tab("Ustawienia pomiaru")
        measure_tab.grid_columnconfigure(1, weight=1)

        frame = ctk.CTkFrame(measure_tab, corner_radius=12)
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)

        # ---------------- LISTA URZĄDZEŃ ----------------
        ctk.CTkLabel(frame, text="Audio device:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.device_combo = ctk.CTkComboBox(frame, values=[], command=self._on_device_change)
        self.device_combo.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Input channel
        ctk.CTkLabel(frame, text="Input channel:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.input_combo = ctk.CTkComboBox(frame, values=[])
        self.input_combo.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Output channel
        ctk.CTkLabel(frame, text="Output channel:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.output_combo = ctk.CTkComboBox(frame, values=[])
        self.output_combo.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Sample rate
        ctk.CTkLabel(frame, text="Sample rate:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.sample_rate_combo = ctk.CTkComboBox(frame, values=["44100", "48000", "88200", "96000", "192000"])
        self.sample_rate_combo.set("48000")
        self.sample_rate_combo.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

        # Buffer size
        ctk.CTkLabel(frame, text="Buffer size (frames):").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.buffer_size_combo = ctk.CTkComboBox(frame, values=["64", "128", "256", "512", "1024"])
        self.buffer_size_combo.set("256")
        self.buffer_size_combo.grid(row=4, column=1, padx=10, pady=10, sticky="ew")

        # Test tone
        self.test_btn = ctk.CTkButton(frame, text="Test tone (1 kHz)", command=self._play_test)
        self.test_btn.grid(row=5, column=1, padx=10, pady=15, sticky="w")

        # Input meter
        ctk.CTkLabel(frame, text="Input level:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.meter_bar = ctk.CTkProgressBar(frame, width=220)
        self.meter_bar.grid(row=6, column=1, padx=10, pady=10, sticky="w")
        self.meter_bar.set(0)

        self.meter_btn = ctk.CTkButton(frame, text="Start input meter", command=self._toggle_meter)
        self.meter_btn.grid(row=7, column=1, padx=10, pady=10, sticky="w")

        # Load devices initially
        self._load_devices()

        # ---------------------------------------------------------
        # TAB 2 — USTAWIENIA GENEROWANIA (na razie puste)
        # ---------------------------------------------------------
        gen_tab = self.tabs.tab("Ustawienia generowania")

        empty_label = ctk.CTkLabel(
            gen_tab,
            text="Tu pojawią się ustawienia generowania IR.\nNa razie strona jest pusta.",
            font=("Roboto", 16),
            justify="center"
        )
        empty_label.pack(pady=40)


    # =====================================================================
    # --- DEVICE HANDLING (jak wcześniej) ---
    # =====================================================================

    def _load_devices(self):
        all_dev = sd.query_devices()
        grouped = {}

        junk = ["mapper", "mapowanie", "default", "podstawowy", "smart sound", "microsoft"]

        for idx, dev in enumerate(all_dev):
            name = dev["name"]

            if any(j in name.lower() for j in junk):
                continue

            if "(" in name and ")" in name:
                label = name[:name.rfind("(")].strip()
                base = name[name.rfind("(")+1:name.rfind(")")].strip()
            else:
                base = name
                label = name

            if base not in grouped:
                grouped[base] = {"name": base, "inputs": [], "outputs": []}

            if dev["max_input_channels"] > 0:
                grouped[base]["inputs"].append({"label": f"Input – {label}", "index": idx})
            if dev["max_output_channels"] > 0:
                grouped[base]["outputs"].append({"label": f"Output – {label}", "index": idx})

        self.audio_devices = list(grouped.values())

        dev_names = [d["name"] for d in self.audio_devices]
        self.device_combo.configure(values=dev_names)

        if dev_names:
            self.device_combo.set(dev_names[0])
            self._on_device_change(dev_names[0])


    def _on_device_change(self, device_name):
        dev = next((d for d in self.audio_devices if d["name"] == device_name), None)
        if not dev:
            return

        inputs = [i["label"] for i in dev["inputs"]] or ["Brak wejścia"]
        outputs = [o["label"] for o in dev["outputs"]] or ["Brak wyjścia"]

        self.input_combo.configure(values=inputs)
        self.output_combo.configure(values=outputs)

        self.input_combo.set(inputs[0])
        self.output_combo.set(outputs[0])

    # =====================================================================
    # --- TEST TONE + INPUT METER ---
    # =====================================================================

    def _play_test(self):
        out_idx = self.get_selected_output_index()
        if out_idx is None:
            show_error("Nie wybrano wyjścia audio.")
            return

        try:
            play_test_tone(out_idx, samplerate=int(self.sample_rate_combo.get()))
        except Exception as e:
            show_error(f"Nie udało się odtworzyć sygnału testowego.\n\nSzczegóły:\n{e}")

    def _toggle_meter(self):
        in_idx = self.get_selected_input_index()
        if in_idx is None:
            show_error("Nie wybrano wejścia audio.")
            return

        try:
            if self.input_monitor is None or not self.input_monitor.running:
                self.input_monitor = InputMonitor(self.meter_bar, in_idx,
                                                  samplerate=int(self.sample_rate_combo.get()))
                self.input_monitor.start()
                self.meter_btn.configure(text="Stop input meter")
            else:
                self.input_monitor.stop()
                self.meter_btn.configure(text="Start input meter")
        except Exception as e:
            show_error(f"Nie udało się uruchomić input metera.\n\nSzczegóły:\n{e}")

    # =====================================================================
    # --- HELPERS ---
    # =====================================================================

    def get_selected_input_index(self):
        dev = next((d for d in self.audio_devices if d["name"] == self.device_combo.get()), None)
        if not dev:
            return None

        selected = self.input_combo.get()
        for i in dev["inputs"]:
            if i["label"] == selected:
                return i["index"]
        return None

    def get_selected_output_index(self):
        dev = next((d for d in self.audio_devices if d["name"] == self.device_combo.get()), None)
        if not dev:
            return None

        selected = self.output_combo.get()
        for o in dev["outputs"]:
            if o["label"] == selected:
                return o["index"]
        return None




class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        title = ctk.CTkLabel(
            self,
            text="ℹ️ O programie",
            font=("Roboto", 24, "bold")
        )
        title.pack(pady=(20, 10), anchor="w", padx=20)

        subtitle = ctk.CTkLabel(
            self,
            text="Easy IResponse\nAplikacja do pomiaru i generowania odpowiedzi impulsowych.\n"
                 "Autor: Ty 🙂",
            justify="left"
        )
        subtitle.pack(pady=(0, 20), anchor="w", padx=20)


# --------------------------------------------------
# GŁÓWNE OKNO APLIKACJI
# --------------------------------------------------
class EasyIResponseApp(ctk.CTk):

    def _safe_close(self):
        """Bezpieczne zamknięcie aplikacji – zatrzymuje timery i monitory."""

        # 1. zatrzymaj input metera
        try:
            if "settings" in self.pages:
                page = self.pages["settings"]
                if hasattr(page, "input_monitor") and page.input_monitor:
                    page.input_monitor.stop()
        except:
            pass

        # 2. zatrzymaj after() stron
        try:
            for page in self.pages.values():
                if hasattr(page, "after_id") and page.after_id is not None:
                    try:
                        page.after_cancel(page.after_id)
                    except:
                        pass
        except:
            pass

        # 3. zatrzymaj animację
        try:
            if hasattr(self, "anim_after_id"):
                self.after_cancel(self.anim_after_id)
        except:
            pass

        # 4. zamknij okno
        try:
            self.destroy()
        except:
            pass

    def __init__(self):
        super().__init__()
        self.anim_after_id = None

        # Okno
        self.title("Easy IResponse")
        self.geometry("1100x650")
        self.minsize(900, 550)

        # Grid główny
        self.grid_columnconfigure(0, weight=0)   # sidebar
        self.grid_columnconfigure(1, weight=1)   # content
        self.grid_rowconfigure(0, weight=1)

        # Sidebar i content
        self._create_sidebar()
        self._create_content_area()

        # Podstrony
        self.pages = {}
        self._create_pages()

        self.current_page = None
        self.nav_buttons = {}

        # Tworzymy przyciski nawigacji po tym, jak mamy pages
        self._create_nav_buttons()

        # Start na stronie pomiaru IR
        self.set_active_page("measurement", animate=False)
        self.protocol("WM_DELETE_WINDOW", self._safe_close)

    # ---------------- SIDEBAR ----------------
    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=230)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_rowconfigure(0, weight=0)   # tytuł
        self.sidebar.grid_rowconfigure(1, weight=0)   # odstęp
        self.sidebar.grid_rowconfigure(2, weight=0)
        self.sidebar.grid_rowconfigure(3, weight=0)
        self.sidebar.grid_rowconfigure(4, weight=1)   # rozpycha środek
        self.sidebar.grid_rowconfigure(5, weight=0)
        self.sidebar.grid_rowconfigure(6, weight=0)

        title_label = ctk.CTkLabel(
            self.sidebar,
            text="Easy IResponse",
            font=("Roboto", 22, "bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="IR measurement & synthesis",
            font=("Roboto", 11)
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

    def _create_nav_buttons(self):
        # Przyciski główne (środek)
        btn_measure = ctk.CTkButton(
            self.sidebar,
            text="🎤  Pomiar IR",
            anchor="w",
            command=lambda: self.set_active_page("measurement")
        )
        btn_measure.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        btn_generator = ctk.CTkButton(
            self.sidebar,
            text="🌊  Generator IR",
            anchor="w",
            command=lambda: self.set_active_page("generator")
        )
        btn_generator.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        # Przyciski dolne
        btn_settings = ctk.CTkButton(
            self.sidebar,
            text="⚙️  Ustawienia",
            anchor="w",
            command=lambda: self.set_active_page("settings")
        )
        btn_settings.grid(row=5, column=0, padx=15, pady=(10, 5), sticky="ew")

        btn_about = ctk.CTkButton(
            self.sidebar,
            text="ℹ️  O programie",
            anchor="w",
            command=lambda: self.set_active_page("about")
        )
        btn_about.grid(row=6, column=0, padx=15, pady=(0, 15), sticky="ew")

        # Zapisz przyciski do późniejszego podświetlania
        self.nav_buttons = {
            "measurement": btn_measure,
            "generator": btn_generator,
            "settings": btn_settings,
            "about": btn_about,
        }

    # ---------------- CONTENT ----------------
    def _create_content_area(self):
        self.content_container = ctk.CTkFrame(self)
        self.content_container.grid(row=0, column=1, sticky="nsew")
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

    def _create_pages(self):
        self.pages["measurement"] = MeasurementPage(self.content_container, self)
        self.pages["generator"] = GeneratorPage(self.content_container, self)
        self.pages["settings"] = SettingsPage(self.content_container, self)
        self.pages["about"] = AboutPage(self.content_container, self)

        # Ustaw wstępne pozycjonowanie (poza ekranem)
        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            page.place_forget()

    # ---------------- LOGIKA STRON + ANIMACJA ----------------
    def set_active_page(self, name: str, animate: bool = True):
        if name not in self.pages:
            return

        new_page = self.pages[name]

        if self.current_page is new_page:
            return

        # Podświetlenie przycisków w sidebarze
        self._highlight_nav_button(name)

        if self.current_page is None or not animate:
            # Pierwsze wyświetlenie – bez animacji
            if self.current_page is not None:
                self.current_page.place_forget()
            new_page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.current_page = new_page
            return

        # Animacja "slide z prawej"
        self._animate_slide(self.current_page, new_page)
        self.current_page = new_page

    def _highlight_nav_button(self, active_name: str):
        # Kolory pod przyciski (możesz dopasować pod swój motyw)
        active_color = "#d71920"
        inactive_color = "#1a1a1a"

        for name, btn in self.nav_buttons.items():
            if name == active_name:
                btn.configure(fg_color=active_color)
            else:
                btn.configure(fg_color=inactive_color)

    def _animate_slide(self, old_page: ctk.CTkFrame, new_page: ctk.CTkFrame):
        # Upewniamy się, że mamy aktualny rozmiar
        self.update_idletasks()
        width = self.content_container.winfo_width()
        if width <= 1:
            width = 800  # fallback

        # Ustaw nową stronę z prawej
        new_page.place(x=width, y=0, relwidth=1, relheight=1)

        steps = 20
        dx = width / steps
        duration_ms = 10  # czas między krokami

        def step(i):
            if i > steps:
                old_page.place_forget()
                new_page.place(relx=0, rely=0, relwidth=1, relheight=1)
                return

            x_new = width - dx * i
            x_old = -dx * i

            new_page.place(x=x_new, y=0, relwidth=1, relheight=1)
            old_page.place(x=x_old, y=0, relwidth=1, relheight=1)

            self.anim_after_id = self.after(duration_ms, step, i + 1)


        step(0)


# --------------------------------------------------
# START
# --------------------------------------------------
if __name__ == "__main__":
    app = EasyIResponseApp()
    try:
        app.mainloop()
    except:
        pass
