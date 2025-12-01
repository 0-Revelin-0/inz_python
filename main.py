from tkinter import filedialog
import customtkinter as ctk
import sounddevice as sd
import threading
import numpy as np
import os
import time
import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path
import soundfile as sf

#---------- Integracja między plikami ----------
from measurement_engine import measure_ir
from spl_calibration import PinkNoisePlayer, measure_input_level, InputLevelMonitor




# --- Wyłączenie wewnętrznych after() CustomTkinter ---

ctk.deactivate_automatic_dpi_awareness()

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
# =====================================================================
#  POMIAR ODPOWIEDZI IMPULSOWEJ — MeasurementPage
# =====================================================================

class MeasurementPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # ==============================================================
        # GŁÓWNY UKŁAD STRONY: LEWA KOLUMNA + PRAWA KOLUMNA
        # ==============================================================
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self, corner_radius=12)
        main_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # ==============================================================
        # LEWA KOLUMNA — USTAWIENIA POMIARU
        # ==============================================================
        left = ctk.CTkFrame(main_frame, corner_radius=12)
        left.grid(row=0, column=0, padx=15, pady=15, sticky="ns")

        # ---------- Kalibracja SPL ----------
        ctk.CTkLabel(left, text="Kalibracja SPL:", font=("Arial", 18, "bold")).pack(anchor="w", pady=(5, 2))
        self.spl_label = ctk.CTkLabel(left, text="Poziom niezmierzony.",
                                      font=("Arial", 14))
        self.spl_label.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(left, text="Kalibracja SPL", font=("Arial", 18, "bold")).pack(
            anchor="w", pady=(10, 5)
        )

        self.pinknoise_button = ctk.CTkButton(
            left,
            text="Start Pink Noise",
            command=self._toggle_pink_noise
        )
        self.pinknoise_button.pack(fill="x", pady=(5, 10))

        self.calib_status_label = ctk.CTkLabel(
            left,
            text="Brak pomiaru",
            text_color="#888888",
            anchor="center",
            justify="center",
        )
        self.calib_status_label.pack(fill="x", pady=(5, 5))

        # ---------- Parametry pomiaru ----------
        ctk.CTkLabel(left, text="Parametry pomiaru:", font=("Arial", 18, "bold")).pack(anchor="w", pady=(20, 10))

        self.sweep_length = self._make_param(left, "Długość sweepa [s]:", "5")
        self.start_freq = self._make_param(left, "Start freq [Hz]:", "20")
        self.end_freq = self._make_param(left, "End freq [Hz]:", "20000")
        self.ir_length = self._make_param(left, "Długość IR [s]:", "3")
        self.fade_time = self._make_param(left, "Fade [s]:", "0.05")

        # ---------- Folder zapisu IR ----------
        ctk.CTkLabel(left, text="Folder zapisu IR:", font=("Arial", 18, "bold")).pack(anchor="w", pady=(20, 5))

        folder_frame = ctk.CTkFrame(left)
        folder_frame.pack(fill="x", pady=5)

        self.output_dir_var = ctk.StringVar(value=str(Path.home()))
        self.folder_entry = ctk.CTkEntry(folder_frame, textvariable=self.output_dir_var)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(folder_frame, text="Wybierz folder...", width=130,
                      command=self._choose_folder).pack(side="right")

        # ---------- Start measurement ----------
        self.start_button = ctk.CTkButton(left, text="Start measurement", fg_color="#d71920",
                                          hover_color="#b01015", command=self._start_measurement)
        self.start_button.pack(fill="x", pady=(30, 5))

        # Pasek postępu pomiaru IR
        self.progress_bar = ctk.CTkProgressBar(left, height=12)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 15))

        # --- Dolny pasek statusu (footer) ---
        self.footer = ctk.CTkFrame(self, fg_color="transparent")
        self.footer.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(20, 10))

        self.status_label = ctk.CTkLabel(
            self.footer,
            text="",
            font=("Arial", 16),
            text_color="white",
            anchor="center",
            justify="center"
        )
        self.status_label.pack(fill="x", pady=(5, 5))

        # ==============================================================
        # PRAWA KOLUMNA — WYKRESY
        # ==============================================================
        plot_frame = ctk.CTkFrame(main_frame, corner_radius=12)
        plot_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)

        # ---------- Matplotlib figure: 2 subplots ----------
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor="#111111", tight_layout=True)

        # Impulse Response
        self.ax_ir = self.fig.add_subplot(2, 1, 1)
        self.ax_ir.set_facecolor("#111111")
        self.ax_ir.tick_params(colors="white")
        self.ax_ir.grid(True, color="#444444", alpha=0.3)
        self.ax_ir.set_title("Impulse Response", color="white")
        self.ax_ir.set_xlabel("Czas [s]", color="white")
        self.ax_ir.set_ylabel("Amplituda", color="white")

        # Magnitude Response
        self.ax_mag = self.fig.add_subplot(2, 1, 2)
        self.ax_mag.set_facecolor("#111111")
        self.ax_mag.tick_params(colors="white")
        self.ax_mag.grid(True, color="#444444", alpha=0.3)
        self.ax_mag.set_title("Magnitude Response", color="white")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
        self.ax_mag.set_ylabel("Poziom [dB]", color="white")

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self._clear_plots()

    # =====================================================================
    # FUNKCJE POMOCNICZE
    # =====================================================================
    def _toggle_pink_noise(self):
        # jeśli pink noise jest włączony → wyłącz
        if hasattr(self, "pinknoise_running") and self.pinknoise_running:
            self._stop_pink_noise()
            self.pinknoise_running = False
            self.pinknoise_button.configure(text="Start Pink Noise")
            return

        # jeśli pink noise jest wyłączony → włącz
        self._start_pink_noise()
        self.pinknoise_running = True
        self.pinknoise_button.configure(text="Stop Pink Noise")

    def _make_param(self, parent, label_text, default):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=4)

        ctk.CTkLabel(frame, text=label_text).pack(side="left")
        entry = ctk.CTkEntry(frame, width=80)
        entry.insert(0, default)
        entry.pack(side="right")
        return entry

    def _choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir_var.set(folder)

    def _clear_plots(self):
        self.ax_ir.cla()
        self.ax_mag.cla()

        # IR labels
        self.ax_ir.set_facecolor("#111111")
        self.ax_ir.grid(True, color="#444444", alpha=0.3)
        self.ax_ir.set_title("Impulse Response", color="white")
        self.ax_ir.set_xlabel("Czas [s]", color="white")
        self.ax_ir.set_ylabel("Amplituda", color="white")

        # MAG labels
        self.ax_mag.set_facecolor("#111111")
        self.ax_mag.grid(True, color="#444444", alpha=0.3)
        self.ax_mag.set_title("Magnitude Response", color="white")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
        self.ax_mag.set_ylabel("Poziom [dB]", color="white")

        self.canvas.draw()

    def _update_progress(self, value: float):
        """Ustawia pasek postępu 0.0–1.0"""
        try:
            self.progress_bar.set(value)
            self.progress_bar.update_idletasks()
        except:
            pass

    # =====================================================================
    # FUNKCJE LOGICZNE (PLACEHOLDERS — DZIAŁAJĄCE)
    # =====================================================================

    # def _calibrate_spl(self):
    #     self.spl_label.configure(text="(calibration running...)")
    #     self.status_label.configure(text="Kalibracja SPL... (placeholder)")
    #
    #     self.after(800, lambda: self.spl_label.configure(text="75 dB SPL zmierzone"))

    def _start_measurement(self):
        """Start pomiaru IR – wywołuje measurement_engine w osobnym wątku."""
        # Na wszelki wypadek zatrzymujemy kalibrację SPL (pink noise + monitor)
        try:
            self._stop_pink_noise()
        except Exception:
            pass
        """Start pomiaru IR – wywołuje measurement_engine w osobnym wątku."""
        # 1. Parsowanie parametrów z panelu po lewej
        try:
            sweep_len = float(self.sweep_length.get())
            start_f = float(self.start_freq.get())
            end_f = float(self.end_freq.get())
            ir_len = float(self.ir_length.get())
            fade = float(self.fade_time.get())
        except ValueError:
            show_error("Błędne parametry pomiaru.\nSprawdź, czy wszystkie pola są liczbami.")
            return

        # sanity check
        if end_f <= start_f:
            show_error("End freq musi być większe niż Start freq.")
            return

        # 2. Folder wyjściowy
        output_dir = self.output_dir_var.get()
        if not os.path.isdir(output_dir):
            show_error("Wybrany folder zapisu IR nie istnieje.")
            return

        # 3. Konfiguracja audio z SettingsPage
        audio_cfg = self.controller.get_measurement_audio_config()
        if audio_cfg is None:
            show_error("Najpierw skonfiguruj poprawnie urządzenia audio w zakładce 'Ustawienia'.")
            return

        params = {
            "sweep_length": sweep_len,
            "start_freq": start_f,
            "end_freq": end_f,
            "ir_length": ir_len,
            "fade_time": fade,
        }

        # 4. UI: czyścimy wykresy, blokujemy przycisk
        self._clear_plots()
        self.status_label.configure(text="Trwa pomiar IR...")
        self.start_button.configure(state="disabled")

        self._update_progress(0.0)

        # 5. Worker w osobnym wątku (żeby nie blokować GUI)
        def worker():
            try:
                self.after(0, lambda: self._update_progress(0.1))
                ir, freqs, mag_db = measure_ir(params, audio_cfg)
                self.after(0, lambda: self._update_progress(0.7))
                fs = audio_cfg["sample_rate"]

                # Zapis IR do WAV
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"IR_{int(start_f)}-{int(end_f)}Hz_{timestamp}.wav"
                filepath = os.path.join(output_dir, filename)

                sf.write(filepath, ir, fs)
                self.after(0, lambda: self._update_progress(0.9))

            except Exception as e:
                # Aktualizacja GUI MUSI być przez self.after(...)
                def on_error():
                    show_error(f"Błąd podczas pomiaru:\n\n{e}")
                    self.status_label.configure(text="Błąd pomiaru.")
                    self.start_button.configure(state="normal")

                self.after(0, on_error)
                return

            # Funkcja do aktualizacji wykresów w wątku GUI
            def update_plots():
                # --- IR ---
                self.ax_ir.cla()
                self.ax_ir.set_facecolor("#111111")
                self.ax_ir.grid(True, color="#444444", alpha=0.3)
                self.ax_ir.set_title("Impulse Response", color="white")
                self.ax_ir.set_xlabel("Czas [s]", color="white")
                self.ax_ir.set_ylabel("Amplituda", color="white")

                t = np.arange(len(ir)) / fs
                self.ax_ir.plot(t, ir, linewidth=0.9)

                # --- Magnitude ---
                self.ax_mag.cla()
                self.ax_mag.set_facecolor("#111111")
                self.ax_mag.grid(True, color="#444444", alpha=0.3)
                self.ax_mag.set_title("Magnitude Response", color="white")
                self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
                self.ax_mag.set_ylabel("Poziom [dB]", color="white")

                self.ax_mag.semilogx(freqs, mag_db, linewidth=0.9)

                self.canvas.draw()

                self.status_label.configure(
                    text=f"Pomiar zakończony. Zapisano plik:\n{filename}"
                )
                self.start_button.configure(state="normal")

            self.after(0, lambda: self._update_progress(1.0))
            self.after(0, update_plots)


        threading.Thread(target=worker, daemon=True).start()

    # def _start_pink_noise(self):
    #     audio_cfg = self.controller.get_measurement_audio_config()
    #     if audio_cfg is None:
    #         show_error("Skonfiguruj urządzenia audio w Ustawieniach.")
    #         return
    #
    #     if not hasattr(self, "pink_player"):
    #         self.pink_player = PinkNoisePlayer()
    #
    #     self.pink_player.start(audio_cfg)
    #     self.calib_status_label.configure(text="Kalibracja…", text_color="#cccccc")
    #
    #     self._poll_input_level()

    def _stop_pink_noise(self):
        if hasattr(self, "pink_player"):
            self.pink_player.stop()

        if hasattr(self, "input_level_monitor"):
            self.input_level_monitor.stop()

        self.calib_status_label.configure(text="Zatrzymano szum", text_color="#888888")

    def _start_pink_noise(self):
        settings_page = self.controller.pages["settings"]
        if hasattr(settings_page, "stop_input_monitor"):
            settings_page.stop_input_monitor()

        audio_cfg = self.controller.get_measurement_audio_config()
        if audio_cfg is None:
            show_error("Skonfiguruj urządzenia audio w Ustawieniach.")
            return

        # Pink Noise
        if not hasattr(self, "pink_player"):
            self.pink_player = PinkNoisePlayer()

        self.pink_player.start(audio_cfg)
        self.calib_status_label.configure(text="Kalibracja…", text_color="#cccccc")

        # --- START CIĄGŁEGO MONITORA WEJŚCIA ---

        def update_level(rms_db, peak_db):
            # ta funkcja jest wywoływana w wątku audio → opakowujemy ją w after()
            def do_update():
                if peak_db > -1:
                    txt = f"Peak {peak_db:.1f} dBFS — ZA GŁOŚNO!"
                    color = "#ff4444"
                elif rms_db < -40:
                    txt = f"RMS {rms_db:.1f} dBFS — Za cicho"
                    color = "#ffaa00"
                elif rms_db > -8:
                    txt = f"RMS {rms_db:.1f} dBFS — Może być za głośno"
                    color = "#ff8800"
                else:
                    txt = f"RMS {rms_db:.1f} dBFS — OK"
                    color = "#44cc44"

                self.calib_status_label.configure(text=txt, text_color=color)

            # zlecenie aktualizacji w głównym wątku Tkintera
            self.after(0, do_update)

        self.input_level_monitor = InputLevelMonitor(audio_cfg, update_level)
        self.input_level_monitor.start()

    # def _poll_input_level(self):
    #     if not hasattr(self, "pink_player"):
    #         return
    #     if not self.pink_player.running:
    #         return
    #
    #     audio_cfg = self.controller.get_measurement_audio_config()
    #     result = measure_input_level(audio_cfg, duration=0.2)
    #
    #     rms_db = result["rms_db"]
    #     peak_db = result["peak_db"]
    #
    #     if peak_db > -1:
    #         txt = f"Peak {peak_db:.1f} dBFS — ZA GŁOŚNO!"
    #         color = "#ff4444"
    #     elif rms_db < -40:
    #         txt = f"RMS {rms_db:.1f} dBFS — Za cicho"
    #         color = "#ffaa00"
    #     elif rms_db > -8:
    #         txt = f"RMS {rms_db:.1f} dBFS — Może być za głośno"
    #         color = "#ff8800"
    #     else:
    #         txt = f"RMS {rms_db:.1f} dBFS — OK"
    #         color = "#44cc44"
    #
    #     self.calib_status_label.configure(text=txt, text_color=color)
    #     self.after(300, self._poll_input_level)



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

            # Aktualizacja paska w wątku GUI
            def do_update(lvl=level):
                self.progress_bar.set(lvl)

            self.progress_bar.after(0, do_update)

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

    def stop_input_monitor(self):
        """Bezpieczne zatrzymanie miernika wejścia."""
        if hasattr(self, "input_monitor") and self.input_monitor is not None:
            try:
                self.input_monitor.stop()
            except:
                pass

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
        """Bezpieczne zamknięcie aplikacji – zatrzymuje monitory i kończy mainloop bez destroy()."""

        # 1. zatrzymaj input metera (jak masz)
        try:
            if hasattr(self, "pages") and "settings" in self.pages:
                page = self.pages["settings"]
                if hasattr(page, "input_monitor") and page.input_monitor:
                    page.input_monitor.stop()
        except Exception:
            pass

        # 2. anuluj after() które SAM utworzyłeś (np. w MeasurementPage czy innych stronach)
        try:
            if hasattr(self, "pages"):
                for page in self.pages.values():
                    if hasattr(page, "after_id") and page.after_id is not None:
                        try:
                            page.after_cancel(page.after_id)
                        except Exception:
                            pass
        except Exception:
            pass

        # 3. ZAMIAST destroy:
        #    - schowaj okno
        #    - zatrzymaj pętlę zdarzeń Tkintera
        self.withdraw()  # ukryj okno
        self.quit()  # zatrzymaj mainloop / interpreter Tcl

    def __init__(self):
        super().__init__()
        self.anim_after_id = None

        # Okno
        self.title("Easy IResponse")
        self.geometry("1200x800")
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

    def get_measurement_audio_config(self):
        """
        Zwraca słownik z ustawieniami audio dla pomiaru IR
        na podstawie zakładki 'Ustawienia pomiaru'.
        """
        settings_page: SettingsPage = self.pages["settings"]

        in_idx = settings_page.get_selected_input_index()
        out_idx = settings_page.get_selected_output_index()

        if in_idx is None or out_idx is None:
            return None

        # sample rate
        try:
            sr = int(settings_page.sample_rate_combo.get())
        except ValueError:
            sr = 48000

        # buffer size
        try:
            buf = int(settings_page.buffer_size_combo.get())
        except ValueError:
            buf = 256

        return {
            "input_device": in_idx,
            "output_device": out_idx,
            "sample_rate": sr,
            "buffer_size": buf,
        }
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
