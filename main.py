from tkinter import filedialog
import customtkinter as ctk
import sounddevice as sd
import threading
import numpy as np
import os
import time
import sys
import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path
import soundfile as sf
import webbrowser
from matplotlib.ticker import LogLocator, StrMethodFormatter
from matplotlib.ticker import ScalarFormatter




#---------- Integracja między plikami ----------
from measurement_engine import measure_ir
from spl_calibration import PinkNoisePlayer, measure_input_level, InputLevelMonitor

#---------- Zaciąganie theme w .exe i .py ----------
def resource_path(relative_path: str) -> str:
    """Znajduje pliki także w .exe (PyInstaller)."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# --- Wyłączenie wewnętrznych after() CustomTkinter ---

ctk.deactivate_automatic_dpi_awareness()

# --------------------------------------------------
# KONFIGURACJA GLOBALNA
# --------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(resource_path("dark-red.json"))




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

        # Dane z ostatniego pomiaru (mono / stereo)
        self.last_ir = None  # do prostego użytku (mono lub aktualnie wybrany kanał)
        self.last_freqs = None
        self.last_mag = None
        self.last_fs = None

        # Dane per-kanał dla trybu stereo
        self.last_ir_L = None
        self.last_ir_R = None
        self.last_mag_L = None
        self.last_mag_R = None
        self.last_is_stereo = False
        self.current_channel = "L"  # 'L' albo 'R' na wykresie

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
        # ctk.CTkLabel(left, text="Kalibracja SPL:", font=("Arial", 18, "bold")).pack(anchor="w", pady=(5, 2))
        # self.spl_label = ctk.CTkLabel(left, text="Poziom niezmierzony.",
        #                               font=("Arial", 14))
        # self.spl_label.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(left, text="Kalibracja SPL", font=("Arial", 18, "bold")).pack(
            anchor="n", pady=(10, 5)
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
        self.start_freq = self._make_param(left, "Start freq [Hz]:", "100")
        self.end_freq = self._make_param(left, "End freq [Hz]:", "10000")
        self.ir_length = self._make_param(left, "Długość IR [s]:", "8")
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
        self.progress_bar.pack(fill="x", pady=(40, 20))


        self.status_label = ctk.CTkLabel(
            left,
            text="Status pomiaru",
            font=("Arial", 14),
            text_color="white",
            anchor="n",
            justify="center"
        )
        self.status_label.pack(fill="x", pady=(40, 20))

        # # --- Dolny pasek statusu (footer) ---
        # self.footer = ctk.CTkFrame(self, fg_color="transparent")
        # self.footer.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(20, 10))
        #
        # self.status_label = ctk.CTkLabel(
        #     self.footer,
        #     text="",
        #     font=("Arial", 16),
        #     text_color="white",
        #     anchor="center",
        #     justify="center"
        # )
        # self.status_label.pack(fill="x", pady=(5, 5))

        # ==============================================================
        # PRAWA KOLUMNA — WYKRESY
        # ==============================================================
        plot_frame = ctk.CTkFrame(main_frame, corner_radius=12)
        plot_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)

        # Panel wyboru kanału do podglądu (L / R)
        channel_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        channel_frame.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkLabel(
            channel_frame,
            text="Kanał do podglądu:"
        ).pack(side="left", padx=(0, 10))

        self.channel_selector = ctk.CTkSegmentedButton(
            channel_frame,
            values=["L", "R"],
            command=self._on_channel_change
        )
        self.channel_selector.pack(side="left")
        self.channel_selector.set("L")

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

    def update_plots(self):
        """Aktualizuje wykresy bazując na ostatnim pomiarze (mono lub stereo)."""

        # Brak danych – nic nie rysujemy
        if (not self.last_is_stereo and self.last_ir is None) and self.last_ir_L is None:
            return

        fs = self.last_fs
        freqs = self.last_freqs
        if fs is None or freqs is None:
            return

        # Wybór kanału do rysowania
        if self.last_is_stereo:
            if self.current_channel == "R" and self.last_ir_R is not None:
                ir_full = self.last_ir_R
                mag_db = self.last_mag_R
                channel_label = "Right"
            else:
                ir_full = self.last_ir_L
                mag_db = self.last_mag_L
                channel_label = "Left"
        else:
            ir_full = self.last_ir
            mag_db = self.last_mag
            channel_label = "Mono"

        if ir_full is None or mag_db is None:
            return

        # --- IR ---
        self.ax_ir.cla()
        self.ax_ir.set_facecolor("#111111")
        self.ax_ir.grid(True, color="#444444", alpha=0.3)
        self.ax_ir.set_title(f"Impulse Response ({channel_label})", color="white")
        self.ax_ir.set_xlabel("Czas [s]", color="white")
        self.ax_ir.set_ylabel("Amplituda", color="white")

        # 1) Szukamy największego piku IR
        peak_idx = np.argmax(np.abs(ir_full))

        # 2) Ile przed i po piku chcemy pokazać na wykresie
        pre_ms = 50  # 50 ms przed pikiem
        post_ms = self.controller.get_ir_window_ms()  # dynamiczne z ustawień

        pre_samples = int((pre_ms / 1000.0) * fs)
        post_samples = int((post_ms / 1000.0) * fs)

        start = max(0, peak_idx - pre_samples)
        end = min(len(ir_full), peak_idx + post_samples)

        ir_segment = ir_full[start:end]

        # 3) Downsampling TYLKO do rysowania, żeby GUI było płynne
        MAX_PLOT_POINTS = 20000
        if len(ir_segment) > MAX_PLOT_POINTS:
            factor = len(ir_segment) // MAX_PLOT_POINTS
            ir_plot = ir_segment[::factor]
            t = np.arange(len(ir_plot)) * (factor / fs)
        else:
            ir_plot = ir_segment
            t = np.arange(len(ir_plot)) / fs

        # 4) Rysowanie IR
        self.ax_ir.plot(t, ir_plot, linewidth=0.9, color="#4fc3f7")

        # --- MAGNITUDE ---
        self.ax_mag.cla()
        self.ax_mag.set_facecolor("#111111")
        self.ax_mag.grid(True, color="#444444", alpha=0.3)
        self.ax_mag.set_title(f"Magnitude Response ({channel_label})", color="white")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
        self.ax_mag.set_ylabel("Poziom [dB]", color="white")

        smoothing = self.controller.get_smoothing_fraction()

        try:
            from measurement_engine import smooth_mag_response
            if smoothing is None:
                mag_plot = mag_db
            else:
                mag_plot = smooth_mag_response(freqs, mag_db, fraction=smoothing)
        except Exception:
            mag_plot = mag_db

        self.ax_mag.semilogx(freqs, mag_plot, linewidth=1.5, color="#0096FF")

        self.ax_mag.xaxis.set_major_locator(LogLocator(base=10.0))
        self.ax_mag.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))

        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        self.ax_mag.xaxis.set_major_formatter(formatter)

        try:
            start_f = float(self.start_freq.get())
            end_f = float(self.end_freq.get())
            self.ax_mag.set_xlim(start_f * 0.9, end_f * 1.1)
        except Exception:
            pass

        self.canvas.draw()

    def _on_channel_change(self, value: str):
        """Obsługa przełączania kanału L/R na wykresach."""
        self.current_channel = value
        # Samo przerysowanie – update_plots wybierze odpowiedni kanał
        self.update_plots()


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
        # --- Clear IR plot ---
        self.ax_ir.cla()
        self.ax_ir.set_facecolor("#111111")
        self.ax_ir.grid(True, color="#444444", alpha=0.3)
        self.ax_ir.set_title("Impulse Response", color="white")
        self.ax_ir.set_xlabel("Czas [s]", color="white")
        self.ax_ir.set_ylabel("Amplituda", color="white")

        # --- Clear Magnitude plot ---
        self.ax_mag.cla()
        self.ax_mag.set_facecolor("#111111")
        self.ax_mag.grid(True, color="#444444", alpha=0.3)
        self.ax_mag.set_title("Magnitude Response", color="white")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
        self.ax_mag.set_ylabel("Poziom [dB]", color="white")

        # Ustawiamy skalę logarytmiczną JUŻ NA STARCIE
        self.ax_mag.set_xscale("log")

        # TICKI LOGARYTMICZNE

        # Major ticks: 100 Hz, 1000 Hz, 10000 Hz
        self.ax_mag.xaxis.set_major_locator(LogLocator(base=10.0))

        # Minor ticks w log10 (2..9: 200,300,...,900)
        self.ax_mag.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))

        # Formatter liczb (np. '100', '1000', '10000')
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        self.ax_mag.xaxis.set_major_formatter(formatter)

        # Domyślny zakres
        self.ax_mag.set_xlim(100, 10000)

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

        # Zabezpieczenie: IR nie krótsza niż sweep
        if ir_len < sweep_len:
            ir_len = sweep_len
            # poprawiamy wartość w polu tekstowym, żeby user widział co się stało
            self.ir_length.delete(0, "end")
            self.ir_length.insert(0, str(sweep_len))
            self.status_label.configure(
                text="Długość IR była krótsza niż sweep.\nUstawiono IR = długość sweepa."
            )


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
                self.after(0, lambda: self._update_progress(0.2))

                ir, freqs, mag_db, recorded = measure_ir(params, audio_cfg)
                fs = audio_cfg["sample_rate"]

                # Rozpoznanie trybu mono / stereo na podstawie kształtu IR
                if hasattr(ir, "ndim") and ir.ndim == 2 and ir.shape[1] >= 2:
                    # STEREO
                    self.last_is_stereo = True
                    self.current_channel = "L"

                    self.last_ir_L = ir[:, 0]
                    self.last_ir_R = ir[:, 1]
                    self.last_ir = self.last_ir_L

                    # mag_db: (F, 2)
                    self.last_mag_L = mag_db[:, 0]
                    self.last_mag_R = mag_db[:, 1]
                    self.last_mag = self.last_mag_L
                else:
                    # MONO
                    self.last_is_stereo = False

                    if ir.ndim > 1:
                        ir_mono = ir[:, 0]
                    else:
                        ir_mono = ir

                    self.last_ir = ir_mono
                    self.last_ir_L = ir_mono
                    self.last_ir_R = None

                    if mag_db.ndim == 1:
                        mag_mono = mag_db
                    else:
                        mag_mono = mag_db[..., 0]

                    self.last_mag = mag_mono
                    self.last_mag_L = mag_mono
                    self.last_mag_R = None

                self.last_freqs = freqs
                self.last_fs = fs

                self.after(0, lambda: self._update_progress(0.4))

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir_local = output_dir  # zamrożenie w closure

                # --- ZAPIS SUROWEGO NAGRANIA SWEEPA ---
                if self.last_is_stereo and hasattr(recorded, "ndim") and recorded.ndim == 2 and recorded.shape[1] >= 2:
                    rec_L = recorded[:, 0]
                    rec_R = recorded[:, 1]

                    rec_filename_L = f"RECORDED_L_{timestamp}.wav"
                    rec_filename_R = f"RECORDED_R_{timestamp}.wav"

                    sf.write(os.path.join(output_dir_local, rec_filename_L), rec_L, fs)
                    sf.write(os.path.join(output_dir_local, rec_filename_R), rec_R, fs)
                else:
                    rec_filename = f"RECORDED_{timestamp}.wav"
                    sf.write(os.path.join(output_dir_local, rec_filename), recorded, fs)

                self.after(0, lambda: self._update_progress(0.65))

                # --- ZAPIS IR ---
                if self.last_is_stereo:
                    ir_filename_L = f"IR_L_{int(start_f)}-{int(end_f)}Hz_{timestamp}.wav"
                    ir_filename_R = f"IR_R_{int(start_f)}-{int(end_f)}Hz_{timestamp}.wav"

                    sf.write(os.path.join(output_dir_local, ir_filename_L), self.last_ir_L, fs)
                    sf.write(os.path.join(output_dir_local, ir_filename_R), self.last_ir_R, fs)

                    saved_info = f"{ir_filename_L}\n{ir_filename_R}"
                else:
                    ir_filename = f"IR_{int(start_f)}-{int(end_f)}Hz_{timestamp}.wav"
                    sf.write(os.path.join(output_dir_local, ir_filename), self.last_ir, fs)
                    saved_info = ir_filename

                self.after(0, lambda: self._update_progress(0.8))

            except Exception as e:
                err_msg = f"Błąd podczas pomiaru:\n\n{e}"

                def on_error(msg=err_msg):
                    show_error(msg)
                    self.status_label.configure(text="Błąd pomiaru.")
                    self.start_button.configure(state="normal")

                self.after(0, on_error)
                return

            # Finalizacja – przerysowanie wykresów i komunikat
            self.after(0, lambda: self._update_progress(1.0))
            self.after(0, self.update_plots)
            self.after(
                0,
                lambda: self.status_label.configure(
                    text=f"Pomiar zakończony. Zapisano plik(i):\n{saved_info}"
                ),
            )
            self.after(0, lambda: self.start_button.configure(state="normal"))

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
        tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
        sd.play(tone, samplerate=samplerate, device=output_device_index)
        sd.wait()

    threading.Thread(target=_worker, daemon=True).start()



class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):

        self._ir_window_after_id = None

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
        measure_tab.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(measure_tab, corner_radius=12)
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)

        # ---------------- WEJŚCIA / WYJŚCIA ----------------
        # Dwa osobne inputy: Left / Right
        ctk.CTkLabel(frame, text="Input (Left):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.input_L_combo = ctk.CTkComboBox(frame, values=[])
        self.input_L_combo.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(frame, text="Input (Right):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.input_R_combo = ctk.CTkComboBox(frame, values=[])
        self.input_R_combo.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

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

        # Smoothing
        ctk.CTkLabel(frame, text="Smoothing (Magnitude):").grid(
            row=5, column=0, padx=10, pady=10, sticky="w"
        )
        self.smoothing_combo = ctk.CTkComboBox(
            frame,
            values=["Raw", "1/24 octave", "1/12 octave", "1/6 octave", "1/3 octave"],
            command=self._on_smoothing_change
        )
        self.smoothing_combo.set("1/6 octave")
        self.smoothing_combo.grid(row=5, column=1, padx=10, pady=10, sticky="ew")

        # IR window after peak (ms)
        ctk.CTkLabel(frame, text="IR window after peak [ms]:").grid(
            row=6, column=0, padx=10, pady=10, sticky="w"
        )
        self.ir_window_entry = ctk.CTkEntry(frame, width=120)
        self.ir_window_entry.grid(row=6, column=1, padx=10, pady=10, sticky="ew")
        self.ir_window_entry.insert(0, "500")
        self.ir_window_entry.bind("<KeyRelease>", self._on_ir_window_change)

        # Measurement mode (mono / stereo)
        ctk.CTkLabel(frame, text="Measurement mode:").grid(
            row=7, column=0, padx=10, pady=10, sticky="w"
        )
        self.measure_mode_combo = ctk.CTkComboBox(
            frame,
            values=["Mono", "Stereo"]
        )
        self.measure_mode_combo.set("Mono")
        self.measure_mode_combo.grid(row=7, column=1, padx=10, pady=10, sticky="ew")

        # Input meter
        ctk.CTkLabel(frame, text="Input level:").grid(row=8, column=0, padx=10, pady=10, sticky="w")
        self.meter_bar = ctk.CTkProgressBar(frame, width=220)
        self.meter_bar.grid(row=8, column=1, padx=10, pady=10, sticky="w")
        self.meter_bar.set(0)

        self.meter_btn = ctk.CTkButton(frame, text="Start input meter", command=self._toggle_meter)
        self.meter_btn.grid(row=9, column=1, padx=10, pady=10, sticky="w")

        # Test tone
        self.test_btn = ctk.CTkButton(frame, text="Test tone (1 kHz)", command=self._play_test)
        self.test_btn.grid(row=10, column=1, padx=10, pady=15, sticky="w")

        # Załaduj listę wejść/wyjść
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
        """
        Buduje płaskie listy:
        - self.input_entries:  [{label, index}]
        - self.output_entries: [{label, index}]
        gdzie label = 'Input – xxx (Urządzenie)' / 'Output – yyy (Urządzenie)'.
        """
        all_dev = sd.query_devices()

        junk = ["mapper", "mapowanie", "default", "podstawowy", "smart sound", "microsoft"]

        self.input_entries = []
        self.output_entries = []

        for idx, dev in enumerate(all_dev):
            name = dev["name"]

            # odfiltrowujemy śmieciowe urządzenia
            if any(j in name.lower() for j in junk):
                continue

            # Rozdzielamy: "Głośnik PC (Realtek HD Audio ...)" → label + base
            if "(" in name and ")" in name:
                label = name[:name.rfind("(")].strip()
                base = name[name.rfind("(") + 1:name.rfind(")")].strip()
            else:
                label = name
                base = name

            # Wejścia
            if dev["max_input_channels"] > 0:
                disp = f"Input – {label} ({base})"
                self.input_entries.append({"label": disp, "index": idx})

            # Wyjścia
            if dev["max_output_channels"] > 0:
                disp = f"Output – {label} ({base})"
                self.output_entries.append({"label": disp, "index": idx})

        # Ustawiamy wartości w comboboxach
        input_values = [e["label"] for e in self.input_entries] or ["Brak wejścia"]
        output_values = [e["label"] for e in self.output_entries] or ["Brak wyjścia"]

        # Ustawiamy wartości w comboboxach
        input_values = [e["label"] for e in self.input_entries] or ["Brak wejścia"]
        output_values = [e["label"] for e in self.output_entries] or ["Brak wyjścia"]

        # Oba inputy (L/R) korzystają z tej samej listy
        self.input_L_combo.configure(values=input_values)
        self.input_R_combo.configure(values=input_values)
        self.output_combo.configure(values=output_values)

        # Domyślny wybór
        self.input_L_combo.set(input_values[0])
        self.input_R_combo.set(input_values[0])
        self.output_combo.set(output_values[0])


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

    def _get_input_index_from_label(self, label: str):
        """Pomocniczo: zamiana etykiety z comboboxa na index urządzenia."""
        if not hasattr(self, "input_entries"):
            return None
        for entry in self.input_entries:
            if entry["label"] == label:
                return entry["index"]
        return None

    def get_selected_input_L_index(self):
        """Index urządzenia wejściowego wybranego dla kanału Left."""
        selected = self.input_L_combo.get()
        return self._get_input_index_from_label(selected)

    def get_selected_input_R_index(self):
        """Index urządzenia wejściowego wybranego dla kanału Right."""
        selected = self.input_R_combo.get()
        return self._get_input_index_from_label(selected)

    def get_selected_input_index(self):
        """
        Zachowanie wsteczne – używane np. przez input meter.
        Traktujemy Left jako "główne" wejście.
        """
        return self.get_selected_input_L_index()

    def get_selected_output_index(self):
        """Zwraca index urządzenia wyjściowego wybranego w output_combo."""
        if not hasattr(self, "output_entries"):
            return None

        selected = self.output_combo.get()
        for entry in self.output_entries:
            if entry["label"] == selected:
                return entry["index"]
        return None

    def get_measurement_mode(self):
        """Zwraca tryb pomiaru: 'Mono' lub 'Stereo'."""
        try:
            return self.measure_mode_combo.get()
        except AttributeError:
            # Na wszelki wypadek, gdyby combo nie istniało
            return "Mono"


    def _on_smoothing_change(self, value):
        # Dynamiczne przerysowanie wykresu Magnitude
        try:
            measurement_page = self.controller.pages["measurement"]
            measurement_page.update_plots()
        except:
            pass

    def get_smoothing_fraction(self):
        v = self.smoothing_combo.get()

        if v == "Raw":
            return None
        if "1/24" in v:
            return 24
        if "1/12" in v:
            return 12
        if "1/6" in v:
            return 6
        if "1/3" in v:
            return 3

        return None

    def _on_ir_window_change(self, event=None):
        if hasattr(self, "_ir_window_after_id") and self._ir_window_after_id is not None:
            try:
                self.after_cancel(self._ir_window_after_id)
            except:
                pass

        # start debounce (300 ms)
        self._ir_window_after_id = self.after(300, self._apply_ir_window_change)

    def _apply_ir_window_change(self):
        try:
            val = int(self.ir_window_entry.get())
            if val <= 0:
                return
        except ValueError:
            return

        try:
            measurement_page = self.controller.pages["measurement"]
            measurement_page.update_plots()
        except Exception:
            pass
        if self.ir_window_entry.get().strip() == "":
            return

    def get_ir_window_ms(self):
        """Zwraca okno IR po piku w ms (z pola w ustawieniach)."""
        try:
            val = int(self.ir_window_entry.get())
            if val <= 0:
                return 500
            return val
        except Exception:
            return 500



class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # ------------------ Tytuł strony ------------------
        title = ctk.CTkLabel(
            self,
            text="ℹ️  O programie",
            font=("Roboto", 28, "bold")
        )
        title.pack(pady=(20, 5), anchor="w", padx=20)

        # ------------------ Wersja aplikacji ------------------
        version_label = ctk.CTkLabel(
            self,
            text="Wersja aplikacji: v1.1",
            font=("Roboto", 15),
            text_color="#cccccc"
        )
        version_label.pack(anchor="w", padx=20, pady=(0, 3))

        # ------------------ Klikalny link GitHub ------------------
        def open_github():
            webbrowser.open("https://github.com/0-Revelin-0/inz_python.git")

        github_link = ctk.CTkLabel(
            self,
            text="Repozytorium GitHub (kliknij, aby otworzyć)",
            font=("Roboto", 15, "underline"),
            text_color="#4da6ff",
            cursor="hand2"
        )
        github_link.bind("<Button-1>", lambda e: open_github())
        github_link.pack(anchor="w", padx=20, pady=(0, 15))

        # ------------------ TABVIEW ------------------
        tabs = ctk.CTkTabview(self, width=920, height=560)
        tabs.pack(fill="both", expand=True, padx=20, pady=20)

        # Tworzenie tabów
        tabs.add("Opis programu")
        tabs.add("Funkcjonalności")
        tabs.add("Instrukcja pomiaru")
        tabs.add("Instrukcja generowania IR")      # NOWY TAB
        tabs.add("Instrukcja splotu IR z audio")   # NOWY TAB
        tabs.add("O autorze")
        tabs.add("Informacje techniczne")

        # ------------------ Helper do sekcji ------------------
        def add_section(tab, title_text, body_text):
            frame = ctk.CTkFrame(tab, fg_color="transparent")
            frame.pack(fill="both", expand=True, padx=20, pady=15)

            # Nagłówek sekcji
            ctk.CTkLabel(
                frame,
                text=title_text,
                font=("Roboto", 20, "bold")
            ).pack(anchor="w", pady=(0, 10))

            # Tekst sekcji
            ctk.CTkLabel(
                frame,
                text=body_text,
                font=("Roboto", 16),
                justify="left",
                wraplength=850
            ).pack(anchor="w", pady=5)

        # ======================================================
        # 1. OPIS PROGRAMU
        # ======================================================

        opis = (
            "Easy IResponse to zaawansowana aplikacja służąca do pomiaru oraz syntezy akustycznej "
            "odpowiedzi impulsowej pomieszczeń z zastosowaniem metody Exponential Sine Sweep (ESS). "
            "Program integruje generację sygnału testowego, jego odtworzenie, rejestrację odpowiedzi "
            "toru elektroakustycznego oraz dekonwolucję widmową w celu uzyskania czystej odpowiedzi "
            "liniowej. Aplikacja służy również do analizy IR, generowania syntetycznych IR, "
            "a także — w rozszerzeniach — do wykonywania splotu IR z sygnałem audio w procesach "
            "auralizacji wnętrz."
        )
        add_section(tabs.tab("Opis programu"), "Opis programu", opis)

        # ======================================================
        # 2. FUNKCJONALNOŚCI
        # ======================================================

        funkcje = (
            "• Pomiar odpowiedzi impulsowej metodą ESS (Exponential Sine Sweep)\n"
            "• Obsługa pomiaru MONO oraz STEREO\n"
            "• Stereo z dwoma niezależnymi wejściami (Input L / Input R)\n"
            "• Możliwość pracy z dwoma różnymi urządzeniami wejściowymi jednocześnie\n"
            "• Wspólna normalizacja IR dla kanałów L i R – niezbędna przy pomiarach HRTF\n"
            "• Automatyczna dekonwolucja i rekonstrukcja IR\n"
            "• Analiza IR w dziedzinie czasu i częstotliwości\n"
            "• Wykresy z wyborem kanału (L / R) i dynamicznym oknem IR\n"
            "• Zapis surowego nagrania oraz IR osobno dla kanałów L i R\n"
            "• Kalibracja SPL z użyciem różowego szumu\n"
            "• Konfigurowalny smoothing (Raw, 1/24, 1/12, 1/6, 1/3 okt.)\n"
            "• Integracja z urządzeniami audio w czasie rzeczywistym\n"
            "• Moduły rozszerzeń: generowanie IR, splot IR z audio"
        )

        add_section(tabs.tab("Funkcjonalności"), "Funkcjonalności", funkcje)

        # ======================================================
        # 3. INSTRUKCJA POMIARU
        # ======================================================

        instrukcja = (
            "1. Wejdź w zakładkę „Ustawienia” i wybierz urządzenia audio.\n"
            "   • Input (Left) – lewy kanał pomiarowy\n"
            "   • Input (Right) – prawy kanał pomiarowy\n"
            "   • Możesz wybrać jedno urządzenie 2-kanałowe lub dwa różne urządzenia wejściowe\n\n"
            "2. Wybierz tryb pomiaru:\n"
            "   • MONO – używany jest tylko kanał Left\n"
            "   • STEREO – program jednocześnie nagrywa Input L i Input R\n"
            "     (idealne do pomiarów HRTF lub pomiarów dwukanałowych)\n\n"
            "3. Ustaw parametry sweepa, sample rate i buffer size.\n"
            "4. Wykonaj opcjonalną kalibrację SPL.\n"
            "5. Naciśnij „Start measurement”.\n\n"
            "Podczas pomiaru:\n"
            "• Program generuje sweep\n"
            "• Odtwarza sygnał na wyjściu\n"
            "• Nagrywa kanał L i R równolegle\n"
            "• Wykonuje dekonwolucję\n"
            "• Normalizuje oba kanały tym samym współczynnikiem\n\n"
            "Zapisane pliki:\n"
            "• RECORDED_L_*.wav – nagranie surowe (lewy kanał)\n"
            "• RECORDED_R_*.wav – nagranie surowe (prawy kanał)\n"
            "• IR_L_*.wav – odpowiedź impulsowa lewego kanału\n"
            "• IR_R_*.wav – odpowiedź impulsowa prawego kanału\n\n"
            "Na wykresach możesz przełączać kanał (L / R) do podglądu IR i charakterystyki."
        )

        add_section(tabs.tab("Instrukcja pomiaru"), "Instrukcja pomiaru IR", instrukcja)

        # ======================================================
        # 4. PUSTY TAB – INSTRUKCJA GENEROWANIA IR
        # ======================================================

        add_section(
            tabs.tab("Instrukcja generowania IR"),
            "Instrukcja generowania IR",
            "Ta sekcja zostanie uzupełniona."
        )

        # ======================================================
        # 5. PUSTY TAB – INSTRUKCJA SPLOTU IR Z AUDIO
        # ======================================================

        add_section(
            tabs.tab("Instrukcja splotu IR z audio"),
            "Instrukcja splotu IR z sygnałem audio",
            "Ta sekcja zostanie uzupełniona."
        )

        # ======================================================
        # 6. O AUTORZE
        # ======================================================

        autor = (
            "Autor: Oskar Racułt\n"
            "Kierunek: Inżynieria Akustyczna\n"
            "Wydział Inżynierii Mechanicznej i Robotyki, Akademia Górniczo-Hutnicza w Krakowie\n\n"
            "Aplikacja opracowana jako część pracy inżynierskiej pt.:\n"
            "„Easy IResponse – Aplikacja do pomiaru i syntezy odpowiedzi impulsowej pomieszczeń "
            "na potrzeby auralizacji wnętrz”."
        )
        add_section(tabs.tab("O autorze"), "O autorze", autor)

        # ======================================================
        # 7. INFORMACJE TECHNICZNE
        # ======================================================

        techniczne = (
            "• Algorytm pomiarowy: Exponential Sine Sweep (ESS)\n"
            "• Tryby pracy: MONO oraz STEREO\n"
            "• Stereo:\n"
            "   – obsługa dwóch wejść: Input L / Input R\n"
            "   – możliwość korzystania z dwóch różnych urządzeń wejściowych\n"
            "   – automatyczne wyrównanie długości nagrania\n"
            "   – wspólna normalizacja IR dla kanałów L i R (wymagane przy HRTF)\n"
            "• Dekonwolucja: widmowe odwracanie sweepa, FFT(recorded) × FFT(inverse)\n"
            "• Windowing: fade-out oraz wyrównanie piku do t=0\n"
            "• Format audio: WAV 32-bit float\n"
            "• Wykresy: IR + Magnitude Response, smoothing do 1/24 okt.\n"
            "• Wymagania systemowe: Windows 10/11, Python 3.10+\n"
            "• Biblioteki: numpy, sounddevice, soundfile, customtkinter, matplotlib"
        )

        add_section(tabs.tab("Informacje techniczne"), "Informacje techniczne", techniczne)




# --------------------------------------------------
# GŁÓWNE OKNO APLIKACJI
# --------------------------------------------------
class EasyIResponseApp(ctk.CTk):

    def get_smoothing_fraction(self):
        settings_page = self.pages["settings"]
        return settings_page.get_smoothing_fraction()

    def get_ir_window_ms(self):
        settings_page = self.pages["settings"]
        return settings_page.get_ir_window_ms()

    def _safe_close(self):
        # 1. Stop input meter
        try:
            self.pages["settings"].stop_input_meter()
        except Exception:
            pass

        # 2. Stop pink noise + monitor wejścia z kalibracji
        try:
            self.pages["measurement"]._stop_pink_noise()
        except Exception:
            pass

        # 3. Anuluj ewentualną animację slide (if any)
        try:
            if hasattr(self, "anim_after_id") and self.anim_after_id is not None:
                self.after_cancel(self.anim_after_id)
        except Exception:
            pass

        # 4. (opcjonalnie) Anuluj after() w stronach, jeśli kiedyś dodasz tam takie atrybuty
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

        # 5. Ostatecznie zamknij aplikację – JEDNO destroy, nic więcej
        self.destroy()

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

        Zawiera:
        - input_device_L / input_device_R  – indexy wejść (Left / Right)
        - input_device                      – alias na Left (dla SPL, input meter)
        - input_channels                    – 1 (mono) lub 2 (stereo)
        - measurement_mode                  – 'Mono' / 'Stereo'
        """
        settings_page: SettingsPage = self.pages["settings"]

        in_L = settings_page.get_selected_input_L_index()
        in_R = settings_page.get_selected_input_R_index()
        out_idx = settings_page.get_selected_output_index()

        if in_L is None or out_idx is None:
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

        # tryb pomiaru
        mode = settings_page.get_measurement_mode()
        mode_lower = str(mode).lower()
        stereo = mode_lower.startswith("stereo")

        # liczba kanałów z punktu widzenia measurement_engine
        input_channels = 2 if stereo else 1

        # Jeśli stereo i oba inputy to ten sam device → upewnij się, że ma ≥ 2 kanały
        if stereo and in_L == in_R:
            try:
                dev_info = sd.query_devices(in_L)
                max_in = int(dev_info.get("max_input_channels", 1))
            except Exception:
                max_in = 1

            if max_in < 2:
                show_error(
                    f"Urządzenie wejściowe ma tylko {max_in} kanał(ów).\n"
                    f"Nie można użyć go w trybie stereo jako źródła L/R.\n"
                    f"Zmień urządzenie lub wybierz różne wejścia dla Left/Right."
                )
                return None

        return {
            "input_device": in_L,  # alias na Left – dla SPL / input metera
            "input_device_L": in_L,
            "input_device_R": in_R,
            "output_device": out_idx,
            "sample_rate": sr,
            "buffer_size": buf,
            "input_channels": input_channels,
            "measurement_mode": mode,
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
