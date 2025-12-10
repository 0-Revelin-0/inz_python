from tkinter import filedialog
import customtkinter as ctk
import sounddevice as sd
import tkinter as tk
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

class ToolTip:
    def __init__(self, widget, text, delay=600):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.after_id = None

        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)

    def schedule(self, event=None):
        self.after_id = self.widget.after(self.delay, self.show)

    def show(self, event=None):
        if self.tip_window is not None:
            return

        # pozycjonowanie tooltipa względem widgetu
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20

        # okno tooltipa
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            background="#222222",
            foreground="white",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
            font=("Arial", 11)
        )
        label.pack()

    def hide(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class ConvolutionPage(ctk.CTkFrame):
    """
    GUI do splotu audio:
    - Mono / Stereo (dynamicznie zmienia ilość pól IR)
    - HRTF on/off
    - Wet/Dry
    - Plik audio
    - Plik wynikowy przyklejony na dole
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # ------- Zmienne -------
        self.mode_var = ctk.StringVar(value="Mono")
        self.hrtf_var = ctk.BooleanVar(value=False)

        self.ir1_var = ctk.StringVar()
        self.ir2_var = ctk.StringVar()
        self.audio_var = ctk.StringVar()
        self.output_var = ctk.StringVar()
        self.wetdry_var = ctk.DoubleVar(value=100.0)

        # ------- Layout główny -------
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, width=420)
        left.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")

        # stałe wiersze + jeden rozpychacz
        for r in range(10):
            left.grid_rowconfigure(r, weight=0)
        left.grid_rowconfigure(8, weight=1)   # pusty wiersz, który dopycha dół

        # Prawy panel (podgląd)
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        # ---------------- Tytuł ----------------
        title = ctk.CTkLabel(left, text="🎧  Splot audio", font=("Roboto", 22, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(5, 15))

        # ---------------- Tryb Mono / Stereo ----------------
        self.mode_seg = ctk.CTkSegmentedButton(
            left,
            values=["Mono", "Stereo"],
            variable=self.mode_var,
            command=self._update_ir_fields
        )
        self.mode_seg.grid(row=1, column=0, sticky="ew", pady=(5, 15))

        # ---------------- HRTF ----------------
        self.hrtf_switch = ctk.CTkSwitch(left, text="Użyj HRTF", variable=self.hrtf_var)
        self.hrtf_switch.grid(row=2, column=0, sticky="w", pady=(0, 15))

        # ---------------- IR FIELDS ----------------
        self.ir_frame = ctk.CTkFrame(left)
        self.ir_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        self._update_ir_fields()

        # ---------------- Plik audio ----------------
        ctk.CTkLabel(left, text="Plik audio:", font=("Arial", 14, "bold")).grid(
            row=4, column=0, sticky="w"
        )

        audio_row = ctk.CTkFrame(left)
        audio_row.grid(row=5, column=0, sticky="ew", pady=(5, 15))

        ctk.CTkEntry(audio_row, textvariable=self.audio_var).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ctk.CTkButton(
            audio_row, text="Wybierz", width=100, command=self._choose_audio
        ).pack(side="right")

        # ---------------- Wet/Dry ----------------
        ctk.CTkLabel(left, text="Wet/Dry (%)", font=("Arial", 14, "bold")).grid(
            row=6, column=0, sticky="w"
        )

        wd_row = ctk.CTkFrame(left)
        wd_row.grid(row=7, column=0, sticky="ew", pady=(5, 0))

        self.wetdry_slider = ctk.CTkSlider(
            wd_row, from_=0, to=100, variable=self.wetdry_var
        )
        self.wetdry_slider.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.wetdry_value = ctk.CTkLabel(wd_row, text="100%")
        self.wetdry_value.pack(side="right")

        self.wetdry_var.trace_add("write", self._update_wetdry_label)

        # ----------- Placeholder podglądu -----------
        ctk.CTkLabel(
            right,
            text="Tu będzie podgląd IR / audio / przebiegów.\nNa razie tylko GUI.",
            font=("Arial", 18),
        ).grid(row=0, column=0, sticky="nsew")

        # ---------------- DÓŁ: Plik wynikowy + Start ----------------
        bottom = ctk.CTkFrame(left)
        bottom.grid(row=9, column=0, sticky="ew", pady=(10, 0))

        # Plik wynikowy
        ctk.CTkLabel(bottom, text="Plik wynikowy:", font=("Arial", 14, "bold")).pack(
            anchor="w"
        )

        out_row = ctk.CTkFrame(bottom)
        out_row.pack(fill="x", pady=(5, 10))

        ctk.CTkEntry(out_row, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ctk.CTkButton(
            out_row, text="Wybierz", width=100, command=self._choose_output
        ).pack(side="right")

        # Start button
        ctk.CTkButton(
            bottom,
            text="▶ Start splotu",
            fg_color="#d71920",
            hover_color="#b01015",
            command=self._start_placeholder,
        ).pack(fill="x")

    # =============================================================
    # POLA IR – dynamicznie pokazujemy 1 lub 2 OKIENKA
    # =============================================================
    def _update_ir_fields(self, *_):
        for w in self.ir_frame.winfo_children():
            w.destroy()

        mode = self.mode_var.get()

        if mode == "Mono":
            self._make_ir_row(self.ir1_var, "IR mono:")
        else:
            self._make_ir_row(self.ir1_var, "IR Left:")
            self._make_ir_row(self.ir2_var, "IR Right:")

    def _make_ir_row(self, var, label_text):
        row = ctk.CTkFrame(self.ir_frame)
        row.pack(fill="x", pady=3)

        ctk.CTkLabel(row, text=label_text).pack(side="left", padx=(0, 5))
        ctk.CTkEntry(row, textvariable=var).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ctk.CTkButton(
            row, text="Wybierz", width=90, command=lambda: self._choose_ir(var)
        ).pack(side="right")

    # =============================================================
    # FUNKCJE PLIKÓW
    # =============================================================
    def _choose_ir(self, var):
        path = fd.askopenfilename(filetypes=[("WAV", "*.wav")])
        if path:
            var.set(path)

    def _choose_audio(self):
        path = fd.askopenfilename(filetypes=[("WAV", "*.wav")])
        if path:
            self.audio_var.set(path)

    def _choose_output(self):
        path = fd.asksaveasfilename(defaultextension=".wav")
        if path:
            self.output_var.set(path)

    # =============================================================
    # WET/DRY label update
    # =============================================================
    def _update_wetdry_label(self, *_):
        self.wetdry_value.configure(text=f"{int(self.wetdry_var.get())}%")

    # Placeholder
    def _start_placeholder(self):
        print("Splot będzie tu zaimplementowany.")






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
        main_frame.grid_columnconfigure(1, weight=2)
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
        self.sweep_length.bind("<KeyRelease>", self._on_sweep_change)

        self.start_freq = self._make_param(left, "Start freq [Hz]:", "100")
        self.end_freq = self._make_param(left, "End freq [Hz]:", "10000")
        self.ir_length = self._make_param(left, "Długość IR [s]:", "8")

        ToolTip(
            self.ir_length,
            "Długość odpowiedzi impulsowej.\n"
            "W trybie uśredniania (averages > 1) IR musi być\n"
            "≤ długości sweepa, aby uniknąć aliasingu ogona."
        )

        self.avg_count = self._make_param(left, "Uśrednianie (liczba uśrednień):", "1")

        ctk.CTkLabel(left, text="Tryb pomiaru:", font=("Arial", 16, "bold")).pack(anchor="w", pady=(15, 5))

        self.measure_mode_var = ctk.StringVar(value="single")

        self.mode_single = ctk.CTkRadioButton(
            left, text="Single sweep",
            variable=self.measure_mode_var, value="single",
            command=self._on_mode_change
        )
        self.mode_single.pack(anchor="w", pady=(5, 5))

        self.mode_avg = ctk.CTkRadioButton(
            left, text="Averaging (Farina)",
            variable=self.measure_mode_var, value="average",
            command=self._on_mode_change
        )
        self.mode_avg.pack(anchor="w", pady=(5, 5))

        self._on_mode_change()

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

    def _on_mode_change(self):
        mode = self.measure_mode_var.get()
        sweep_len = float(self.sweep_length.get())

        if mode == "average":
            # --- IR length = sweep (wymuszone) ---
            self.ir_length.configure(state="normal")
            self.ir_length.delete(0, "end")
            self.ir_length.insert(0, str(sweep_len))
            self.ir_length.configure(state="disabled")

            # --- averages odblokowane, minimum 2 ---
            self.avg_count.configure(state="normal")
            try:
                avg = int(self.avg_count.get())
            except:
                avg = 1
            if avg < 2:
                self.avg_count.delete(0, "end")
                self.avg_count.insert(0, "2")

        else:  # SINGLE SWEEP
            # --- IR length edytowalne, NIE nadpisujemy go! ---
            self.ir_length.configure(state="normal")

            # --- averages = 1 + blokada ---
            self.avg_count.configure(state="normal")
            self.avg_count.delete(0, "end")
            self.avg_count.insert(0, "1")
            self.avg_count.configure(state="disabled")

    def _on_sweep_change(self, event=None):
        """Aktualizacja IR length tylko w trybie AVERAGING."""

        mode = self.measure_mode_var.get()
        if mode != "average":
            return  # w trybie single nie zmieniamy nic

        try:
            new_len = float(self.sweep_length.get())
        except ValueError:
            return

        # Ustawiamy IR = sweep length (mimo że IR jest disabled)
        self.ir_length.configure(state="normal")
        self.ir_length.delete(0, "end")
        self.ir_length.insert(0, str(new_len))
        self.ir_length.configure(state="disabled")

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
            avg_count = int(self.avg_count.get())
        except ValueError:
            show_error("Błędne parametry pomiaru.\nSprawdź, czy wszystkie pola są liczbami.")
            return

        # Po parsowaniu sweep_len, start_f, end_f, ir_len, fade, avg_count

        if start_f <= 0:
            show_error("Start freq musi być > 0 Hz.")
            return

        if avg_count < 1:
            avg_count = 1
            self.avg_count.delete(0, "end")
            self.avg_count.insert(0, "1")

        # Zabezpieczenie: IR nie krótsza niż sweep
        if ir_len < sweep_len:
            ir_len = sweep_len
            # poprawiamy wartość w polu tekstowym, żeby user widział co się stało
            self.ir_length.delete(0, "end")
            self.ir_length.insert(0, str(sweep_len))
            self.status_label.configure(
                text="Długość IR była krótsza niż sweep.\nUstawiono IR = długość sweepa."
            )

        # Zabezpieczenie teoretyczne: przy averages > 1
        # IR nie może być dłuższa niż sweep (inaczej aliasing ogona IR)
        if avg_count > 1 and ir_len > sweep_len:
            ir_len = sweep_len
            self.ir_length.delete(0, "end")
            self.ir_length.insert(0, str(sweep_len))
            self.status_label.configure(
                text="Przy uśrednianiu IR nie może być dłuższa niż sweep.\n"
                     "Ustawiono IR = długość sweepa."
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
            "averages": avg_count,
        }

        params["mode"] = self.measure_mode_var.get()

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

        # -------------------------------------------------
        # GŁÓWNY UKŁAD – identyczny jak w MeasurementPage
        # -------------------------------------------------
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self, corner_radius=12)
        main_frame.grid(row=0, column=0, columnspan=2,
                        sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=0)  # lewy panel
        main_frame.grid_columnconfigure(1, weight=1)  # wykresy
        main_frame.grid_rowconfigure(0, weight=1)

        # =================================================
        # LEWA KOLUMNA — PARAMETRY GENEROWANIA IR
        # =================================================
        left_panel = ctk.CTkFrame(main_frame, corner_radius=12, width=300)
        left_panel.grid_propagate(False)
        left_panel.grid(row=0, column=0, padx=15, pady=15, sticky="ns")
        left_panel.grid_columnconfigure(0, weight=1)

        # Układ pionowy: góra (ustawienia) + rozpychacz + dół (ścieżka + przycisk)
        left_panel.grid_rowconfigure(0, weight=0)   # top_panel
        left_panel.grid_rowconfigure(1, weight=1)   # pusty wiersz – rozpycha
        left_panel.grid_rowconfigure(2, weight=0)   # bottom_panel

        # ------------------------- GÓRNY BLOK USTAWIEŃ -------------------------
        top_panel = ctk.CTkFrame(left_panel, fg_color="transparent")
        top_panel.grid(row=0, column=0, sticky="new")
        top_panel.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            top_panel,
            text="Generator IR",
            font=("Roboto", 22, "bold")
        )
        title.grid(row=0, column=0, sticky="w", pady=(5, 15))

        # 1. Mix Early / Late
        mix_frame = ctk.CTkFrame(top_panel)
        mix_frame.grid(row=1, column=0, sticky="ew", pady=(5, 15))
        mix_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            mix_frame,
            text="Charakter pogłosu (Early / Late):",
            font=("Roboto", 14)
        ).grid(row=0, column=0, sticky="w", pady=(5, 2))

        self.mix_slider = ctk.CTkSlider(
            mix_frame, from_=0, to=100, number_of_steps=100
        )
        self.mix_slider.set(30)
        self.mix_slider.grid(row=1, column=0, sticky="ew", padx=5)

        self.mix_label = ctk.CTkLabel(
            mix_frame,
            text="Early: 30%   Late: 70%"
        )
        self.mix_label.grid(row=2, column=0, sticky="w", pady=5)

        self.mix_slider.configure(command=self._update_mix_label)

        # 2. Parametry techniczne (u góry, jak było)
        params_frame = ctk.CTkFrame(top_panel)
        params_frame.grid(row=2, column=0, sticky="ew", pady=10)
        params_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(params_frame, text="Długość IR [s]:") \
            .grid(row=0, column=0, sticky="w", pady=5)
        self.ir_length_entry = ctk.CTkEntry(params_frame, width=110)
        self.ir_length_entry.insert(0, "3.0")
        self.ir_length_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # ------------------------- DOLNY BLOK: ŚCIEŻKA + PRZYCISK -------------------------
        bottom_panel = ctk.CTkFrame(left_panel, fg_color="transparent")
        bottom_panel.grid(row=2, column=0, sticky="sew", pady=(0, 5), padx=0)
        bottom_panel.grid_columnconfigure(0, weight=1)

        # 4. Folder / plik wyjściowy IR (nad przyciskiem, ale razem na dole)
        output_frame = ctk.CTkFrame(bottom_panel)
        output_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10))
        output_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_frame,
            text="Plik wyjściowy IR:"
        ).grid(row=0, column=0, sticky="w")

        self.ir_output_var = ctk.StringVar()
        self.output_entry = ctk.CTkEntry(
            output_frame,
            textvariable=self.ir_output_var
        )
        self.output_entry.grid(
            row=1, column=0, sticky="ew", pady=5, padx=(0, 5)
        )

        ctk.CTkButton(
            output_frame,
            text="Wybierz...",
            command=self._choose_output_path,
            width=110
        ).grid(row=1, column=1, padx=5, pady=5, sticky="e")

        # 3. Przycisk generowania – NA SAMYM DOLE
        self.generate_button = ctk.CTkButton(
            bottom_panel,
            text="▶ Generuj IR",
            fg_color="#d71920",
            hover_color="#b01015",
            command=self._on_generate_ir_clicked,
        )
        self.generate_button.grid(
            row=1, column=0, sticky="ew", pady=(0, 5)
        )

        # =================================================
        # PRAWA KOLUMNA — WYKRESY (identyczne jak MeasurementPage)
        # =================================================
        plot_frame = ctk.CTkFrame(main_frame, corner_radius=12)
        plot_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=0)  # (rezerwowane na nagłówek jeśli kiedyś dodasz)
        plot_frame.grid_rowconfigure(1, weight=1)  # fig/canvas

        # ---------- Figure z dwoma subplotami ----------
        self.fig = Figure(
            figsize=(6, 5),
            dpi=100,
            facecolor="#111111",
            tight_layout=True
        )

        # Impulse Response
        self.ax_ir = self.fig.add_subplot(2, 1, 1)
        self.ax_ir.set_facecolor("#111111")
        self.ax_ir.grid(True, color="#444444", alpha=0.3)
        self.ax_ir.set_title("Impulse Response", color="white")
        self.ax_ir.set_xlabel("Czas [s]", color="white")
        self.ax_ir.set_ylabel("Amplituda", color="white")
        self.ax_ir.tick_params(colors="white")

        # Magnitude Response
        self.ax_mag = self.fig.add_subplot(2, 1, 2)
        self.ax_mag.set_facecolor("#111111")
        self.ax_mag.grid(True, color="#444444", alpha=0.3)
        self.ax_mag.set_title("Magnitude Response", color="white")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
        self.ax_mag.set_ylabel("Poziom [dB]", color="white")
        self.ax_mag.tick_params(colors="white")

        # Canvas – tak jak w MeasurementPage
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=10
        )

        # inicjalne wyczyszczenie / ustawienie skali
        self._clear_plots()

    # =========================================================
    # KONFIGURACJA WYKRESÓW – identyczna jak w MeasurementPage
    # =========================================================
    def _clear_plots(self):
        # --- IR ---
        self.ax_ir.cla()
        self.ax_ir.set_facecolor("#111111")
        self.ax_ir.grid(True, color="#444444", alpha=0.3)
        self.ax_ir.set_title("Impulse Response", color="white")
        self.ax_ir.set_xlabel("Czas [s]", color="white")
        self.ax_ir.set_ylabel("Amplituda", color="white")
        self.ax_ir.tick_params(colors="white")

        # --- Magnitude ---
        self.ax_mag.cla()
        self.ax_mag.set_facecolor("#111111")
        self.ax_mag.grid(True, color="#444444", alpha=0.3)
        self.ax_mag.set_title("Magnitude Response", color="white")
        self.ax_mag.set_xlabel("Częstotliwość [Hz]", color="white")
        self.ax_mag.set_ylabel("Poziom [dB]", color="white")
        self.ax_mag.tick_params(colors="white")

        # skala logarytmiczna + ticki jak w MeasurementPage
        self.ax_mag.set_xscale("log")
        self.ax_mag.xaxis.set_major_locator(LogLocator(base=10.0))
        self.ax_mag.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10))
        )

        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        self.ax_mag.xaxis.set_major_formatter(formatter)

        # domyślny zakres (tu masz 20–20000, pomiar ma 100–10000; możesz zbliżyć, jeśli chcesz 1:1)
        self.ax_mag.set_xlim(20, 20000)

        self.canvas.draw()

    # =========================================================
    # LOGIKA GUI
    # =========================================================
    def _update_mix_label(self, value):
        early = int(float(value))
        late = 100 - early
        self.mix_label.configure(text=f"Early: {early}%   Late: {late}%")

    def _on_generate_ir_clicked(self):
        # Tu później podłączysz engine generowania IR,
        # na razie tylko placeholder.
        print("GENEROWANIE IR – tutaj podłączysz kod generujący IR")

    def _choose_output_path(self):
        # wybór KATALOGU – logika zapisu w engine
        path = filedialog.askdirectory()
        if path:
            self.ir_output_var.set(path)





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
        self.tabs = ctk.CTkTabview(self, width=950, height=620)
        self.tabs.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")

        self.tabs.add("Ustawienia pomiaru")
        self.tabs.add("Ustawienia generowania")

        gen_tab = self.tabs.tab("Ustawienia generowania")
        gen_tab.grid_rowconfigure(0, weight=1)
        gen_tab.grid_columnconfigure(0, weight=1)

        # ---------------------------------------------------------
        # TAB 1 — USTAWIENIA POMIARU (WSZYSTKO CO JUŻ MAMY)
        # ---------------------------------------------------------
        measure_tab = self.tabs.tab("Ustawienia pomiaru")
        measure_tab.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(measure_tab, corner_radius=12)
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)

        # ==========================
        #   UKŁAD USTAWIEŃ POMIARU
        # ==========================

        row_i = 0

        # Input device
        ctk.CTkLabel(frame, text="Input device:").grid(row=row_i, column=0, padx=10, pady=5, sticky="w")
        self.input_device_combo = ctk.CTkComboBox(frame, values=[])
        self.input_device_combo.grid(row=row_i, column=1, padx=10, pady=5, sticky="ew")
        row_i += 1

        # Output device
        ctk.CTkLabel(frame, text="Output device:").grid(row=row_i, column=0, padx=10, pady=5, sticky="w")
        self.output_combo = ctk.CTkComboBox(frame, values=[])
        self.output_combo.grid(row=row_i, column=1, padx=10, pady=5, sticky="ew")
        row_i += 1

        # Sample rate
        ctk.CTkLabel(frame, text="Sample rate [Hz]:").grid(row=row_i, column=0, padx=10, pady=5, sticky="w")
        self.sample_rate_combo = ctk.CTkComboBox(frame, values=["44100", "48000", "88200", "96000"])
        self.sample_rate_combo.set("48000")
        self.sample_rate_combo.grid(row=row_i, column=1, padx=10, pady=5, sticky="ew")
        row_i += 1

        # Buffer size
        ctk.CTkLabel(frame, text="Buffer size [frames]:").grid(row=row_i, column=0, padx=10, pady=5, sticky="w")
        self.buffer_size_combo = ctk.CTkComboBox(frame, values=["64", "128", "256", "512", "1024"])
        self.buffer_size_combo.set("256")
        self.buffer_size_combo.grid(row=row_i, column=1, padx=10, pady=5, sticky="ew")
        row_i += 1

        # Smoothing
        ctk.CTkLabel(frame, text="Smoothing:").grid(row=row_i, column=0, padx=10, pady=5, sticky="w")
        self.smoothing_combo = ctk.CTkComboBox(
            frame,
            values=["Raw", "1/24", "1/12", "1/6", "1/3"],
            command=self._on_smoothing_change
        )
        self.smoothing_combo.set("1/6")
        self.smoothing_combo.grid(row=row_i, column=1, padx=10, pady=5, sticky="ew")
        row_i += 1

        # Add stretch to column 1
        frame.grid_columnconfigure(1, weight=1)

        # IR window after peak (ms)
        ctk.CTkLabel(frame, text="IR window after peak [ms]:").grid(
            row=5, column=0, padx=10, pady=10, sticky="w"
        )
        self.ir_window_entry = ctk.CTkEntry(frame, width=120)
        self.ir_window_entry.grid(row=5, column=1, padx=10, pady=10, sticky="ew")
        self.ir_window_entry.insert(0, "500")
        self.ir_window_entry.bind("<KeyRelease>", self._on_ir_window_change)

        # Measurement mode (mono / stereo)
        ctk.CTkLabel(frame, text="Measurement mode:").grid(
            row=6, column=0, padx=10, pady=10, sticky="w"
        )
        self.measure_mode_combo = ctk.CTkComboBox(
            frame,
            values=["Mono", "Stereo"]
        )
        self.measure_mode_combo.set("Mono")
        self.measure_mode_combo.grid(row=6, column=1, padx=10, pady=10, sticky="ew")

        # Input meter
        ctk.CTkLabel(frame, text="Test input:").grid(row=7, column=0, padx=10, pady=10, sticky="w")
        self.meter_bar = ctk.CTkProgressBar(frame, width=220)
        self.meter_bar.grid(row=7, column=1, padx=10, pady=10, sticky="e")
        self.meter_bar.set(0)

        self.meter_btn = ctk.CTkButton(frame, text="Start input meter", command=self._toggle_meter)
        self.meter_btn.grid(row=7, column=1, padx=10, pady=10, sticky="w")

        # Test tone
        ctk.CTkLabel(frame, text="Test output:").grid(row=9, column=0, padx=10, pady=10, sticky="w")
        self.test_btn = ctk.CTkButton(frame, text="Test tone (1 kHz)", command=self._play_test)
        self.test_btn.grid(row=9, column=1, padx=10, pady=15, sticky="w")

        # Załaduj listę wejść/wyjść
        self._load_devices()

        # =========================================================
        # SCROLLOWALNY OBSZAR DLA USTAWIEŃ GENEROWANIA
        # =========================================================

        # Canvas + Scrollbar
        canvas = ctk.CTkCanvas(gen_tab, bg="black", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(gen_tab, orientation="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=scrollbar.set)

        # Właściwy frame na treść
        gen_frame = ctk.CTkFrame(canvas, corner_radius=12)
        canvas_window = canvas.create_window((0, 0), window=gen_frame, anchor="nw")

        gen_frame.grid_columnconfigure(0, weight=1)

        # Dopasowanie szerokości frame do szerokości canvasa
        def _resize_frame(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _resize_frame)

        # Automatyczny scroll-region (gdy zawartość rośnie)
        def _update_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        gen_frame.bind("<Configure>", _update_scrollregion)

        # Rozciąganie wszystkiego na całą stronę
        gen_tab.grid_rowconfigure(0, weight=1)
        gen_tab.grid_columnconfigure(0, weight=1)

        # =========================================================
        # 0) PARAMETRY GENEROWANIA (Sample rate)
        # =========================================================
        section_gen = ctk.CTkLabel(gen_frame, text="Parametry generowania",
                                   font=("Roboto", 20, "bold"))
        section_gen.grid(row=0, column=0, sticky="w", pady=(10, 5), padx=10)

        gen_params_frame = ctk.CTkFrame(gen_frame)
        gen_params_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 15))
        gen_params_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(gen_params_frame, text="Sample rate [Hz]:    ").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.gen_sample_rate_combo = ctk.CTkComboBox(
            gen_params_frame,
            values=["44100", "48000", "88200", "96000"],
            width=120
        )
        self.gen_sample_rate_combo.set("48000")
        self.gen_sample_rate_combo.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        # =========================================================
        # 1) GEOMETRIA POMIESZCZENIA
        # =========================================================
        section_geo = ctk.CTkLabel(gen_frame, text="Geometria pomieszczenia", font=("Roboto", 20, "bold"))
        section_geo.grid(row=2, column=0, sticky="w", pady=(10, 5), padx=10)

        geo_frame = ctk.CTkFrame(gen_frame)
        geo_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 15))
        geo_frame.grid_columnconfigure((1), weight=1)

        ctk.CTkLabel(geo_frame, text="Szerokość W [m]:    ").grid(row=0, column=0, sticky="w", pady=3)
        self.room_w = ctk.CTkEntry(geo_frame, width=100)
        self.room_w.insert(0, "5.0")
        self.room_w.grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkLabel(geo_frame, text="Długość L [m]:    ").grid(row=1, column=0, sticky="w", pady=3)
        self.room_l = ctk.CTkEntry(geo_frame, width=100)
        self.room_l.insert(0, "7.0")
        self.room_l.grid(row=1, column=1, sticky="ew", padx=5)

        ctk.CTkLabel(geo_frame, text="Wysokość H [m]:    ").grid(row=2, column=0, sticky="w", pady=3)
        self.room_h = ctk.CTkEntry(geo_frame, width=100)
        self.room_h.insert(0, "2.7")
        self.room_h.grid(row=2, column=1, sticky="ew", padx=5)

        # =========================================================
        # 2) WSPÓŁCZYNNIKI POCHŁANIANIA – TABELA
        # =========================================================
        section_abs = ctk.CTkLabel(
            gen_frame,
            text="Pochłanianie powierzchni",
            font=("Roboto", 20, "bold")
        )
        section_abs.grid(row=4, column=0, sticky="w", pady=(10, 5), padx=10)

        # Zewnętrzna ramka na całą szerokość
        abs_outer = ctk.CTkFrame(gen_frame)
        abs_outer.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 15))
        abs_outer.grid_columnconfigure(0, weight=1)

        # Właściwa tabela – będzie wyśrodkowana w abs_outer
        abs_frame = ctk.CTkFrame(abs_outer)
        abs_frame.grid(row=0, column=0)

        freqs = ["125", "250", "500", "1k", "2k", "4k"]
        surfaces = ["Ściany", "Sufit", "Podłoga"]

        # Nagłówki kolumn
        ctk.CTkLabel(abs_frame, text="").grid(row=0, column=0, padx=5)
        for i, f in enumerate(freqs):
            ctk.CTkLabel(abs_frame, text=f"{f} Hz").grid(row=0, column=i + 1, padx=10)

        self.abs_entries = {}

        for r, surf in enumerate(surfaces):
            ctk.CTkLabel(abs_frame, text=surf).grid(row=r + 1, column=0, padx=10, pady=5, sticky="w")
            for c, f in enumerate(freqs):
                e = ctk.CTkEntry(abs_frame, width=60)
                e.insert(0, "0.20")
                e.grid(row=r + 1, column=c + 1, padx=5, pady=5)
                self.abs_entries[(surf, f)] = e

        # =========================================================
        # 3) PARAMETRY WCZESNYCH ODBIĆ (FDN)
        # =========================================================
        section_fdn = ctk.CTkLabel(gen_frame, text="Wczesne odbicia (FDN)", font=("Roboto", 20, "bold"))
        section_fdn.grid(row=6, column=0, sticky="w", pady=(10, 5), padx=10)

        fdn_frame = ctk.CTkFrame(gen_frame)
        fdn_frame.grid(row=7, column=0, sticky="ew", padx=10, pady=(0, 15))
        fdn_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(fdn_frame, text="Liczba promieni:    ").grid(row=0, column=0, sticky="w", pady=5)
        self.rays_entry = ctk.CTkEntry(fdn_frame, width=80)
        self.rays_entry.insert(0, "8")
        self.rays_entry.grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(fdn_frame, text="Liczba odbić/promień:     ").grid(row=1, column=0, sticky="w", pady=5)
        self.reflections_entry = ctk.CTkEntry(fdn_frame, width=80)
        self.reflections_entry.insert(0, "20")
        self.reflections_entry.grid(row=1, column=1, sticky="w")

        ctk.CTkLabel(fdn_frame, text="Rozrzut pierwszego odbicia [%]:     ").grid(row=2, column=0, sticky="w", pady=5)
        self.first_dev_entry = ctk.CTkEntry(fdn_frame, width=80)
        self.first_dev_entry.insert(0, "15")
        self.first_dev_entry.grid(row=2, column=1, sticky="w")

        ctk.CTkLabel(fdn_frame, text="Rozrzut mean free path [%]:    ").grid(row=3, column=0, sticky="w", pady=5)
        self.mfp_dev_entry = ctk.CTkEntry(fdn_frame, width=80)
        self.mfp_dev_entry.insert(0, "10")
        self.mfp_dev_entry.grid(row=3, column=1, sticky="w")

        # =========================================================
        # 4) PARAMETRY PÓŹNEGO POGŁOSU (T60)
        # =========================================================
        section_t60 = ctk.CTkLabel(gen_frame, text="Późny pogłos (T60)", font=("Roboto", 20, "bold"))
        section_t60.grid(row=8, column=0, sticky="w", pady=(10, 5), padx=10)

        t60_frame = ctk.CTkFrame(gen_frame)
        t60_frame.grid(row=9, column=0, sticky="ew", padx=10, pady=(0, 15))
        t60_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(t60_frame, text="Tryb obliczania T60:").grid(row=0, column=0, sticky="w", pady=5)

        self.t60_mode = ctk.StringVar(value="auto")
        self.t60_auto = ctk.CTkRadioButton(
            t60_frame, text="Auto (Sabine)", variable=self.t60_mode, value="auto",
            command=self._toggle_t60_manual
        )
        self.t60_auto.grid(row=0, column=2, sticky="w")

        self.t60_manual = ctk.CTkRadioButton(
            t60_frame, text="Ręczne", variable=self.t60_mode, value="manual",
            command=self._toggle_t60_manual
        )
        self.t60_manual.grid(row=0, column=3, sticky="w", padx=15)

        # Tabela T60 manual
        self.t60_entries = {}
        t60_freqs = ["125", "250", "500", "1k", "2k", "4k"]

        t60_table = ctk.CTkFrame(t60_frame)
        t60_table.grid(row=1, column=0, columnspan=3, pady=10)

        ctk.CTkLabel(t60_table, text="Czas pogłosu T60 [s]").grid(row=0, column=0, columnspan=7, pady=5)

        ctk.CTkLabel(t60_table, text="").grid(row=1, column=0)
        for i, f in enumerate(t60_freqs):
            ctk.CTkLabel(t60_table, text=f"{f} Hz").grid(row=1, column=i + 1, padx=10)

        for i, f in enumerate(t60_freqs):
            e = ctk.CTkEntry(t60_table, width=60)
            e.insert(0, "0.6")
            e.grid(row=2, column=i + 1, padx=5, pady=5)
            self.t60_entries[f] = e

        # Domyślnie pola manual T60 są zablokowane
        for e in self.t60_entries.values():
            e.configure(state="disabled")

    # =====================================================================
    # --- DEVICE HANDLING (jak wcześniej) ---
    # =====================================================================

    def _toggle_t60_manual(self):
        manual = (self.t60_mode.get() == "manual")
        for e in self.t60_entries.values():
            e.configure(state="normal" if manual else "disabled")

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

        # Oba inputy (L/R) korzystają z tej samej listy
        self.input_device_combo.configure(values=input_values)
        self.output_combo.configure(values=output_values)

        # Domyślny wybór
        self.input_device_combo.set(input_values[0])
        self.output_combo.set(output_values[0])

    def get_selected_input_device_index(self):
        selected = self.input_device_combo.get()
        for entry in self.input_entries:
            if entry["label"] == selected:
                return entry["index"]
        return None

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
        in_idx = self.get_selected_input_device_index()

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
        """
        Zachowanie wsteczne – alias na główne wejście.
        Teraz używamy jednego urządzenia input_device_combo.
        """
        return self.get_selected_input_device_index()

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

        def add_section(frame, title, text):
            # Tytuł sekcji
            title_label = ctk.CTkLabel(
                frame,
                text=title,
                font=("Arial", 20, "bold"),
                anchor="w"
            )
            title_label.pack(anchor="w", pady=(10, 5))

            # Scrollowalny tekst sekcji
            textbox = ctk.CTkTextbox(
                frame,
                wrap="word",
                font=("Arial", 15),
                height=350  # możesz dostosować wysokość
            )
            textbox.pack(fill="both", expand=True, padx=10, pady=5)

            # Wstaw tekst i zablokuj edytowanie
            textbox.insert("0.0", text)
            textbox.configure(state="disabled")

            return textbox

        # ======================================================
        # 1. OPIS PROGRAMU
        # ======================================================

        opis = (
            "Easy IResponse to aplikacja do precyzyjnego pomiaru odpowiedzi impulsowej (IR) "
            "z użyciem metody Exponential Sine Sweep (ESS) zgodnej z techniką Fariny. "
            "Program łączy generację sweepa, odtwarzanie, jednoczesne nagrywanie sygnału, "
            "dekonwolucję oraz analizę częstotliwościową.\n\n"
            "Aplikacja obsługuje pomiary MONO oraz STEREO, w tym dwa niezależne wejścia "
            "wejściowe (Input L oraz Input R), co umożliwia pomiary dwukanałowe, pomiary HRTF "
            "oraz rejestrację odpowiedzi dwóch mikrofonów jednocześnie. "
            "Kanały są normalizowane wspólnie, zapewniając zachowanie relacji poziomów.\n\n"
            "Program obsługuje uśrednianie wielu sweepów (concatenated sweep averaging), "
            "które znacząco poprawia stosunek sygnał/szum (SNR). Zaimplementowano pełny model "
            "uśredniania synchronicznego: sklejanie sweepów w jednym buforze, dzielenie nagrania "
            "na okna oraz dekonwolucję okna uśrednionego.\n\n"
            "Interfejs aplikacji pozwala konfigurować wszystkie parametry pomiarowe, przeglądać IR, "
            "charakterystyki amplitudowe oraz eksportować wyniki do plików WAV."
        )

        add_section(tabs.tab("Opis programu"), "Opis programu", opis)

        # ======================================================
        # 2. FUNKCJONALNOŚCI
        # ======================================================

        funkcje = (
            "• Pomiar odpowiedzi impulsowej metodą ESS (Exponential Sine Sweep)\n"
            "• Tryby pomiarowe: MONO oraz STEREO\n"
            "• Obsługa dwóch niezależnych wejść audio (Input L / Input R)\n"
            "• Możliwość użycia jednego urządzenia 2-kanałowego lub dwóch osobnych urządzeń\n"
            "• Jednoczesne nagrywanie dwóch kanałów w trybie stereo\n"
            "• Wspólna normalizacja IR kanałów L i R — wymagana przy pomiarach HRTF\n"
            "• Uśrednianie wielu sweepów (concatenated sweep averaging) poprawiające SNR\n"
            "• Automatyczne sprawdzanie zgodności długości IR z długością sweepa\n"
            "• Pełna dekonwolucja widmowa (FFT · inverse-sweep)\n"
            "• Generacja sweepa z fade-out, aby uniknąć klików na łączeniach\n"
            "• Analiza IR w dziedzinie czasu i częstotliwości\n"
            "• Dynamiczny wybór kanału do wyświetlania (L / R)\n"
            "• Surowe nagrania + IR zapisywane osobno dla obu kanałów\n"
            "• Kalibracja SPL z monitoringiem poziomu wejściowego w czasie rzeczywistym\n"
            "• Smoothing charakterystyki: Raw, 1/24, 1/12, 1/6, 1/3 okt.\n"
            "• Zmienne okno wizualizacji IR za pikiem (ms)\n"
            "• Integracja z real-time audio (sounddevice)\n"
        )
        add_section(tabs.tab("Funkcjonalności"), "Funkcjonalności", funkcje)

        # ======================================================
        # 3. INSTRUKCJA POMIARU
        # ======================================================

        instrukcja = (
            "1. W zakładce „Ustawienia pomiaru” wybierz urządzenia audio:\n"
            "   • Input (Left) — lewy kanał pomiarowy\n"
            "   • Input (Right) — prawy kanał pomiarowy\n"
            "   • Output — urządzenie odtwarzające sweep\n"
            "   Możesz użyć dwóch osobnych urządzeń wejściowych.\n\n"

            "2. Wybierz tryb pracy:\n"
            "   • MONO — nagrywany jest tylko kanał Left\n"
            "   • STEREO — nagrywane są oba kanały równolegle\n\n"

            "3. Skonfiguruj parametry:\n"
            "   • Długość sweepa\n"
            "   • Zakres częstotliwości start/end\n"
            "   • Fade (domyślnie 0.05 s — usuwa klik przy końcu sweepa)\n"
            "   • Długość IR\n"
            "   • Liczbę uśrednień (averages)\n\n"

            "4. Ważne zasady dotyczące uśredniania:\n"
            "   • Aplikacja generuje (averages + 1) sklejonych sweepów.\n"
            "   • Pierwsze okno nagrania jest odrzucane.\n"
            "   • Kolejne okna są uśredniane synchronicznie.\n"
            "   • Aby uniknąć aliasingu czasowego, długość IR musi być "
            "≤ długości sweepa, gdy averages > 1. Program automatycznie to wymusza.\n\n"

            "5. Wykonaj kalibrację SPL (opcjonalnie), aby ustawić poprawny poziom nagrania.\n\n"

            "6. Naciśnij „Start measurement”:\n"
            "   • Program odtworzy sklejone sweepy\n"
            "   • Nagranie będzie równoległe (L / R)\n"
            "   • IR zostaną zdekoniugowane i normalizowane jednym wspólnym współczynnikiem\n\n"

            "7. Zapisane pliki:\n"
            "   • RECORDED_L_*.wav — surowe nagranie lewego kanału\n"
            "   • RECORDED_R_*.wav — surowe nagranie prawego kanału\n"
            "   • IR_L_*.wav — odpowiedź impulsowa kanału L\n"
            "   • IR_R_*.wav — odpowiedź impulsowa kanału R\n\n"

            "8. Wykresy:\n"
            "   • IR jest przycinana wizualnie do okna za głównym pikiem\n"
            "   • Charakterystyka amplitudowa może być wygładzona"
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
            "• Algorytm pomiarowy: Exponential Sine Sweep (ESS) wg Fariny\n"
            "• Generacja sweepa z fade-out, aby pierwszy i ostatni punkt były równe zero\n"
            "• Uśrednianie concat sweepów: sklejanie bez przerwy próbki do próbki\n"
            "• Dzielenie nagrania na okna długości jednego sweepa (pierwsze pomijane)\n"
            "• Uśrednianie synchroniczne poprawiające SNR\n"
            "• Wymuszenie IR_length ≤ sweep_length, gdy averages > 1 (ochrona przed aliasingiem)\n"
            "• Dekonwolucja: FFT(recorded) × conj(FFT(inverse_sweep))\n"
            "• Wspólna normalizacja IR kanałów L i R — identyczny współczynnik\n"
            "• Format nagrań i IR: WAV 32-bit float\n"
            "• Zachowano zgodność długości kanałów stereo oraz pre-align piku IR\n"
            "• Wykresy: czasowe + częstotliwościowe z smoothingiem\n"
            "• Biblioteki: numpy, soundfile, sounddevice, matplotlib, customtkinter\n"
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
            self.pages["settings"].stop_input_monitor()
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
        self.resizable(False, False)

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

        # Ustawienia układu
        self.sidebar.grid_rowconfigure(0, weight=0)  # tytuł
        self.sidebar.grid_rowconfigure(1, weight=0)  # opis
        self.sidebar.grid_rowconfigure(2, weight=0)  # Pomiar IR
        self.sidebar.grid_rowconfigure(3, weight=0)  # Generator IR
        self.sidebar.grid_rowconfigure(4, weight=0)  # Splot audio
        self.sidebar.grid_rowconfigure(5, weight=1)  # ROZPYCHA — puste miejsce
        self.sidebar.grid_rowconfigure(6, weight=0)  # Ustawienia
        self.sidebar.grid_rowconfigure(7, weight=0)  # O programie

        # Tytuł
        title_label = ctk.CTkLabel(
            self.sidebar, text="Easy IResponse", font=("Roboto", 22, "bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        subtitle = ctk.CTkLabel(
            self.sidebar, text="IR measurement & synthesis", font=("Roboto", 11)
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

    def _create_nav_buttons(self):

        btn_measure = ctk.CTkButton(
            self.sidebar, text="🎤  Pomiar IR", anchor="w",
            command=lambda: self.set_active_page("measurement")
        )
        btn_measure.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        btn_generator = ctk.CTkButton(
            self.sidebar, text="🌊  Generator IR", anchor="w",
            command=lambda: self.set_active_page("generator")
        )
        btn_generator.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        btn_convolution = ctk.CTkButton(
            self.sidebar, text="🎧  Splot audio", anchor="w",
            command=lambda: self.set_active_page("convolution")
        )
        btn_convolution.grid(row=4, column=0, padx=15, pady=5, sticky="ew")

        # Rozpychająca przerwa — row 5

        btn_settings = ctk.CTkButton(
            self.sidebar, text="⚙️  Ustawienia", anchor="w",
            command=lambda: self.set_active_page("settings")
        )
        btn_settings.grid(row=6, column=0, padx=15, pady=5, sticky="ew")

        btn_about = ctk.CTkButton(
            self.sidebar, text="ℹ️  O programie", anchor="w",
            command=lambda: self.set_active_page("about")
        )
        btn_about.grid(row=7, column=0, padx=15, pady=15, sticky="ew")

        self.nav_buttons = {
            "measurement": btn_measure,
            "generator": btn_generator,
            "convolution": btn_convolution,
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
        self.pages["convolution"] = ConvolutionPage(self.content_container, self)  # <-- DODANE
        self.pages["settings"] = SettingsPage(self.content_container, self)
        self.pages["about"] = AboutPage(self.content_container, self)

        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            page.place_forget()

    def get_measurement_audio_config(self):
        """
        Zwraca słownik z ustawieniami audio dla pomiaru IR
        na podstawie zakładki 'Ustawienia pomiaru'.

        Zawiera:
          - input_device      – index wejścia (jedno urządzenie)
          - output_device     – index wyjścia
          - sample_rate       – fs [Hz]
          - buffer_size       – rozmiar bufora (frames)
          - input_channels    – 1 (Mono) lub 2 (Stereo)
          - measurement_mode  – 'Mono' / 'Stereo'
        """
        settings_page: SettingsPage = self.pages["settings"]

        in_dev = settings_page.get_selected_input_device_index()
        out_idx = settings_page.get_selected_output_index()

        if in_dev is None or out_idx is None:
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

        # liczba kanałów
        input_channels = 2 if stereo else 1

        # sprawdź, czy urządzenie ma wystarczającą liczbę kanałów wejściowych
        try:
            dev_info = sd.query_devices(in_dev)
            max_in = int(dev_info.get("max_input_channels", 1))
        except Exception:
            max_in = 1

        if input_channels > max_in:
            show_error(
                f"Urządzenie wejściowe ma tylko {max_in} kanał(ów).\n"
                f"Nie można użyć go w trybie {'stereo' if stereo else 'mono'}."
            )
            return None

        return {
            "input_device": in_dev,
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
