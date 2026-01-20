# 🎧 Easy IResponse

**Easy IResponse** to desktopowa aplikacja audio napisana w Pythonie, przeznaczona do pomiaru, syntezy oraz wykorzystania odpowiedzi impulsowych (*Impulse Response, IR*) w zastosowaniach akustycznych i przetwarzaniu dźwięku.

Projekt został zrealizowany w ramach **pracy inżynierskiej z zakresu Inżynierii Akustycznej** i łączy klasyczne metody pomiarowe z praktycznym silnikiem splotu audio oraz obsługą HRTF.

---

## ✨ Funkcjonalności

### 📏 Pomiar odpowiedzi impulsowej
- pomiar IR metodą **logarytmicznego sweepa (Farina)**
- tryb pojedynczego pomiaru oraz uśredniania wielu pomiarów
- automatyczna dekonwolucja sygnału
- wizualizacja odpowiedzi impulsowej oraz charakterystyki częstotliwościowej

---

### 🧪 Synteza odpowiedzi impulsowej
- generowanie sztucznej odpowiedzi impulsowej na podstawie parametrów akustycznych
- rozdział odpowiedzi na:
  - dźwięk bezpośredni
  - wczesne odbicia
  - pogłos właściwy (late reverb)
- regulacja czasu pogłosu (**T60**)
- pasmowe sterowanie pochłanianiem w domenie częstotliwości (oktawowo)

---

### 🎚️ Splot audio
- splot dowolnego pliku audio z odpowiedzią impulsową
- obsługa trybu **mono** oraz **stereo**
- płynny miks sygnału **wet / dry**
- odsłuch **preview** bez zapisu do pliku
- normalizacja sygnału wyjściowego

---

### 🎧 HRTF
- import baz HRTF z plików `.mat`
- aplikowanie HRTF na poziomie odpowiedzi impulsowej
- binauralny odsłuch słuchawkowy
- regulacja azymutu i elewacji źródła dźwięku

---

## 🖥️ Interfejs użytkownika
- aplikacja desktopowa z graficznym interfejsem użytkownika (GUI)
- logiczny podział na zakładki:
  - Measurement
  - Generator
  - Convolution
  - Settings
  - About
- wizualizacja danych w czasie rzeczywistym
- ciemny motyw interfejsu

---

## 🛠️ Technologie
- **Python**
- **Tkinter / CustomTkinter**
- **NumPy**, **SciPy**
- **Matplotlib**
- obsługa plików **WAV** oraz **MAT**

---

## 🎓 Cel projektu
Celem projektu jest praktyczna implementacja:
- metod pomiaru odpowiedzi impulsowej,
- syntezy pogłosu,
- splotu audio,
- binauralnego renderingu dźwięku (HRTF),

z naciskiem na zastosowania **edukacyjne, badawcze i inżynierskie**.

---

## 🚀 Status
Projekt rozwijany w ramach pracy inżynierskiej.  
Możliwa dalsza rozbudowa o kolejne modele akustyczne oraz funkcje DSP.

---

## Wymagania środowiskowe

Do uruchomienia aplikacji z kodu źródłowego wymagane jest środowisko **Python 3.10**.

---

## Wersja wykonywalna (Windows)

Skompilowana wersja aplikacji dla systemu Windows (.exe) jest dostępna w sekcji **Releases** tego repozytorium:

👉 https://github.com/0-Revelin-0/inz_python/releases

Plik wykonywalny został wygenerowany z użyciem narzędzia **PyInstaller** i nie wymaga zainstalowanego środowiska Python.
Wystarczy pobrać plik `.exe` i uruchomić aplikację w systemie Windows.



