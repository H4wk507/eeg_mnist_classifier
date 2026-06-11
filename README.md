# Klasyfikowanie cyfr z sygnału EEG

Projekt zaliczeniowy na przedmiot **Akwizycja danych biomedycznych**.

Celem projektu jest sprawdzenie, czy na podstawie sygnału EEG zarejestrowanego w chwili, gdy
badany **widzi na ekranie cyfrę (0–9) i jednocześnie o niej myśli**, da się odtworzyć
(sklasyfikować), którą cyfrę mu pokazano. Innymi słowy - klasyczne zadanie rozpoznawania cyfr
0–9, tyle że nie z obrazów, lecz z aktywności mózgu wywołanej ich oglądaniem.

Projekt obejmuje pełną ścieżkę pracy z danymi biomedycznymi: akwizycję/wczytanie surowego sygnału,
preprocessing (filtracja, usuwanie artefaktów, referencja), eksploracyjną analizę danych (EDA),
ekstrakcję cech, trenowanie modeli klasyfikacyjnych oraz uczciwą ewaluację z analizą błędów.

## Zbiór danych

Wykorzystujemy publiczny zbiór **[MindBigData2022_MNIST_EP](https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_EP)**
(wariant **EP**, rejestrowany czujnikiem **Emotiv EPOC**). Dane pobierane są automatycznie
z Hugging Face Hub przez bibliotekę `datasets`.

**Protokół akwizycji:** jednemu badanemu (David Vivancos) wyświetlano na ekranie cyfrę 0–9
i przez ~2 s rejestrowano sygnał EEG, gdy badany ją **widział i jednocześnie o niej myślał**.
To nie była więc czysta wyobraźnia: jest tu realny bodziec wzrokowy, dlatego sygnał różnicującKlasyfikowanie cyfr z sygnału EEGy
to w dużej mierze **odpowiedź wzrokowa** (visual evoked potential). To z kolei tłumaczy obserwację
z EDA, że cyfry najsilniej różnicują kanały potyliczne **O1/O2 (kora wzrokowa)**. Etykieta `-1`
oznacza nagrania bez bodźca (do kontrastu), które odrzucamy.

Charakterystyka sygnału:

| Parametr                    | Wartość                                                      |
| --------------------------- | ------------------------------------------------------------ |
| Urządzenie                  | Emotiv EPOC (14 kanałów)                                     |
| Kanały                      | `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4` |
| Częstotliwość próbkowania   | 128 Hz                                                       |
| Długość pojedynczego trialu | 256 próbek (≈ 2 s)                                           |
| Klasy                       | cyfry 0–9 (etykieta `-1` = brak bodźca, odrzucana)           |
| Liczba trialów              | ~52 tys. (train) + ~13 tys. (test)                           |
| Kształt po wczytaniu        | `(N, 14, 256)`                                               |

Klasy są w przybliżeniu zbalansowane (~1300 trialów na cyfrę w zbiorze testowym), dlatego jako
miary jakości używamy zarówno _accuracy_, jak i _macro F1_.

## Struktura projektu

```
.
├── notebooks/
│   ├── 01_EDA.ipynb             # eksploracyjna analiza danych (korelacja, FC, wPLI, DTW)
│   ├── 02_model.ipynb           # pierwsza wersja modelu (1D CNN na surowym sygnale)
│   ├── 02_modelv2.ipynb         # EEGNet + baseline RandomForest
│   ├── 02_modelv2_augment.ipynb # wariant z augmentacją i oknowaniem
│   ├── 02_modelv2_mne.ipynb     # wariant z preprocessingiem MNE
│   └── eeg_lib.py               # wspólna logika: preprocessing, cechy, modele, trening, ewaluacja
├── models/                      # wytrenowane wagi (.pth) + wykresy (krzywe uczenia, macierze pomyłek)
├── report_assets/               # rysunki do raportu (tSNE, DTW, wPLI, confusion matrix, ...)
├── papers/                      # notatki z literatury (Khaleghi, Kumari)
├── glossary.md                  # słowniczek pojęć (EEG, FC, wPLI, DTW, MNE)
├── notes.md                     # robocze notatki i wnioski
├── call-22.05.md                # notatki z konsultacji
├── raport_ep.docx               # raport końcowy
├── pyproject.toml               # zależności (zarządzane przez uv)
└── Makefile                     # zadania pomocnicze (lint, format, clean)
```

## Metodologia

### 1. Preprocessing

Cała wspólna logika znajduje się w [`notebooks/eeg_lib.py`](notebooks/eeg_lib.py). Dostępne są dwie ścieżki:

- **MNE** (`mne_preprocess`) - usunięcie offsetu DC, konwersja odczytów ADC na wolty,
  filtr notch 50 Hz (sieciówka), filtr pasmowy 1–45 Hz, referencja uśredniona oraz
  odrzucanie epok z artefaktami progiem 150 µV.
- **SciPy** (`filter_signal`) - detrend liniowy, zero-fazowy bandpass Butterwortha, korekcja
  baseline, próg peak-to-peak liczony jako `mediana + k·MAD` **wyłącznie na zbiorze treningowym**
  (brak wycieku danych).

Standaryzacja z-score liczona jest na statystykach ze zbioru treningowego (`zscore_fit` / `zscore_apply`)
lub per epoka (`zscore_per_epoch`).

### 2. Eksploracyjna analiza danych (EDA)

W [`01_EDA.ipynb`](notebooks/01_EDA.ipynb) badamy, czy w sygnale w ogóle istnieje sygnał
różnicujący cyfry:

- **Korelacja Pearsona** między kanałami, uśredniona per cyfra,
- **Functional / Spectral Connectivity** oraz **wPLI** (weighted Phase Lag Index) w pasmach theta i alpha,
- **DTW** (Dynamic Time Warping) - odległość między uśrednionymi przebiegami poszczególnych cyfr,
  liczona per kanał (m.in. O1/O2),
- analiza, **które kanały** najsilniej różnicują cyfry.

Pojęcia te są wyjaśnione w [`glossary.md`](glossary.md).

### 3. Modele

- **Baseline klasyczny** - RandomForest na cechach inżynierskich: log-moc w 5 pasmach (delta/theta/alpha/beta/gamma,
  estymowana metodą Welcha) + parametry Hjortha (activity, mobility, complexity).
- **EEGNet1D** - lekka sieć 1D CNN traktująca kanały EEG jako kanały wejściowe `Conv1d` (+ tSNE na embeddingach).
- **EEGNet (Lawhern et al. 2018)** - kompaktowy CNN dedykowany sygnałowi EEG (~12,5 tys. parametrów).

Trening: optymalizator Adam, harmonogram cosine LR, early stopping na walidacyjnym macro-F1,
opcjonalna augmentacja (cykliczne przesunięcie w czasie, szum gaussowski, channel dropout) oraz
ewaluacja oknowa z uśrednianiem predykcji per epoka.

### 4. Ewaluacja

Podział train/val = 80/20 (stratyfikowany), osobny zbiór testowy. Metryki: accuracy, macro F1,
macierz pomyłek, raport per klasa oraz wizualizacja tSNE embeddingów (czy sieć separuje klasy).

## Wyniki

Najlepszy model (EEGNet) osiąga na zbiorze testowym **accuracy ≈ 0.20** i **macro F1 ≈ 0.195**,
czyli ok. **2× powyżej poziomu losowego** (0.10 dla 10 klas).

Wnioski:

- Model najlepiej radzi sobie z cyframi **0, 1, 7** (najwyższy recall i F1), a najsłabiej z **2, 3, 8, 9**.
  Jest to spójne z EDA - cyfry `1` i `7` najbardziej odstawały na kanałach wzrokowych (O1/O2).
- Krzywe uczenia się zbiegają, różnica train–val jest niewielka (~3%) - **nie ma silnego przeuczenia**.
- Filtracja MNE praktycznie nie zmienia wyniku klasyfikacji.
- Wąskim gardłem są **dane, nie model** - sygnał jest słaby, a różnice między klasami niewielkie.
  Klasyfikator znajduje sygnał dokładnie tam, gdzie wskazała go EDA.

Wykresy (krzywe uczenia, macierze pomyłek) znajdują się w katalogach `models/` oraz `report_assets/`,
a pełne omówienie w `raport_ep.docx`.

## Instalacja

Projekt korzysta z menedżera pakietów [`uv`](https://docs.astral.sh/uv/) i Pythona ≥ 3.12.

1. Zainstaluj `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Zainstaluj zależności z pliku blokady:

```bash
uv sync
```

3. Aktywuj środowisko wirtualne:

```bash
source .venv/bin/activate
```

## Uruchomienie

Notatniki możesz uruchomić w VS Code albo bezpośrednio przez Jupyter:

```bash
uv run --with jupyter jupyter notebook
```

Sugerowana kolejność: `01_EDA.ipynb` → `02_modelv2.ipynb` (lub wariant `_mne` / `_augment`).
Dane pobiorą się automatycznie z Hugging Face przy pierwszym uruchomieniu.

## Development

Zadania pomocnicze są w `Makefile`:

- `make lint` - sprawdzenie kodu narzędziem **ruff**,
- `make format` - formatowanie kodu (**ruff**),
- `make clean` - usunięcie plików cache (`__pycache__`, `.mypy_cache`, `.ruff_cache`).

## Literatura

- Lawhern et al. (2018), _EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces_.
- Notatki z prac źródłowych: [`papers/khaleghi.md`](papers/khaleghi.md), [`papers/kumari.md`](papers/kumari.md).
- Zbiór danych: David Vivancos, _MindBigData 2022_ (wariant EP, Emotiv EPOC).

## Autorzy

- Piotr Skowroński
- Jakub Pietrasik
- Krzysztof Czuba
