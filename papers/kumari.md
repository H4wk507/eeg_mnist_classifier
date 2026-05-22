Surowy sygnał EEG z 14-kanałowego EPOC, konwertowany na obrazy spektrogramów za pomocą STFT.
Te obrazy trafiaja do wejścia CNN.

(B, 14, 256) -> STFT per kanał -> (B, 14, H, W)

4 warstwy CNN 2D, max pooling, FC layer
