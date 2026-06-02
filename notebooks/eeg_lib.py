"""Wspólna logika dla 02_model.ipynb — preprocessing, cechy, modele, trening.

Wariant EP (MindBigData2022_MNIST_EP): 14 kanałów, 128 Hz, 256 próbek (2 s), ~52k trialów.
Połączenie doświadczeń z dwóch projektów:
- z projektu EP: surowy sygnał -> 1D CNN, tSNE na embeddingach,
- z raportu Cap64: porządny preprocessing (bandpass/detrend/baseline/odrzucanie artefaktów),
  klasyczny baseline (RandomForest na bandpower+Hjorth), EEGNet (Lawhern), uczciwa ewaluacja.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, detrend, filtfilt, welch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

FS = 128  # częstotliwość próbkowania EP [Hz]
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 40)]


# --------------------------------------------------------------------------- #
#  1. Preprocessing (kolejność jak w raporcie Cap64)
# --------------------------------------------------------------------------- #
def df_to_raw(df):
    """DataFrame -> (X[N,14,256] float32, y[N] int64). Odrzuca label == -1."""
    df = df[df["label"] != -1]
    y = df["label"].values.astype(np.int64)
    X = df.drop("label", axis=1).values.astype(np.float32).reshape(-1, 14, 256)
    return X, y


def filter_signal(X, fs=FS, band=(1.0, 40.0), order=4, baseline_ms=100.0):
    """detrend liniowy -> bandpass zero-phase (Butterworth) -> baseline correction.

    Notch 50 Hz pominięty świadomie: bandpass do 40 Hz już wycina sieciówkę.
    Wszystko per trial per kanał, wektorowo po osi czasu. Zwraca nową tablicę.
    """
    X = detrend(X, axis=2, type="linear")
    nyq = fs / 2.0
    b, a = butter(order, [band[0] / nyq, band[1] / nyq], btype="band")
    X = filtfilt(b, a, X, axis=2)
    n_base = max(1, int(baseline_ms / 1000.0 * fs))  # ~13 próbek przy 128 Hz
    X = X - X[:, :, :n_base].mean(axis=2, keepdims=True)
    return X.astype(np.float32)


def reject_threshold(X_train, k=8.0):
    """Próg peak-to-peak per kanał: mediana + k * MAD (liczone TYLKO na train)."""
    ptp = X_train.max(axis=2) - X_train.min(axis=2)  # (N, 14)
    med = np.median(ptp, axis=0)
    mad = np.median(np.abs(ptp - med), axis=0) + 1e-8
    return med + k * mad  # (14,)


def reject_mask(X, ptp_thresh, flat_eps=1e-6):
    """True = zachowaj trial. Odrzuca, gdy któryś kanał > próg p2p albo flatline."""
    ptp = X.max(axis=2) - X.min(axis=2)  # (N, 14)
    std = X.std(axis=2)  # (N, 14)
    over = (ptp > ptp_thresh).any(axis=1)
    flat = (std < flat_eps).any(axis=1)
    return ~(over | flat)


def zscore_fit(X_train):
    """z-score per kanał — statystyki z train (brak wycieku z val/test)."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1,14,1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std


def zscore_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------- #
#  2. Cechy klasyczne (RandomForest baseline) — bandpower + Hjorth
# --------------------------------------------------------------------------- #
def hjorth(X):
    """Parametry Hjortha per trial per kanał: activity, mobility, complexity."""
    d1 = np.diff(X, axis=2)
    d2 = np.diff(d1, axis=2)
    v0 = X.var(axis=2) + 1e-12
    v1 = d1.var(axis=2) + 1e-12
    v2 = d2.var(axis=2) + 1e-12
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return activity, mobility, complexity


def bandpower_hjorth_features(X, fs=FS):
    """(N,14,256) -> (N, 14*8). 5 log-mocy pasmowych (Welch) + 3 Hjorth na kanał."""
    f, P = welch(X, fs=fs, nperseg=min(128, X.shape[2]), axis=2)  # P: (N,14,Fbins)
    feats = []
    for _, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        bp = P[:, :, m].sum(axis=2)  # (N,14)
        feats.append(np.log(bp + 1e-12))
    feats.extend(hjorth(X))
    return np.concatenate([fdim.reshape(len(X), -1) for fdim in feats], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
#  3. Modele
# --------------------------------------------------------------------------- #
class EEGNet1D(nn.Module):
    """1D CNN: kanały EEG jako kanały wejściowe Conv1d (podejście z projektu EP)."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(14, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def embed(self, x):
        return nn.Flatten()(self.features(x))  # 128-D (do tSNE)


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018) — kompaktowy CNN dedykowany EEG."""

    def __init__(self, n_ch=14, n_time=256, n_classes=10, F1=8, D=2, F2=16, kern=64, p=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, n_ch, n_time)).flatten(1).shape[1]
        self.classify = nn.Linear(flat, n_classes)

    def forward(self, x):
        return self.classify(self.embed(x))

    def embed(self, x):
        return self.features(x.unsqueeze(1)).flatten(1)


# --------------------------------------------------------------------------- #
#  4. Augmentacja (label-preserving, na GPU) — wykorzystuje skalę zbioru EP
# --------------------------------------------------------------------------- #
def augment(x, max_shift=15, jitter=0.1, ch_drop=0.1):
    """Time-shift (cykliczny) + szum gaussowski + channel dropout. x: (B,14,T)."""
    B, C, T = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    idx = (torch.arange(T, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % T
    x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, C, -1))
    x = x + torch.randn_like(x) * jitter
    mask = (torch.rand(B, C, 1, device=x.device) > ch_drop).float()
    return x * mask


# --------------------------------------------------------------------------- #
#  5. Trening + ewaluacja
# --------------------------------------------------------------------------- #
def make_loader(X, y, batch=128, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return p, accuracy_score(t, p), f1_score(t, p, average="macro")


def train_model(model, train_loader, val_loader, device, *, epochs=25, lr=1e-3,
                weight_decay=1e-4, patience=5, time_budget_s=300, use_augment=True,
                log_every=0):
    """Adam + cosine LR + early stopping na val macro-F1 + twardy limit czasu.

    Zwraca (model z najlepszymi wagami, history). history: train_loss/train_acc/val_acc/val_f1.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, waited = -1.0, None, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_augment:
                xb = augment(xb)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(xb)
            run_correct += (out.argmax(1) == yb).sum().item()
            run_n += len(xb)
        sched.step()

        _, val_acc, val_f1 = evaluate(model, val_loader, device)
        hist["train_loss"].append(run_loss / run_n)
        hist["train_acc"].append(run_correct / run_n)
        hist["val_acc"].append(val_acc)
        hist["val_f1"].append(val_f1)
        if log_every and (ep % log_every == 0 or ep == 1):
            print(f"  ep {ep:02d}  train_acc={hist['train_acc'][-1]:.3f}  "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, waited = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop @ epoch {ep} (best val_f1={best_f1:.3f})")
                break
        if time.time() - t0 > time_budget_s:
            print(f"  time budget hit @ epoch {ep} ({time.time()-t0:.0f}s)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist
