# ============================================
# Aimers: Transformer / Informer(Lite) 전체 스크립트
# - 입력: 28일(기본) → 7일 예측 (SMAPE on original scale)
# - 선택:
#     MODEL_TYPE in {"transformer", "informer"}  (기본: transformer)
#     POOLING in {"last", "expdecay", "horizon"} (기본: last)
#     TRAIN_FIXED_LEN: 28 (권장) / None(28~56 랜덤)
# ============================================

import os
import glob
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# =========================
# 설정
# =========================
LOOKBACK_MIN, LOOKBACK_MAX = 28, 56   # Train: 가변(28~56), Val/Test: 28 고정
PREDICT = 7

BATCH_SIZE = 64
EPOCHS = 120
LR = 5e-4                  # 안정성 위해 낮춤
WEIGHT_DECAY = 1e-4
PATIENCE = 10
TRAIN_FRACTION = 0.85

# 핵심 A/B 스위치
MODEL_TYPE = "transformer"   # {"transformer", "informer"}
POOLING    = "last"          # {"last", "expdecay", "horizon"}
TRAIN_FIXED_LEN = 28         # 28 권장 / None(28~56 랜덤)

USE_STORE_WEIGHT = True
STORE_WEIGHT_DICT = {"담하": 1.5, "미라시아": 1.5}

EMB_DIM_STORE = 16
EMB_DIM_MENU  = 16

# Transformer/Informer 하이퍼
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 3
FFN_HIDDEN  = 256
DROPOUT     = 0.2
# Informer(Lite) distill 설정 (시퀀스 절반 축소 횟수)
INFORMER_DISTILL_STAGES = 1  # 0~2 권장. 28 길이에선 0~1면 충분

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/content/drive/MyDrive/aimers"
DATA_DIR = os.path.join(BASE_DIR, "data")

# 경로
MODEL_DIR = os.path.join(BASE_DIR, f"{MODEL_TYPE}_{POOLING}_model")
LOG_PATH = os.path.join(BASE_DIR, "logs", f"{MODEL_TYPE}_{POOLING}_log.txt")
SUBMIT_PATH = os.path.join(BASE_DIR, f"{MODEL_TYPE}_{POOLING}_submission.csv")

WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model_weights.pt")
STORE_CLASSES_PATH = os.path.join(MODEL_DIR, "le_store_classes.npy")
MENU_CLASSES_PATH  = os.path.join(MODEL_DIR, "le_menu_classes.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_stats.npz")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# =========================
# 유틸
# =========================
def write_log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# =========================
# 포지셔널 인코딩
# =========================
def sinusoidal_position_encoding(T: int, C: int, device=None):
    pe = torch.zeros(T, C, device=device)
    position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2, device=device).float() * (-math.log(10000.0) / C))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, T, C)

# =========================
# 전처리
# =========================
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    if '업체명' not in df.columns or '메뉴' not in df.columns:
        df[['업체명', '메뉴']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

    df['dayofyear'] = df['영업일자'].dt.dayofyear
    df['month']     = df['영업일자'].dt.month
    df['day']       = df['영업일자'].dt.day
    df['week']      = df['영업일자'].dt.isocalendar().week.astype(int)

    df['sin_day']   = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day']   = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['Day_of_month_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['Day_of_month_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['Week_sin']  = np.sin(2 * np.pi * df['week'] / 53)
    df['Week_cos']  = np.cos(2 * np.pi * df['week'] / 53)
    df['is_weekend']= df['영업일자'].dt.weekday.apply(lambda x: int(x >= 5))
    df['weekday']   = df['영업일자'].dt.weekday + 1

    df['매출수량'] = df['매출수량'].clip(lower=0)
    df['매출수량_log'] = np.log1p(df['매출수량'])

    df = df.sort_values(['업체명', '메뉴', '영업일자']).reset_index(drop=True)
    return df

def add_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(['업체명', '메뉴', '영업일자'])

    df['lag7'] = df.groupby(['업체명', '메뉴'])['매출수량'].shift(7)

    df['dow_mean_4w'] = df.groupby(['업체명', '메뉴'])['매출수량'].shift(7).groupby(
        [df['업체명'], df['메뉴']]
    ).rolling(4, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    def _nz_ratio(s: pd.Series):
        total = s.rolling(28, min_periods=1).mean().shift(1)
        nz = s.where(s > 0).rolling(28, min_periods=1).mean().shift(1)
        return nz / (total + 1e-6)
    df['nz_mean_4w_ratio'] = df.groupby(['업체명', '메뉴'])['매출수량'].transform(_nz_ratio)

    def _zero_runlen_grp(x: pd.Series):
        zr, cnt = [], 0
        for v in x.shift(1, fill_value=0):
            if v == 0: cnt += 1
            else: cnt = 0
            zr.append(cnt)
        return pd.Series(zr, index=x.index)
    df['zero_runlen'] = df.groupby(['업체명', '메뉴'])['매출수량'].transform(_zero_runlen_grp)

    ma7_past = df.groupby(['업체명', '메뉴'])['매출수량'].apply(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).reset_index(level=[0,1], drop=True)
    df['weekly_pattern'] = df['매출수량'] - ma7_past

    df['weekly_pattern']  = df['weekly_pattern'].fillna(0).clip(-1000, 1000)
    df['zero_runlen']     = df['zero_runlen'].fillna(0).clip(0, 56)
    df['nz_mean_4w_ratio']= df['nz_mean_4w_ratio'].fillna(0).clip(0, 2)
    for c in ['lag7', 'dow_mean_4w']:
        df[c] = df[c].fillna(0).clip(0, 5000)

    return df

FEATURES_ALL = [
    '매출수량_log',
    'sin_day','cos_day','Month_sin','Month_cos',
    'Day_of_month_sin','Day_of_month_cos',
    'Week_sin','Week_cos','is_weekend','weekday',
    'lag7','dow_mean_4w','nz_mean_4w_ratio','zero_runlen',
    'weekly_pattern'
]

def fit_standardizer(train_df: pd.DataFrame, feature_list):
    X = train_df[feature_list].astype(np.float32).values
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1e-6, sigma)
    return mu, sigma

def apply_standardizer(df: pd.DataFrame, feature_list, mu, sigma):
    X = df[feature_list].astype(np.float32).values
    Xz = (X - mu) / sigma
    df_std = df.copy()
    df_std[feature_list] = Xz
    return df_std

def fit_label_encoders(df: pd.DataFrame):
    le_store = LabelEncoder().fit(df['업체명'])
    le_menu  = LabelEncoder().fit(df['메뉴'])
    return le_store, le_menu

def build_sequences(df: pd.DataFrame, le_store, le_menu, feature_list, fixed_len: int|None=None):
    """
    Train: fixed_len=TRAIN_FIXED_LEN (기본 28), None이면 [28,56] 랜덤
    Val/Test: fixed_len=28
    """
    df = df.copy()
    df['store_idx'] = le_store.transform(df['업체명'])
    df['menu_idx']  = le_menu.transform(df['메뉴'])

    seqs, lengths, stores, menus, y_log, y_raw, store_names = [], [], [], [], [], [], []

    for (store, menu), g in df.groupby(['업체명', '메뉴']):
        g = g.sort_values('영업일자')
        data = g[feature_list].values.astype(np.float32)
        target_log = g['매출수량_log'].values.astype(np.float32)
        target_raw = g['매출수량'].values.astype(np.float32)

        max_start = len(data) - LOOKBACK_MIN - PREDICT + 1
        if max_start <= 0:
            continue

        for i in range(max_start):
            L = int(fixed_len) if fixed_len is not None else np.random.randint(LOOKBACK_MIN, LOOKBACK_MAX + 1)
            if i + L + PREDICT > len(data): break
            seqs.append(data[i:i+L])
            lengths.append(L)
            y_log.append(target_log[i+L:i+L+PREDICT])
            y_raw.append(target_raw[i+L:i+L+PREDICT])
            stores.append(g['store_idx'].iloc[0])
            menus.append(g['menu_idx'].iloc[0])
            store_names.append(store)

    return (
        seqs,
        np.array(lengths),
        np.array(stores, dtype=np.int64),
        np.array(menus, dtype=np.int64),
        np.array(y_log, dtype=np.float32),
        np.array(y_raw, dtype=np.float32),
        np.array(store_names)
    )

# =========================
# Dataset/Collate
# =========================
class GlobalDataset(Dataset):
    def __init__(self, seqs, lengths, stores, menus, y_log, y_raw, store_names=None):
        self.seqs = seqs
        self.lengths = lengths
        self.stores = stores
        self.menus = menus
        self.y_log = y_log
        self.y_raw = y_raw
        self.store_names = store_names
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        item = (self.seqs[idx], self.lengths[idx], self.stores[idx], self.menus[idx],
                self.y_log[idx], self.y_raw[idx])
        if self.store_names is not None:
            return item + (self.store_names[idx],)
        return item

def collate_varlen(batch):
    with_names = (len(batch[0]) == 7)
    seqs, lens, stores, menus, ylog, yraw, names = [], [], [], [], [], [], []
    for b in batch:
        if with_names:
            seq, L, s, m, yl, yr, nm = b
            names.append(nm)
        else:
            seq, L, s, m, yl, yr = b
        seqs.append(torch.tensor(seq, dtype=torch.float32))
        lens.append(int(L))
        stores.append(int(s))
        menus.append(int(m))
        ylog.append(torch.tensor(yl, dtype=torch.float32))
        yraw.append(torch.tensor(yr, dtype=torch.float32))

    lengths = torch.tensor(lens, dtype=torch.long)
    B = len(seqs)
    F = seqs[0].size(1)
    T_max = max(lens)
    X_pad = torch.zeros((B, T_max, F), dtype=torch.float32)
    for i, (seq, L) in enumerate(zip(seqs, lens)):
        X_pad[i, :L, :] = seq

    out = (
        X_pad,
        lengths,
        torch.tensor(stores, dtype=torch.long),
        torch.tensor(menus, dtype=torch.long),
        torch.stack(ylog, dim=0),
        torch.stack(yraw, dim=0),
    )
    return out + (names,) if with_names else out

# =========================
# Loss
# =========================
class SMAPELossMaskedOriginalScale(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
    def forward(self, pred_log, target_log, target_raw, weight=None):
        pred_log = torch.clamp(pred_log, -15.0, 15.0)
        pred = torch.expm1(pred_log)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)

        target = target_raw
        mask = (target > 0).float()

        num = torch.abs(pred - target)
        denom = (torch.abs(pred) + torch.abs(target)).clamp(min=self.eps)
        smape = 2.0 * num / denom

        denom_mask = mask.sum(dim=1).clamp(min=1.0)
        smape_masked = (smape * mask).sum(dim=1) / denom_mask

        if weight is not None:
            smape_masked = smape_masked * weight
        return smape_masked.mean()

# =========================
# 풀링(최근성 강조/대안)
# =========================
class LastTokenPool(nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        B, T, C = x.size()
        idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, C)  # (B,1,C)
        return x.gather(dim=1, index=idx).squeeze(1)                     # (B,C)

class ExpDecayPool1D(nn.Module):
    """
    길이 불변 + 최근성 우대 가중 풀링. alpha 학습 가능(초기 0.08 권장).
    """
    def __init__(self, alpha: float = 0.08):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        t = torch.arange(T, device=x.device).float().unsqueeze(0)       # (1,T)
        w = torch.exp(self.alpha.clamp(1e-4, 1.0) * t)                  # (1,T)
        w = w.expand(B, -1)
        idx = torch.arange(T, device=x.device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).float()                     # (B,T)
        w = w * mask
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w_norm = (w / w_sum).unsqueeze(-1)                              # (B,T,1)
        return (x * w_norm).sum(dim=1)                                  # (B,C)

class HorizonAttnPool1D(nn.Module):
    """
    7개 horizon별 쿼리로 각기 다른 중요 타임스텝을 뽑아 concat. (B, 7*C) 반환
    """
    def __init__(self, d_in: int, horizon: int = 7):
        super().__init__()
        self.h = horizon
        self.query = nn.Parameter(torch.randn(horizon, d_in))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        idx = torch.arange(T, device=x.device).unsqueeze(0)
        valid = (idx < lengths.unsqueeze(1))  # (B,T)
        outs = []
        for h in range(self.h):
            q = self.query[h].view(1, 1, C).expand(B, 1, C)             # (B,1,C)
            score = torch.matmul(q, x.transpose(1,2)).squeeze(1)        # (B,T)
            score = score.masked_fill(~valid, -1e9)
            w = torch.softmax(score, dim=-1).unsqueeze(1)               # (B,1,T)
            pooled = torch.bmm(w, x).squeeze(1)                          # (B,C)
            outs.append(pooled)
        return torch.cat(outs, dim=-1)                                   # (B, 7*C)

def make_pool(d_model: int, pooling: str):
    if pooling == "last":
        return LastTokenPool()
    elif pooling == "expdecay":
        return ExpDecayPool1D(alpha=0.08)
    elif pooling == "horizon":
        return HorizonAttnPool1D(d_in=d_model, horizon=PREDICT)
    else:
        raise ValueError("POOLING must be in {'last','expdecay','horizon'}")

# =========================
# 마스크 유틸
# =========================
def lengths_to_key_padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    # True: pad, False: valid  (nn.Transformer는 pad=True를 마스킹함)
    idx = torch.arange(T, device=lengths.device).unsqueeze(0)
    return ~(idx < lengths.unsqueeze(1))

# =========================
# 모델: Transformer
# =========================
class TransformerTS(nn.Module):
    def __init__(self, input_dim, n_stores, n_menus,
                 d_model=128, n_heads=4, n_layers=3, ffn_hidden=256,
                 dropout=0.2, pooling="last", out_dim=7):
        super().__init__()
        self.store_emb = nn.Embedding(n_stores, EMB_DIM_STORE)
        self.menu_emb  = nn.Embedding(n_menus,  EMB_DIM_MENU)

        in_concat = input_dim + EMB_DIM_STORE + EMB_DIM_MENU
        self.in_proj = nn.Linear(in_concat, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_hidden,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        self.pool = make_pool(d_model, pooling)
        head_in = d_model if pooling in ("last","expdecay") else d_model * PREDICT

        self.head = nn.Sequential(
            nn.Linear(head_in, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, out_dim),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.5)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_pad, lengths, store_idx, menu_idx):
        # x_pad: (B,T,F)
        B, T, F = x_pad.size()
        s_emb = self.store_emb(store_idx).unsqueeze(1).expand(-1, T, -1)
        m_emb = self.menu_emb(menu_idx).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_pad, s_emb, m_emb], dim=-1)

        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
        x = self.in_proj(x)  # (B,T,d_model)

        # Positional Encoding (가산)
        pe = sinusoidal_position_encoding(T, x.size(-1), device=x.device)
        x = x + pe

        key_padding_mask = lengths_to_key_padding_mask(lengths, T)  # (B,T), True=pad
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,T,d_model)
        x = self.dropout(x)

        feat = self.pool(x, lengths)  # (B,d_model) or (B, 7*d_model)
        out = self.head(feat)         # (B,7)
        out = torch.clamp(out, -15.0, 15.0)
        return out

# =========================
# 모델: Informer(Lite)
#  - ProbSparse는 생략(간단화). 대신 Distilling(Conv1d stride=2)로 길이 축소만 적용.
#  - 짧은 L(28)에서는 0~1회 distill만 권장.
# =========================
class ConvDistill(nn.Module):
    def __init__(self, d_model: int, dropout=0.0):
        super().__init__()
        # 시간축 축소: (B,T,d) -> (B,T/2,d)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dp   = nn.Dropout(dropout)
    def forward(self, x):
        # x: (B,T,d)
        x = x.transpose(1,2)               # (B,d,T)
        x = self.conv(x)                   # (B,d,T//2)
        x = x.transpose(1,2)               # (B,T//2,d)
        x = self.norm(x)
        x = self.dp(x)
        return x

def down_lengths(lengths):
    return ((lengths + 1) // 2).clamp(min=1)

class InformerLiteTS(nn.Module):
    def __init__(self, input_dim, n_stores, n_menus,
                 d_model=128, n_heads=4, n_layers=3, ffn_hidden=256,
                 dropout=0.2, pooling="last", out_dim=7, distill_stages=1):
        super().__init__()
        self.store_emb = nn.Embedding(n_stores, EMB_DIM_STORE)
        self.menu_emb  = nn.Embedding(n_menus,  EMB_DIM_MENU)

        in_concat = input_dim + EMB_DIM_STORE + EMB_DIM_MENU
        self.in_proj = nn.Linear(in_concat, d_model)

        self.layers = nn.ModuleList()
        self.distills = nn.ModuleList()
        for i in range(n_layers):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_hidden,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.layers.append(nn.TransformerEncoder(enc_layer, num_layers=1))
            # 일부 층 뒤에 distill 삽입
            if distill_stages > 0 and (i < distill_stages):
                self.distills.append(ConvDistill(d_model, dropout=dropout))
            else:
                self.distills.append(None)

        self.dropout = nn.Dropout(dropout)
        self.pool = make_pool(d_model, pooling)
        head_in = d_model if pooling in ("last","expdecay") else d_model * PREDICT
        self.head = nn.Sequential(
            nn.Linear(head_in, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, out_dim),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.5)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_pad, lengths, store_idx, menu_idx):
        B, T, F = x_pad.size()
        s_emb = self.store_emb(store_idx).unsqueeze(1).expand(-1, T, -1)
        m_emb = self.menu_emb(menu_idx).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_pad, s_emb, m_emb], dim=-1)

        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
        x = self.in_proj(x)

        # 초기 PE
        pe = sinusoidal_position_encoding(T, x.size(-1), device=x.device)
        x = x + pe

        cur_lengths = lengths.clone()
        for enc, dist in zip(self.layers, self.distills):
            key_padding_mask = lengths_to_key_padding_mask(cur_lengths, x.size(1))
            x = enc(x, src_key_padding_mask=key_padding_mask)
            x = self.dropout(x)
            if dist is not None and x.size(1) > 1:
                x = dist(x)
                cur_lengths = down_lengths(cur_lengths)

        feat = self.pool(x, cur_lengths)
        out = self.head(feat)
        out = torch.clamp(out, -15.0, 15.0)
        return out

# =========================
# 학습/평가 루프
# =========================
def batch_store_weights(store_names_batch):
    if (not USE_STORE_WEIGHT) or (store_names_batch is None):
        return None
    ws = [STORE_WEIGHT_DICT.get(str(s), 1.0) for s in store_names_batch]
    return torch.tensor(ws, dtype=torch.float32, device=DEVICE)

def train_epoch(model, loader, criterion, optimizer, clip_val=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        if len(batch) == 7:
            X_pad, lengths, sb, mb, y_log_b, y_raw_b, sname_b = batch
        else:
            X_pad, lengths, sb, mb, y_log_b, y_raw_b = batch
            sname_b = None
        X_pad = torch.nan_to_num(X_pad, nan=0.0, posinf=0.0, neginf=0.0)
        X_pad, lengths = X_pad.to(DEVICE), lengths.to(DEVICE)
        sb, mb = sb.to(DEVICE), mb.to(DEVICE)
        y_log_b, y_raw_b = y_log_b.to(DEVICE), y_raw_b.to(DEVICE)
        w = batch_store_weights(sname_b)

        optimizer.zero_grad()
        preds_log = model(X_pad, lengths, sb, mb)
        loss = criterion(preds_log, y_log_b, y_raw_b, w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

        bs = X_pad.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        if len(batch) == 7:
            X_pad, lengths, sb, mb, y_log_b, y_raw_b, sname_b = batch
        else:
            X_pad, lengths, sb, mb, y_log_b, y_raw_b = batch
            sname_b = None
        X_pad = torch.nan_to_num(X_pad, nan=0.0, posinf=0.0, neginf=0.0)
        X_pad, lengths = X_pad.to(DEVICE), lengths.to(DEVICE)
        sb, mb = sb.to(DEVICE), mb.to(DEVICE)
        y_log_b, y_raw_b = y_log_b.to(DEVICE), y_raw_b.to(DEVICE)
        w = batch_store_weights(sname_b)

        preds_log = model(X_pad, lengths, sb, mb)
        loss = criterion(preds_log, y_log_b, y_raw_b, w)

        bs = X_pad.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)

# =========================
# 추론 보조
# =========================
def fallback_ma(arr, horizon=PREDICT, ma_len=14):
    arr = np.asarray(arr, dtype=np.float32)
    base = float(np.maximum(0.0, np.nanmean(arr[-min(len(arr), ma_len):]))) if len(arr) else 0.0
    return np.full(horizon, base, dtype=np.float32)

# =========================
# 메인
# =========================
if __name__ == "__main__":
    # ---------- 데이터 로드 & 전처리 ----------
    raw = pd.read_csv(os.path.join(DATA_DIR, "train", "train.csv"))
    df_all = data_preprocessing(raw)

    # 시간 컷오프
    cut_date = df_all['영업일자'].quantile(TRAIN_FRACTION)
    train_df = df_all[df_all['영업일자'] <= cut_date].copy()
    val_df   = df_all[df_all['영업일자'] >  cut_date].copy()

    # 강화 피처
    train_df = add_strength_features(train_df)
    val_df   = add_strength_features(val_df)

    # 라벨 인코더 (Train fit)
    le_store, le_menu = fit_label_encoders(train_df)
    np.save(STORE_CLASSES_PATH, le_store.classes_)
    np.save(MENU_CLASSES_PATH,  le_menu.classes_)

    # 표준화 (Train fit → 저장)
    mu, sigma = fit_standardizer(train_df, FEATURES_ALL)
    np.savez(SCALER_PATH, mu=mu, sigma=sigma, features=np.array(FEATURES_ALL, dtype=object))

    # 표준화 적용
    train_df_std = apply_standardizer(train_df, FEATURES_ALL, mu, sigma)
    val_df_std   = apply_standardizer(val_df,   FEATURES_ALL, mu, sigma)

    # 시퀀스 생성
    seqs_tr, lens_tr, s_tr, m_tr, ylog_tr, yraw_tr, sname_tr = build_sequences(
        train_df_std, le_store, le_menu, FEATURES_ALL,
        fixed_len=(TRAIN_FIXED_LEN if TRAIN_FIXED_LEN is not None else None)
    )
    seqs_va, lens_va, s_va, m_va, ylog_va, yraw_va, sname_va = build_sequences(
        val_df_std, le_store, le_menu, FEATURES_ALL, fixed_len=LOOKBACK_MIN
    )

    write_log(f"Train samples: {len(seqs_tr)} / Val samples: {len(seqs_va)}")
    write_log(f"Stores: {len(le_store.classes_)} / Menus: {len(le_menu.classes_)}")
    if len(lens_tr):
        write_log(f"Avg L Train: {np.mean(lens_tr):.1f} | Val L: {np.unique(lens_va).tolist()}")

    # 데이터로더
    train_ds = GlobalDataset(seqs_tr, lens_tr, s_tr, m_tr, ylog_tr, yraw_tr, sname_tr)
    val_ds   = GlobalDataset(seqs_va, lens_va, s_va, m_va, ylog_va, yraw_va, sname_va)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False, collate_fn=collate_varlen)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_varlen)

    # ---------- 모델 ----------
    input_dim = len(FEATURES_ALL)
    if MODEL_TYPE == "transformer":
        model = TransformerTS(
            input_dim=input_dim,
            n_stores=len(le_store.classes_),
            n_menus=len(le_menu.classes_),
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            ffn_hidden=FFN_HIDDEN, dropout=DROPOUT,
            pooling=POOLING, out_dim=PREDICT
        ).to(DEVICE)
    elif MODEL_TYPE == "informer":
        model = InformerLiteTS(
            input_dim=input_dim,
            n_stores=len(le_store.classes_),
            n_menus=len(le_menu.classes_),
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            ffn_hidden=FFN_HIDDEN, dropout=DROPOUT,
            pooling=POOLING, out_dim=PREDICT,
            distill_stages=INFORMER_DISTILL_STAGES
        ).to(DEVICE)
    else:
        raise ValueError("MODEL_TYPE must be in {'transformer','informer'}")

    criterion = SMAPELossMaskedOriginalScale()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_loss = float("inf")
    patience = 0

    # ---------- 학습 ----------
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        write_log(f"Epoch {epoch}/{EPOCHS} Train SMAPE: {train_loss:.4f}  Val SMAPE: {val_loss:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            patience = 0
            write_log(f"✅ 모델 저장 (Val SMAPE: {best_loss:.4f})")
        else:
            patience += 1
            if patience > PATIENCE:
                write_log("⏹️ Early stopping")
                break

    # ---------- 추론 ----------
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    store_classes = np.load(STORE_CLASSES_PATH, allow_pickle=True)
    menu_classes  = np.load(MENU_CLASSES_PATH,  allow_pickle=True)
    le_store = LabelEncoder().fit(store_classes)
    le_menu  = LabelEncoder().fit(menu_classes)

    scal = np.load(SCALER_PATH, allow_pickle=True)
    mu = scal['mu']; sigma = scal['sigma']; feat_order = list(scal['features'])

    test_files = sorted(glob.glob(os.path.join(DATA_DIR, "test", "TEST_*.csv")))
    all_preds = []

    for path in test_files:
        prefix = os.path.basename(path).split(".")[0]  # e.g., TEST_00
        test_df = pd.read_csv(path)
        test_df = data_preprocessing(test_df)
        test_df = add_strength_features(test_df)
        test_df_std = apply_standardizer(test_df, feat_order, mu, sigma)

        # 인코딩
        try:
            test_df_std['store_idx'] = le_store.transform(test_df_std['업체명'])
        except ValueError:
            known = set(le_store.classes_)
            test_df_std['store_idx'] = test_df_std['업체명'].apply(
                lambda x: 0 if x not in known else le_store.transform([x])[0]
            )
        try:
            test_df_std['menu_idx'] = le_menu.transform(test_df_std['메뉴'])
        except ValueError:
            known = set(le_menu.classes_)
            test_df_std['menu_idx'] = test_df_std['메뉴'].apply(
                lambda x: 0 if x not in known else le_menu.transform([x])[0]
            )

        grouped = test_df_std.groupby(['업체명', '메뉴'])
        for (store, menu), gdf in grouped:
            gdf = gdf.sort_values('영업일자')
            feats = gdf[feat_order].values.astype(np.float32)
            s_idx = torch.tensor([int(gdf['store_idx'].iloc[0])], dtype=torch.long).to(DEVICE)
            m_idx = torch.tensor([int(gdf['menu_idx'].iloc[0])], dtype=torch.long).to(DEVICE)

            L = min(len(feats), LOOKBACK_MIN)  # 보통 28
            if L < LOOKBACK_MIN:
                preds_raw = fallback_ma(
                    test_df[(test_df['업체명'].eq(store)) & (test_df['메뉴'].eq(menu))]['매출수량'].values,
                    horizon=PREDICT
                )
            else:
                X_test = torch.tensor(feats[-L:][np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
                lengths = torch.tensor([L], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    preds_log = model(X_test, lengths, s_idx, m_idx).cpu().numpy().flatten()
                preds_raw = np.maximum(0.0, np.expm1(preds_log))

            dates = [f"{prefix}+{i+1}일" for i in range(PREDICT)]
            for d, p in zip(dates, preds_raw):
                all_preds.append({
                    "영업일자": d,
                    "영업장명_메뉴명": f"{store}_{menu}",
                    "매출수량": float(np.round(p, 2))
                })

    # ---------- 제출 ----------
    if all_preds:
        write_log(f"총 예측 결과 개수: {len(all_preds)}")
        pred_df = pd.DataFrame(all_preds)
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        out = sample.copy()

        pred_dict = {(r["영업일자"], r["영업장명_메뉴명"]): r["매출수량"] for _, r in pred_df.iterrows()}
        for idx in out.index:
            date = out.loc[idx, "영업일자"]
            for col in out.columns[1:]:
                out.loc[idx, col] = float(pred_dict.get((date, col), 0.0))

        out.to_csv(SUBMIT_PATH, index=False, encoding="utf-8-sig")
        write_log(f"✅ 제출 완료: {SUBMIT_PATH}")
    else:
        write_log("❌ 예측 결과 없음 → 제출 실패")
