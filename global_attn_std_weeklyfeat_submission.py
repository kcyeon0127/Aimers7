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
LR = 5e-4                 # 안정성 위해 낮춤
WEIGHT_DECAY = 1e-4
PATIENCE = 10
TRAIN_FRACTION = 0.85

USE_STORE_WEIGHT = True
STORE_WEIGHT_DICT = {"담하": 1.5, "미라시아": 1.5}

EMB_DIM_STORE = 16
EMB_DIM_MENU  = 16
D_MODEL       = 128
FF_HIDDEN     = 256
ATTN_HEADS    = 2
ATTN_DROPOUT  = 0.1
FF_DROPOUT    = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/content/drive/MyDrive/aimers"
DATA_DIR = os.path.join(BASE_DIR, "data")

# 경로(겹치지 않게 분리)
MODEL_DIR = os.path.join(BASE_DIR, "global_attn_std_weeklyfeat")
LOG_PATH = os.path.join(BASE_DIR, "logs", "global_attn_std_weeklyfeat_log.txt")
SUBMIT_PATH = os.path.join(BASE_DIR, "global_attn_std_weeklyfeat_submission.csv")

WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model_weights.pt")
STORE_CLASSES_PATH = os.path.join(MODEL_DIR, "le_store_classes.npy")
MENU_CLASSES_PATH  = os.path.join(MODEL_DIR, "le_menu_classes.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_stats.npz")  # mean, std, feature_order 저장

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
# 전처리
# =========================
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """기본 파생 + 정렬 + log1p 타깃."""
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    if '업체명' not in df.columns or '메뉴' not in df.columns:
        df[['업체명', '메뉴']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

    # 달력/주기
    df['dayofyear'] = df['영업일자'].dt.dayofyear
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
    df['week'] = df['영업일자'].dt.isocalendar().week.astype(int)

    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['Day_of_month_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['Day_of_month_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['Week_sin'] = np.sin(2 * np.pi * df['week'] / 53)
    df['Week_cos'] = np.cos(2 * np.pi * df['week'] / 53)
    df['is_weekend'] = df['영업일자'].dt.weekday.apply(lambda x: int(x >= 5))
    df['weekday'] = df['영업일자'].dt.weekday + 1

    # 타깃
    df['매출수량'] = df['매출수량'].clip(lower=0)
    df['매출수량_log'] = np.log1p(df['매출수량'])

    # 정렬
    df = df.sort_values(['업체명', '메뉴', '영업일자']).reset_index(drop=True)
    return df

# -------------------------
# 강화 피처 + weekly_pattern (누수 방지 & 안정화 클립)
# -------------------------
def add_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    그룹(업체명, 메뉴) 단위:
      - lag7: 7일 전 매출 (raw)
      - dow_mean_4w: 같은 요일 직전 4회 평균 (raw)
      - nz_mean_4w_ratio: 최근 4주 비제로 평균 / 전체 평균 (둘 다 shift(1))
      - zero_runlen: 직전 연속 0일 수 (shift(1))
      - weekly_pattern: 매출수량 - 과거7일평균(shift(1).rolling(7))
    """
    df = df.copy().sort_values(['업체명', '메뉴', '영업일자'])

    # lag7
    df['lag7'] = df.groupby(['업체명', '메뉴'])['매출수량'].shift(7)

    # dow_mean_4w
    df['dow_mean_4w'] = df.groupby(['업체명', '메뉴'])['매출수량'].shift(7).groupby(
        [df['업체명'], df['메뉴']]
    ).rolling(4, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # nz_mean_4w_ratio
    def _nz_ratio(s: pd.Series):
        total = s.rolling(28, min_periods=1).mean().shift(1)
        nz = s.where(s > 0).rolling(28, min_periods=1).mean().shift(1)
        return nz / (total + 1e-6)
    df['nz_mean_4w_ratio'] = df.groupby(['업체명', '메뉴'])['매출수량'].transform(_nz_ratio)

    # zero_runlen
    def _zero_runlen_grp(x: pd.Series):
        zr, cnt = [], 0
        for v in x.shift(1, fill_value=0):
            if v == 0: cnt += 1
            else: cnt = 0
            zr.append(cnt)
        return pd.Series(zr, index=x.index)
    df['zero_runlen'] = df.groupby(['업체명', '메뉴'])['매출수량'].transform(_zero_runlen_grp)

    # weekly_pattern
    ma7_past = df.groupby(['업체명', '메뉴'])['매출수량'].apply(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).reset_index(level=[0,1], drop=True)
    df['weekly_pattern'] = df['매출수량'] - ma7_past

    # 안정화 클립
    df['weekly_pattern'] = df['weekly_pattern'].fillna(0).clip(-1000, 1000)
    df['zero_runlen'] = df['zero_runlen'].fillna(0).clip(0, 56)
    df['nz_mean_4w_ratio'] = df['nz_mean_4w_ratio'].fillna(0).clip(0, 2)
    for c in ['lag7', 'dow_mean_4w']:
        df[c] = df[c].fillna(0).clip(0, 5000)

    return df

# -------------------------
# 표준화(Train fit → Val/Test transform)
# -------------------------
FEATURES_ALL = [
    '매출수량_log',  # 1
    'sin_day','cos_day','Month_sin','Month_cos',
    'Day_of_month_sin','Day_of_month_cos',
    'Week_sin','Week_cos','is_weekend','weekday',  # +10=11
    'lag7','dow_mean_4w','nz_mean_4w_ratio','zero_runlen',  # +4=15
    'weekly_pattern'  # +1=16
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

# -------------------------
# 인코더/시퀀스 생성
# -------------------------
def fit_label_encoders(df: pd.DataFrame):
    le_store = LabelEncoder().fit(df['업체명'])
    le_menu  = LabelEncoder().fit(df['메뉴'])
    return le_store, le_menu

def build_sequences(df: pd.DataFrame, le_store, le_menu, feature_list, fixed_len: int|None=None):
    """
    df는 이미 data_preprocessing → add_strength_features → standardize 적용되어 있어야 함.
    Train: fixed_len=None → [28,56) 랜덤
    Val/Test: fixed_len=28
    """
    df = df.copy()
    df['store_idx'] = le_store.transform(df['업체명'])
    df['menu_idx']  = le_menu.transform(df['메뉴'])

    seqs, lengths, stores, menus, y_log, y_raw, store_names = [], [], [], [], [], [], []

    for (store, menu), g in df.groupby(['업체명', '메뉴']):
        g = g.sort_values('영업일자')
        data = g[feature_list].values.astype(np.float32)
        target_log = g['매출수량_log'].values.astype(np.float32)  # 표준화 전 원본 log 타깃
        target_raw = g['매출수량'].values.astype(np.float32)

        max_start = len(data) - LOOKBACK_MIN - PREDICT + 1
        if max_start <= 0:
            continue

        for i in range(max_start):
            L = np.random.randint(LOOKBACK_MIN, LOOKBACK_MAX + 1) if fixed_len is None else int(fixed_len)
            if i + L + PREDICT > len(data):
                break
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
# Loss (안정화)
# =========================
class SMAPELossMaskedOriginalScale(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
    def forward(self, pred_log, target_log, target_raw, weight=None):
        pred_log = torch.clamp(pred_log, -15.0, 15.0)  # 폭주 방지
        pred = torch.expm1(pred_log)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)

        target = target_raw
        mask = (target > 0).float()

        num = torch.abs(pred - target)
        denom = (torch.abs(pred) + torch.abs(target)).clamp(min=self.eps)
        smape = 2.0 * num / denom  # (B,PREDICT)

        denom_mask = mask.sum(dim=1).clamp(min=1.0)
        smape_masked = (smape * mask).sum(dim=1) / denom_mask

        if weight is not None:
            smape_masked = smape_masked * weight

        return smape_masked.mean()

# =========================
# Positional Encoding & Mask
# =========================
def sinusoidal_position_encoding(T: int, E: int, device=None):
    position = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, E, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / E))
    pe = torch.zeros(T, E, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1,T,E)

def make_key_padding_mask(lengths: torch.Tensor, T_max: int):
    idxs = torch.arange(T_max, device=lengths.device).unsqueeze(0).expand(lengths.size(0), -1)
    return (idxs >= lengths.unsqueeze(1)).bool()  # True=pad

def masked_mean_pool(x, lengths):
    B, T, E = x.size()
    mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    mask = mask.unsqueeze(-1)
    x = x * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return x.sum(dim=1) / denom

# =========================
# 모델 (MHA 인코더)
# =========================
class AttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=ATTN_HEADS, attn_drop=ATTN_DROPOUT, ff_hidden=FF_HIDDEN, ff_drop=FF_DROPOUT):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=attn_drop)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(ff_drop),
            nn.Linear(ff_hidden, embed_dim),
        )
    def forward(self, x, key_padding_mask=None, pos=None):
        if pos is not None:
            x = x + pos
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        h = self.ln2(x)
        h = self.ff(h)
        x = x + h
        return x

class GlobalAttnWithIDEmb(nn.Module):
    """
    입력(캘린더/강화/weekly_pattern) + 업체/메뉴 임베딩 → 선형투영 → PosEnc → MHA → 마스킹 평균풀링 → FC(7 log1p)
    """
    def __init__(self, input_dim, n_stores, n_menus,
                 embed_store=EMB_DIM_STORE, embed_menu=EMB_DIM_MENU,
                 d_model=D_MODEL, output_dim=PREDICT):
        super().__init__()
        self.store_emb = nn.Embedding(n_stores, embed_store)
        self.menu_emb  = nn.Embedding(n_menus, embed_menu)
        self.proj = nn.Linear(input_dim + embed_store + embed_menu, d_model)
        self.block = AttnBlock(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.5)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_pad, lengths, store_idx, menu_idx):
        B, T, _ = x_pad.size()
        s_emb = self.store_emb(store_idx).unsqueeze(1).expand(-1, T, -1)
        m_emb = self.menu_emb(menu_idx).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_pad, s_emb, m_emb], dim=-1)
        x = self.proj(x)

        pos = sinusoidal_position_encoding(T, x.size(-1), device=x.device)
        kpm = make_key_padding_mask(lengths, T)

        x = self.block(x, key_padding_mask=kpm, pos=pos)
        pooled = masked_mean_pool(x, lengths)
        out = self.head(pooled)
        out = torch.clamp(out, -15.0, 15.0)  # 안전 클램프
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
    # ---------- 데이터 로드 & 기본 전처리 ----------
    raw = pd.read_csv(os.path.join(DATA_DIR, "train", "train.csv"))
    df_all = data_preprocessing(raw)

    # 시간 컷오프
    cut_date = df_all['영업일자'].quantile(TRAIN_FRACTION)
    train_df = df_all[df_all['영업일자'] <= cut_date].copy()
    val_df   = df_all[df_all['영업일자'] >  cut_date].copy()

    # 강화 피처
    train_df = add_strength_features(train_df)
    val_df   = add_strength_features(val_df)

    # 라벨 인코더 (Train에서 fit)
    le_store, le_menu = fit_label_encoders(train_df)
    np.save(STORE_CLASSES_PATH, le_store.classes_)
    np.save(MENU_CLASSES_PATH,  le_menu.classes_)

    # 표준화 통계 (Train에서 fit) → 저장
    mu, sigma = fit_standardizer(train_df, FEATURES_ALL)
    np.savez(SCALER_PATH, mu=mu, sigma=sigma, features=np.array(FEATURES_ALL, dtype=object))

    # 표준화 적용
    train_df_std = apply_standardizer(train_df, FEATURES_ALL, mu, sigma)
    val_df_std   = apply_standardizer(val_df,   FEATURES_ALL, mu, sigma)

    # 시퀀스 생성: Train 가변, Val 28 고정
    seqs_tr, lens_tr, s_tr, m_tr, ylog_tr, yraw_tr, sname_tr = build_sequences(
        train_df_std, le_store, le_menu, FEATURES_ALL, fixed_len=None
    )
    seqs_va, lens_va, s_va, m_va, ylog_va, yraw_va, sname_va = build_sequences(
        val_df_std, le_store, le_menu, FEATURES_ALL, fixed_len=LOOKBACK_MIN
    )

    write_log(f"Train samples: {len(seqs_tr)} / Val samples: {len(seqs_va)}")
    write_log(f"Stores: {len(le_store.classes_)} / Menus: {len(le_menu.classes_)}")
    if len(lens_tr):
        write_log(f"Avg L Train: {np.mean(lens_tr):.1f} | Val L: {np.unique(lens_va).tolist()}")

    # 데이터셋/로더
    train_ds = GlobalDataset(seqs_tr, lens_tr, s_tr, m_tr, ylog_tr, yraw_tr, sname_tr)
    val_ds   = GlobalDataset(seqs_va, lens_va, s_va, m_va, ylog_va, yraw_va, sname_va)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_varlen)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_varlen)

    # 모델
    input_dim = len(FEATURES_ALL)  # 16
    model = GlobalAttnWithIDEmb(
        input_dim=input_dim,
        n_stores=len(le_store.classes_),
        n_menus=len(le_menu.classes_),
        embed_store=EMB_DIM_STORE,
        embed_menu=EMB_DIM_MENU,
        d_model=D_MODEL,
        output_dim=PREDICT,
    ).to(DEVICE)

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
    # 가중치/인코더/스케일러 복원
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

        # 표준화(Train 통계 적용)
        test_df_std = apply_standardizer(test_df, feat_order, mu, sigma)

        # 인코딩(미지 라벨 방어적으로 처리)
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

        # 그룹별 추론
        grouped = test_df_std.groupby(['업체명', '메뉴'])
        for (store, menu), gdf in grouped:
            gdf = gdf.sort_values('영업일자')
            feats = gdf[feat_order].values.astype(np.float32)
            s_idx = torch.tensor([int(gdf['store_idx'].iloc[0])], dtype=torch.long).to(DEVICE)
            m_idx = torch.tensor([int(gdf['menu_idx'].iloc[0])], dtype=torch.long).to(DEVICE)

            # 테스트는 보통 28일 → 고정 28
            L = min(len(feats), LOOKBACK_MIN)
            if L < LOOKBACK_MIN:
                preds_raw = fallback_ma(test_df[test_df['업체명'].eq(store) & test_df['메뉴'].eq(menu)]['매출수량'].values, horizon=PREDICT)
            else:
                X_test = torch.tensor(feats[-L:][np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
                lengths = torch.tensor([L], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    preds_log = model(X_test, lengths, s_idx, m_idx).cpu().numpy().flatten()
                preds_raw = np.maximum(0.0, np.expm1(preds_log))

            # 제출용 날짜 키
            dates = [f"{prefix}+{i+1}일" for i in range(PREDICT)]
            for d, p in zip(dates, preds_raw):
                all_preds.append({
                    "영업일자": d,
                    "영업장명_메뉴명": f"{store}_{menu}",
                    "매출수량": float(np.round(p, 2))
                })

    # ---------- 제출 생성 ----------
    if all_preds:x`
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
