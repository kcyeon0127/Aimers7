import os
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# =========================
# 설정
# =========================
# 가변 윈도 사용: 각 샘플 길이를 28~56 사이에서 랜덤 선택
LOOKBACK_MIN, LOOKBACK_MAX = 28, 56
PREDICT = 7

BATCH_SIZE = 64
EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
TRAIN_FRACTION = 0.85  # 시간 컷오프 분할(Train 비율)

USE_STORE_WEIGHT = True   # 담하/미라시아 가중치 반영 여부
STORE_WEIGHT_DICT = {
    "담하": 1.5,
    "미라시아": 1.5,
}

EMB_DIM_STORE = 16
EMB_DIM_MENU  = 16
HIDDEN_DIM    = 128
DROPOUT_FC    = 0.3
BIDIR         = True      # 양방향 GRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/content/drive/MyDrive/aimers"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "global_gru_menuemb")
LOG_PATH = os.path.join(BASE_DIR, "logs", "global_gru_menuemb_log.txt")
SUBMIT_PATH = os.path.join(BASE_DIR, "global_gru_menuemb_56_submission.csv")  # 파일명은 그대로 유지

# 분리 저장 경로
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model_weights.pt")
STORE_CLASSES_PATH = os.path.join(MODEL_DIR, "le_store_classes.npy")
MENU_CLASSES_PATH  = os.path.join(MODEL_DIR, "le_menu_classes.npy")

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# =========================
# 전처리
# =========================
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """기본 파생변수 + 정렬(업체명, 메뉴, 일자). log1p 타깃 추가."""
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    if '업체명' not in df.columns or '메뉴' not in df.columns:
        df[['업체명', '메뉴']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

    # 달력/주기 피처
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

    # 타깃: 원시 수량(음수 클립) + log1p
    df['매출수량'] = df['매출수량'].clip(lower=0)
    df['매출수량_log'] = np.log1p(df['매출수량'])

    # 그룹 내 정렬
    df = df.sort_values(['업체명', '메뉴', '영업일자']).reset_index(drop=True)
    return df

def create_features(df: pd.DataFrame, le_store=None, le_menu=None, fit_le: bool=False):
    """
    X는 시간/캘린더 + 매출수량_log(첫 채널).
    업체/메뉴는 임베딩 인덱스로만 사용.
    시퀀스 길이는 [LOOKBACK_MIN, LOOKBACK_MAX]에서 샘플마다 랜덤 선택(훈련/검증 공통).
    """
    df = data_preprocessing(df)

    # 인코더 준비
    if fit_le:
        le_store = LabelEncoder().fit(df['업체명'])
        le_menu  = LabelEncoder().fit(df['메뉴'])

    df['store_idx'] = le_store.transform(df['업체명'])
    df['menu_idx']  = le_menu.transform(df['메뉴'])

    features = [
        '매출수량_log',                           # ← 최근 레벨/추세
        'sin_day','cos_day','Month_sin','Month_cos',
        'Day_of_month_sin','Day_of_month_cos',
        'Week_sin','Week_cos','is_weekend','weekday'
    ]

    seqs, lengths, stores, menus, labels_log, labels_raw, store_names = [], [], [], [], [], [], []
    for (store, menu), g in df.groupby(['업체명', '메뉴']):
        g = g.sort_values('영업일자')
        data = g[features].values
        target_log = g['매출수량_log'].values
        target_raw = g['매출수량'].values

        max_start = len(data) - LOOKBACK_MIN - PREDICT + 1
        if max_start <= 0:
            continue

        for i in range(max_start):
            # 매 샘플마다 길이 L을 28~56 사이에서 랜덤 선택
            L = np.random.randint(LOOKBACK_MIN, LOOKBACK_MAX + 1)
            if i + L + PREDICT > len(data):
                break

            seqs.append(data[i:i+L])
            lengths.append(L)
            labels_log.append(target_log[i+L:i+L+PREDICT])
            labels_raw.append(target_raw[i+L:i+L+PREDICT])
            stores.append(g['store_idx'].iloc[0])
            menus.append(g['menu_idx'].iloc[0])
            store_names.append(store)

    # 리스트 그대로 두고 collate_fn에서 패딩 처리
    stores = np.array(stores, dtype=np.int64)
    menus  = np.array(menus, dtype=np.int64)
    store_names = np.array(store_names)

    return seqs, np.array(lengths), stores, menus, np.array(labels_log, dtype=np.float32), \
           np.array(labels_raw, dtype=np.float32), le_store, le_menu, store_names

# =========================
# 데이터셋/모델/로스
# =========================
class GlobalDataset(Dataset):
    """가변 길이 시퀀스: 내부는 리스트로 보관, DataLoader에서 패딩."""
    def __init__(self, seqs, lengths, stores, menus, y_log, y_raw, store_names=None):
        self.seqs = seqs
        self.lengths = lengths
        self.stores = stores
        self.menus = menus
        self.y_log = y_log
        self.y_raw = y_raw
        self.store_names = store_names  # numpy array of strings or None
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        item = (self.seqs[idx], self.lengths[idx], self.stores[idx], self.menus[idx],
                self.y_log[idx], self.y_raw[idx])
        if self.store_names is not None:
            return item + (self.store_names[idx],)
        return item

def collate_varlen(batch):
    """
    배치 내 다양한 길이의 시퀀스를 패딩하고 길이를 텐서로 반환.
    반환:
      X_pad: (B, T_max, F), lengths: (B,), store_idx: (B,), menu_idx: (B,), y_log/y_raw: (B,PREDICT)
      sname(옵션): 리스트[str] 또는 None
    """
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
    if with_names:
        return out + (names,)
    return out

class SMAPELossMaskedOriginalScale(nn.Module):
    """
    예측/타깃은 log1p 텐서이지만, 손실(SMAPE)은 expm1로 원 스케일 변환해 계산.
    실제 수량이 0인 타깃은 마스킹(대회 평가 정합).
    선택적으로 샘플 가중치(weight) 반영.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
    def forward(self, pred_log, target_log, target_raw, weight=None):
        pred = torch.expm1(pred_log).clamp(min=0.0)
        target = target_raw
        mask = (target > 0).float()

        num = torch.abs(pred - target)
        denom = (torch.abs(pred) + torch.abs(target)).clamp(min=self.eps)
        smape = 2.0 * num / denom  # (B, PREDICT)

        denom_mask = mask.sum(dim=1).clamp(min=1.0)
        smape_masked = (smape * mask).sum(dim=1) / denom_mask  # (B,)

        if weight is not None:
            smape_masked = smape_masked * weight

        return smape_masked.mean()

class GlobalGRUWithIDEmb(nn.Module):
    """
    캘린더 피처 + (업체/메뉴 임베딩)을 GRU 입력 채널에 결합.
    가변 길이 입력은 pack_padded_sequence로 패딩 무시.
    출력은 log1p 스케일의 7일 벡터.
    """
    def __init__(self, input_dim, n_stores, n_menus,
                 embed_store=EMB_DIM_STORE, embed_menu=EMB_DIM_MENU,
                 hidden_dim=HIDDEN_DIM, output_dim=PREDICT, dropout_fc=DROPOUT_FC,
                 bidirectional=BIDIR):
        super().__init__()
        self.store_emb = nn.Embedding(n_stores, embed_store)
        self.menu_emb  = nn.Embedding(n_menus, embed_menu)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        in_dim = input_dim + embed_store + embed_menu
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        fc_in = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(256, output_dim),
        )

    def forward(self, x_pad, lengths, store_idx, menu_idx):
        """
        x_pad: (B, T_max, F), lengths: (B,)
        """
        B, T_max, F = x_pad.size()
        s_emb = self.store_emb(store_idx).unsqueeze(1).expand(-1, T_max, -1)  # (B, T_max, Es)
        m_emb  = self.menu_emb(menu_idx).unsqueeze(1).expand(-1, T_max, -1)  # (B, T_max, Em)
        x = torch.cat([x_pad, s_emb, m_emb], dim=-1)                          # (B, T_max, F+Es+Em)

        # pack -> GRU -> (h_n 사용)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)  # h_n: (num_directions, B, H)
        # 양방향이면 [fw_last; bw_last] concat
        if self.bidirectional:
            fw_last = h_n[-2]  # (B, H)
            bw_last = h_n[-1]  # (B, H)
            last = torch.cat([fw_last, bw_last], dim=1)  # (B, 2H)
        else:
            last = h_n[-1]  # (B, H)

        out = self.fc(last)  # (B, PREDICT) in log1p scale
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
# 추론 보조(짧은 그룹 백업)
# =========================
def fallback_ma(arr, horizon=PREDICT, ma_len=14):
    """최근 ma_len일 이동평균으로 horizon 길이 예측."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        base = 0.0
    else:
        base = float(np.maximum(0.0, np.nanmean(arr[-min(len(arr), ma_len):])))
    return np.full(horizon, base, dtype=np.float32)

# =========================
# 메인
# =========================
if __name__ == "__main__":
    # ---------- 학습/검증 분리(시간 기준) ----------
    raw = pd.read_csv(os.path.join(DATA_DIR, "train", "train.csv"))
    df_all = data_preprocessing(raw)

    # 시간 컷오프: 상위 TRAIN_FRACTION 날짜까지 train, 이후 val
    cut_date = df_all['영업일자'].quantile(TRAIN_FRACTION)
    train_df = df_all[df_all['영업일자'] <= cut_date].copy()
    val_df   = df_all[df_all['영업일자'] >  cut_date].copy()

    # 인코더 학습은 train에서
    (_, _, _, _, _, _, le_store, le_menu, _) = create_features(train_df, fit_le=True)
    seqs_tr, lens_tr, s_tr, m_tr, ylog_tr, yraw_tr, _, _, sname_tr = create_features(train_df, le_store, le_menu, fit_le=False)
    seqs_va, lens_va, s_va, m_va, ylog_va, yraw_va, _, _, sname_va = create_features(val_df,   le_store, le_menu, fit_le=False)

    write_log(f"Train samples: {len(seqs_tr)} / Val samples: {len(seqs_va)}")
    write_log(f"Stores: {len(le_store.classes_)} / Menus: {len(le_menu.classes_)}")

    # 데이터셋/로더 (가변 길이 collate)
    train_data = GlobalDataset(seqs_tr, lens_tr, s_tr, m_tr, ylog_tr, yraw_tr, sname_tr)
    val_data   = GlobalDataset(seqs_va, lens_va, s_va, m_va, ylog_va, yraw_va, sname_va)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_varlen)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_varlen)

    # 모델 준비
    # input_dim = 피처 수 = 1(매출수량_log) + 10(달력/주기) = 11
    input_dim = 11
    model = GlobalGRUWithIDEmb(
        input_dim=input_dim,
        n_stores=len(le_store.classes_),
        n_menus=len(le_menu.classes_),
        embed_store=EMB_DIM_STORE,
        embed_menu=EMB_DIM_MENU,
        hidden_dim=HIDDEN_DIM,
        output_dim=PREDICT,
        dropout_fc=DROPOUT_FC,
        bidirectional=BIDIR,
    ).to(DEVICE)

    criterion = SMAPELossMaskedOriginalScale()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_loss = float("inf")
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        write_log(f"Epoch {epoch}/{EPOCHS} Train SMAPE: {train_loss:.4f}  Val SMAPE: {val_loss:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            np.save(STORE_CLASSES_PATH, le_store.classes_)
            np.save(MENU_CLASSES_PATH,  le_menu.classes_)
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

    # 인코더 복원
    store_classes = np.load(STORE_CLASSES_PATH, allow_pickle=True)
    menu_classes  = np.load(MENU_CLASSES_PATH,  allow_pickle=True)
    le_store = LabelEncoder().fit(store_classes)
    le_menu  = LabelEncoder().fit(menu_classes)

    test_files = sorted(glob.glob(os.path.join(DATA_DIR, "test", "TEST_*.csv")))
    all_preds = []

    features_inf = [
        '매출수량_log',  # ← 학습과 동일하게 첫 채널
        'sin_day','cos_day','Month_sin','Month_cos',
        'Day_of_month_sin','Day_of_month_cos',
        'Week_sin','Week_cos','is_weekend','weekday'
    ]

    for path in test_files:
        prefix = os.path.basename(path).split(".")[0]  # e.g., TEST_00
        test_df = pd.read_csv(path)
        test_df = data_preprocessing(test_df)

        # 인코딩(미지 라벨 방어적으로 처리)
        try:
            test_df['store_idx'] = le_store.transform(test_df['업체명'])
        except ValueError:
            known = set(le_store.classes_)
            test_df['store_idx'] = test_df['업체명'].apply(
                lambda x: 0 if x not in known else le_store.transform([x])[0]
            )
        try:
            test_df['menu_idx'] = le_menu.transform(test_df['메뉴'])
        except ValueError:
            known = set(le_menu.classes_)
            test_df['menu_idx'] = test_df['메뉴'].apply(
                lambda x: 0 if x not in known else le_menu.transform([x])[0]
            )

        grouped = test_df.groupby(['업체명', '메뉴'])
        for (store, menu), gdf in grouped:
            gdf = gdf.sort_values('영업일자')
            feats = gdf[features_inf].values
            s_idx = torch.tensor([int(gdf['store_idx'].iloc[0])], dtype=torch.long).to(DEVICE)
            m_idx = torch.tensor([int(gdf['menu_idx'].iloc[0])], dtype=torch.long).to(DEVICE)

            # 사용 가능한 길이(L): 보통 28일(테스트 고정), 최대 56까지 허용
            L = min(len(feats), LOOKBACK_MAX)
            if L < LOOKBACK_MIN:
                # 데이터가 너무 짧으면 백업(MA) 사용
                preds_raw = fallback_ma(gdf['매출수량'].values, horizon=PREDICT)
            else:
                X_test = torch.tensor(feats[-L:][np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
                lengths = torch.tensor([L], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    preds_log = model(X_test, lengths, s_idx, m_idx).cpu().numpy().flatten()
                preds_raw = np.maximum(0.0, np.expm1(preds_log))

            # 제출용 날짜 키 생성
            dates = [f"{prefix}+{i+1}일" for i in range(PREDICT)]
            for d, p in zip(dates, preds_raw):
                all_preds.append({
                    "영업일자": d,
                    "영업장명_메뉴명": f"{store}_{menu}",
                    "매출수량": float(np.round(p, 2))
                })

    # ---------- 제출 생성 ----------
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
