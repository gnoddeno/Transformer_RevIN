:
#!/usr/bin/env python
# power_forecast.py  (K-Fold, 14-day history, 2-day decoder, 1-day prediction)

# ─────────────────────────────────────────────────────────────
# 0. model.py 의 'TotalEmbedding' 의존성 더미 주입 (수정 없이 사용)
# ─────────────────────────────────────────────────────────────
import sys, types, math, argparse, random
import torch, torch.nn as nn
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset, DataLoader
from torch.nn import L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional
from torch.utils.data import ConcatDataset, Subset
from weather import weather_api
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
# === ADD BEGIN (신규 import) ======================================
import os


# === ADD END ======================================================

class _DummyTotalEmbedding(nn.Module):
    def __init__(self, d_model, enc_in, dec_in, dropout):
        super().__init__()
        self.proj = nn.Linear(enc_in + dec_in, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x): return self.drop(self.proj(x))


_mod_root = types.ModuleType("models")
_mod_tfms = types.ModuleType("models.tranformers")  # 오탈자(tranformers) 그대로
_mod_file = types.ModuleType("models.tranformers.transformer")
_mod_file.TotalEmbedding = _DummyTotalEmbedding
sys.modules.update({
    "models": _mod_root,
    "models.tranformers": _mod_tfms,
    "models.tranformers.transformer": _mod_file,
})

from model import TransformerRevIN  # 이제 오류 없이 import


# ════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════
class PowerDataset(Dataset):
    """7 일(history) → 24 h 예측용 시계열 Dataset"""

    def __init__(self, df, scaler: StandardScaler,
                 win=168, label_len=24, pred_len=24, fit=False):
        self.win, self.label_len, self.pred_len = win, label_len, pred_len
        df = df.copy()

        df["mrdDt"] = pd.to_datetime(df["mrdDt"], format="%Y-%m-%d %H")
        h = df["mrdDt"].dt.hour.astype(float)
        df["sin_h"] = np.sin(2 * math.pi * h / 24)
        df["cos_h"] = np.cos(2 * math.pi * h / 24)

        feats = ["pwrQrt", "temperature", "precipitation",
                 "windspeed", "humidity", "sin_h", "cos_h"]
        df[feats] = df[feats].apply(pd.to_numeric, errors="coerce") \
            .interpolate(method="linear", limit_direction="both") \
            .fillna(0.0)
        vals = df[feats].values
        if fit: scaler.fit(vals)
        vals = scaler.transform(vals);
        vals[np.isinf(vals)] = 0.0

        self.vals = torch.tensor(vals, dtype=torch.float32)
        self.times = df["mrdDt"].to_numpy()  # 타임스탬프 저장
        self.len = len(df) - win - pred_len + 1

    def __len__(self): return self.len

    def __getitem__(self, idx):
        s, e, p = idx, idx + self.win, idx + self.win + self.pred_len
        enc = self.vals[s:e]
        dec = self.vals[e - self.label_len:p].clone()
        dec[self.label_len:, 0] = 0.
        tgt = self.vals[e:p, 0]
        return enc, dec, tgt

# ════════════════════════════════════════════
# Utils
# ════════════════════════════════════════════
def causal_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)


def run_epoch(model, loader, crit, opt=None, mask=None, device="cpu", desc: Optional[str] = None):
    train = opt is not None;
    model.train() if train else model.eval()
    tot, n = 0., 0
    iterable = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for enc, dec, tgt in loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            out = model(enc, dec, tgt_mask=mask)
            if out.size(1) != tgt.size(1): out = out[:, -tgt.size(1):]
            loss = crit(out, tgt)
            if train:
                opt.zero_grad();
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
                opt.step()
            tot += loss.item() * tgt.numel();
            n += tgt.numel()
            iterable.set_postfix(loss=loss.item())
    return tot / n


def eval_epoch_real(model, loader, scaler, mask, device="cpu"):
    """평가용: 예측 결과를 원 스케일로 복원하여 MAE 계산"""
    model.eval()
    total_err, total_n = 0.0, 0
    mean0, std0 = scaler.mean_[0], scaler.scale_[0]

    with torch.no_grad():
        for enc, dec, tgt in loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            out = model(enc, dec, tgt_mask=mask)
            if out.size(1) != tgt.size(1):
                out = out[:, -tgt.size(1):]

            # 원 스케일로 복원
            out_KW = out.cpu().numpy() * std0 + mean0
            tgt_KW = tgt.cpu().numpy() * std0 + mean0

            # MAE 계산
            total_err += np.abs(out_KW - tgt_KW).sum()
            total_n += tgt_KW.size

    return total_err / total_n


def preview_batch(model, loader, mask, scaler, device="cpu"):
    ds = loader.dataset
    win, pred_len = ds.win, ds.pred_len
    mean, std = scaler.mean_, scaler.scale_

    model.eval()
    with torch.no_grad():
        enc, dec, tgt = next(iter(loader))
        enc, dec = enc.to(device), dec.to(device)
        out = model(enc, dec, tgt_mask=mask)
        if out.size(1) != tgt.size(1):
            out = out[:, -tgt.size(1):]

        pred = (out.cpu() * std[0] + mean[0]).numpy()[0]  # pwrQrt
        true = (tgt * std[0] + mean[0]).numpy()[0]

        # 원 스케일의 추가 피처 (dataset 저장된 스케일 값 복원)
        block = ds.vals[win: win + pred_len].numpy()  # [24, 7]
        temp = block[:, 1] * std[1] + mean[1]
        prec = block[:, 2] * std[2] + mean[2]
        wind = block[:, 3] * std[3] + mean[3]
        humi = block[:, 4] * std[4] + mean[4]

        times = ds.times[win: win + pred_len]
        lines = []
        for t, p, r, tmp, pr, wd, hm in zip(
                times, pred, true, temp, prec, wind, humi):
            lines.append(
                f"{pd.to_datetime(t).strftime('%Y-%m-%d %H:%M')}  "
                f"pred={p:.4f} | true={r:.4f} | "
                f"T={tmp:.1f}°C  Pcp={pr:.1f}mm  W={wd:.1f}m/s  H={hm:.0f}%")
        return "\n".join(lines)


def safe_tensor(t):
    """NaN/Inf가 있는 텐서를 0으로 대체"""
    if torch.isnan(t).any() or torch.isinf(t).any():
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


# === ADD BEGIN : 무작위 블록 예측 & 시각화 =========================
@torch.no_grad()
# ─────────────────────────────────────────────────────────────
#  추가 기능: 랜덤 블록 예측을 위한 predict_blocks 함수
# ─────────────────────────────────────────────────────────────
def predict_blocks(model, df_sorted, scaler,
                   win, label_len, pred_len,
                   n_blocks=5, device="cpu", seed=42):
    """
    model     : 학습된 TransformerRevIN
    df_sorted : mrdDt 기준 정렬된 DataFrame (sin_h, cos_h 포함)
    scaler    : StandardScaler (fit은 train set에서만)
    win       : 히스토리 윈도우 길이 (예: 168)
    label_len : 디코더 입력으로 쓰는 과거 타깃 길이 (예: 24)
    pred_len  : 예측 길이 (예: 24)
    n_blocks  : 랜덤하게 뽑을 블록 수
    """
    # 1) 필수 파생 피처
    if "sin_h" not in df_sorted.columns or "cos_h" not in df_sorted.columns:
        h = pd.to_datetime(df_sorted["mrdDt"]).dt.hour.astype(float)
        df_sorted["sin_h"] = np.sin(2 * np.pi * h / 24)
        df_sorted["cos_h"] = np.cos(2 * np.pi * h / 24)

    dates = df_sorted["date"].unique()
    hist_days = win // 24
    if len(dates) < hist_days + 1:
        print("[Warn] 데이터가 부족해서 예측을 종료합니다.")
        return

    # 2) 랜덤 블록 시작일 샘플링
    random.seed(seed)
    max_start = len(dates) - (hist_days + 1)
    starts = random.sample(range(0, max_start + 1),
                           k=min(n_blocks, max_start + 1))

    feats = ["pwrQrt", "temperature", "precipitation",
             "windspeed", "humidity", "sin_h", "cos_h"]
    mean0, std0 = scaler.mean_[0], scaler.scale_[0]

    print(f"\n===== predict_blocks: {n_blocks}개 랜덤 블록 예측 =====")
    for no, s in enumerate(starts, 1):
        tgt_date = dates[s + hist_days]
        # target 날짜의 첫 row index
        idxs = df_sorted.index[df_sorted["date"] == tgt_date]
        if len(idxs) == 0:
            print(f"[Block {no:02d}] {tgt_date}  → 날짜 미발견, 스킵")
            continue
        first_idx = idxs[0]
        start_row = first_idx - win
        end_row = first_idx + pred_len

        # 범위 체크
        if start_row < 0 or end_row > len(df_sorted):
            print(f"[Block {no:02d}] {tgt_date}  → 범위 초과, 스킵")
            continue

        # 인코더/디코더 입력 준비
        enc_df = df_sorted.iloc[start_row: first_idx]
        dec_df = df_sorted.iloc[first_idx - label_len: end_row]

        # 스케일링 & NaN/Inf 처리
        enc_np = scaler.transform(enc_df[feats])
        dec_np = scaler.transform(dec_df[feats])
        enc_np = np.nan_to_num(enc_np, nan=0.0, posinf=0.0, neginf=0.0)
        dec_np = np.nan_to_num(dec_np, nan=0.0, posinf=0.0, neginf=0.0)

        # 미래 타깃 0 마스킹
        dec_np[label_len:, 0] = 0.0

        # 텐서 변환
        enc = torch.tensor(enc_np, dtype=torch.float32).unsqueeze(0).to(device)
        dec = torch.tensor(dec_np, dtype=torch.float32).unsqueeze(0).to(device)

        # 예측
        with torch.no_grad():
            out = model(enc, dec,
                        tgt_mask=causal_mask(label_len + pred_len, device))
            # 마지막 pred_len 시점만
            pred_seq = out[0, -pred_len:]  # shape: [pred_len]

        # 만약 NaN 섞여 있으면 스킵
        if torch.isnan(pred_seq).any():
            print(f"[Block {no:02d}] {tgt_date}  → NaN 예측, 스킵")
            continue

        # 역스케일링
        pred = (pred_seq.cpu().numpy() * std0) + mean0
        true = df_sorted.iloc[first_idx: first_idx + pred_len]["pwrQrt"].values

        # 출력 & 시각화
        ts = pd.date_range(f"{tgt_date} 00:00", periods=pred_len, freq="1H")
        mae = np.mean(np.abs(pred - true))
        print(f"\n[Block {no:02d}] {tgt_date}  MAE={mae:.4f}")
        for t, p, y in zip(ts, pred, true):
            print(f"{t:%Y-%m-%d %H:%M}  pred={p:.4f} | true={y:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(ts, true, label="True", linewidth=2)
        plt.plot(ts, pred, label="Pred", linewidth=2)
        plt.title(f"{tgt_date}  MAE={mae:.3f}")
        plt.ylim(0, 2)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()


# === ADD END ======================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=False, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--kfold", type=int, default=2)
    # === ADD BEGIN (새 인자) ======================================
    p.add_argument("--save_dir", type=Path, default=Path("saved_models"))
    p.add_argument("--predict_only", action="store_true")
    p.add_argument("--member_id", default=None)
    p.add_argument("--n_pred_blocks", type=int, default=5)
    p.add_argument("--pretrain_folder", type=Path, default=None,
                   help="여기에 지정된 폴더 안의 모든 CSV로 pre-training")
    p.add_argument("--pretrain_epochs", type=int, default=10,
                   help="pre-training 시킬 epoch 수")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--finetune", type=Path, default=None)
    p.add_argument("--tomorrow", type=bool, default=False)
    p.add_argument("--yesterday", action="store_true")
    # === ADD END ==================================================
    return p.parse_args()


# (eval_random_10_blocks 함수는 그대로 ― 생략)
# ─────────────────────────────────────────────────────────────
# 최근 10개(=80일) 블록: 7일 history → 1일(24h) 예측 & MAE 계산
# ─────────────────────────────────────────────────────────────
def eval_random_10_blocks(model, df_sorted, scaler,
                          win=168, label_len=24, pred_len=24,
                          device="cpu", seed=42):
    # 0) 필수 파생 피처
    if "sin_h" not in df_sorted.columns or "cos_h" not in df_sorted.columns:
        h = pd.to_datetime(df_sorted["mrdDt"]).dt.hour.astype(float)
        df_sorted["sin_h"] = np.sin(2 * np.pi * h / 24)
        df_sorted["cos_h"] = np.cos(2 * np.pi * h / 24)

    dates = df_sorted["date"].unique()
    if len(dates) < 8:
        print("[Warn] 8 일 이하 데이터 — 평가 생략");
        return

    # 무작위 10개 시작점
    random.seed(seed)
    starts = random.sample(range(0, len(dates) - 8 + 1),
                           k=min(10, len(dates) - 7))

    feats = ["pwrQrt", "temperature", "precipitation",
             "windspeed", "humidity", "sin_h", "cos_h"]
    mean0, std0 = scaler.mean_[0], scaler.scale_[0]

    print("\n===== 무작위 10블록(7+1일) 예측 =====")
    agg_abs_err, agg_n = 0.0, 0
    plot_data = []  # 예측 결과 저장용
    for no, s in enumerate(starts, 1):
        tgt_date = dates[s + 7]
        start_row = df_sorted.index[df_sorted["date"] == tgt_date][0] - win
        if start_row < 0 or start_row + win + pred_len > len(df_sorted):
            print(f"[Block {no:02d}] {tgt_date}  → 범위 초과, 스킵")
            continue

        enc_df = df_sorted.iloc[start_row: start_row + win]
        dec_df = df_sorted.iloc[start_row + win - label_len: start_row + win + pred_len].copy()

        # 스케일링 + NaN/Inf 정리
        enc_np = np.nan_to_num(scaler.transform(enc_df[feats]),
                               nan=0.0, posinf=0.0, neginf=0.0)
        dec_np = np.nan_to_num(scaler.transform(dec_df[feats]),
                               nan=0.0, posinf=0.0, neginf=0.0)
        dec_np[label_len:, 0] = 0.0

        enc = torch.tensor(enc_np, dtype=torch.float32).unsqueeze(0).to(device)
        dec = torch.tensor(dec_np, dtype=torch.float32).unsqueeze(0).to(device)
        tgt_raw = df_sorted.iloc[start_row + win: start_row + win + pred_len]["pwrQrt"].values
        tgt = torch.tensor((tgt_raw - mean0) / std0, dtype=torch.float32)

        with torch.no_grad():
            out = model(enc, dec,
                        tgt_mask=causal_mask(label_len + pred_len, device))
            out = out.squeeze(0)[-pred_len:]
        if torch.isnan(out).any():
            print(f"[Block {no:02d}] {tgt_date}  → NaN 발생, 스킵")
            continue

        pred = (out.cpu() * std0 + mean0).numpy()
        true = (tgt * std0 + mean0).numpy()
        blk_mae = np.mean(np.abs(pred - true))
        agg_abs_err += np.abs(pred - true).sum();
        agg_n += pred_len

        feat_rows = df_sorted.iloc[start_row + win: start_row + win + pred_len]
        temp = feat_rows["temperature"].values
        prec = feat_rows["precipitation"].values
        wind = feat_rows["windspeed"].values
        humi = feat_rows["humidity"].values
        print(f"\n[Block {no:02d}] {tgt_date}  MAE={blk_mae:.4f}")
        times = pd.date_range(f"{tgt_date} 00:00", periods=24, freq="1H")
        plot_data.append({
            "block": no, "times": times, "pred": pred, "true": true})

        for ts, p, v, tmp, pr, wd, hm in zip(
                times, pred, true, temp, prec, wind, humi):
            print(f"{ts.strftime('%Y-%m-%d %H:%M')}  "
                  f"pred={p:.4f} | true={v:.4f} | "
                  f"T={tmp:.1f}°C   Pcp={pr:.1f}mm  W={wd:.1f}m/s  H={hm:.0f}%")

    for item in plot_data:
        plt.figure(figsize=(8, 6))
        plt.plot(item["times"], item["true"], label="True", linewidth=2)
        plt.plot(item["times"], item["pred"], label="Pred", linewidth=2)
        plt.ylim(0, 3)
        plt.title(f"Block {item['block']:02d} – {item['times'][0].date()}  (7일→1일)")
        plt.xlabel("Time");
        plt.ylabel("pwrQrt (kW)")
        plt.xticks(rotation=45);
        plt.legend();
        plt.tight_layout()
        plt.show()
    # ──────────────────────────────────────────
    if agg_n:
        print(f"\n--> 10블록 평균 MAE = {agg_abs_err / agg_n:.4f}\n")
    if agg_n:
        print(f"\n--> 10블록 평균 MAE = {agg_abs_err / agg_n:.4f}\n")
    else:
        print("\n→ 10개 블록 모두 NaN 으로 스킵되어 평균 MAE 불가\n")


# ──────────────────────────────────────────
# Pre-training on folder
# ──────────────────────────────────────────
def pretrain_on_folder(folder: Path, args):
    """
    폴더 안의 모든 CSV를 읽어서 합친 뒤에 한 번에 학습(pre-training)하고
    'pretraining.pt' 로 저장합니다.
    """
    from pathlib import Path
    feats = ["pwrQrt", "temperature", "precipitation",
             "windspeed", "humidity", "sin_h", "cos_h"]
    # 1) 폴더 내 CSV 파일 목록
    csvs = sorted(Path(folder).glob("*.csv"))
    if not csvs:
        print(f"[Warn] '{folder}' 안에 CSV 파일을 찾을 수 없습니다.")
        return

    dfs = []
    for f in csvs:  # csv 하나 = 가구 하나라는 전제
        df0 = pd.read_csv(f)
        df0["mrdDt"] = pd.to_datetime(df0["mrdDt"], format="%Y-%m-%d %H")

        # ── ① 30 % 연속 샘플링 ───────────────────────────────────────
        keep_n = int(len(df0) * 0.30)
        start = random.randint(0, len(df0) - keep_n)
        df0 = df0.iloc[start:start + keep_n]  # 연속 구간 선택

        # ── ② 가구 내부 시간순 정렬 ───────────────────────────────
        df0 = df0.sort_values("mrdDt").reset_index(drop=True)

        # ── ③ 파생 피처 추가 ──────────────────────────────────────
        if "sin_h" not in df0.columns or "cos_h" not in df0.columns:
            h = df0["mrdDt"].dt.hour.astype(float)
            df0["sin_h"] = np.sin(2 * np.pi * h / 24)
            df0["cos_h"] = np.cos(2 * np.pi * h / 24)

        dfs.append(df0)

    # ── ④ 가구들 이어붙이기(가구 순서 그대로) ───────────────────────
    df_all = pd.concat(dfs, ignore_index=True)

    # ── ⑤ 스케일러 학습 ────────────────────────────────────────────
    scaler = StandardScaler().fit(
        df_all[feats].apply(pd.to_numeric, errors="coerce")
    )
    print(f"pre-treining: {len(csvs)} 가구")
    # 2) ── 가구(파일)별로 Dataset 만들기  ────────────────────────
    ds_list = []
    for f in csvs:
        df0 = pd.read_csv(f)
        df0["mrdDt"] = pd.to_datetime(df0["mrdDt"], format="%Y-%m-%d %H")
        df0 = df0.sort_values("mrdDt").reset_index(drop=True)
        ds_list.append(
            PowerDataset(df0, scaler,
                         win=args.win, label_len=args.label,
                         pred_len=args.pred, fit=False)
        )

    full_ds = ConcatDataset(ds_list)
    dl = DataLoader(full_ds,
                    batch_size=args.batch,
                    shuffle=True,
                    num_workers=os.cpu_count(),
                    pin_memory=True)

    # 4) 모델 초기화
    model = TransformerRevIN(d_model=160,
                             input_features_count=7,
                             num_encoder_layers=3,
                             num_decoder_layers=3,
                             dim_feedforward=160,
                             kernel_size=3,
                             dropout=0.1,
                             attention_heads=8).to(args.device)
    model.relu = nn.Identity()
    print(f"Model: {model.__class__.__name__}")
    crit = L1Loss()
    opt = AdamW(model.parameters(), lr=5e-4, eps=1e-9)
    mask = causal_mask(args.label + args.pred, args.device)

    # 5) Pre-training 루프
    print(f"\n=== Pre-training on folder '{folder}' for {args.pretrain_epochs} epochs ===")
    for ep in range(1, args.pretrain_epochs + 1):
        loss = run_epoch(model, dl, crit, opt, mask, device=args.device, desc=f"PT{ep}/ {args.pretrain_epochs}")
        print(f"[PT {ep:03d}] train(norm)={loss:.4f}")

    # 6) 가중치 & 스케일러 저장
    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / "pretraining.pt"
    torch.save({
        "weights": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_
    }, ckpt)
    print(f"\n✅ Pre-training complete — saved to {ckpt}\n")


# ──────────────────────────────────────────

# Predict from CSV

def predict_from_csv(
        csv_path: Path,
        ckpt_path: Path,
        n_blocks: int = 5,
        device: str = "cpu"):
    """
    저장된 .pt 와 CSV 하나를 받아서 predict_blocks 만 수행한다.
    """
    if not ckpt_path.exists():
        sys.exit(f"❌ 모델 파일을 찾을 수 없습니다 → {ckpt_path}")
    if not csv_path.exists():
        sys.exit(f"❌ CSV 파일을 찾을 수 없습니다 → {csv_path}")

    # 1) 모델·스케일러 로드 --------------------------------------------------
    state = torch.load(ckpt_path, map_location=device)
    model = TransformerRevIN(
        d_model=160, input_features_count=7,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=160, kernel_size=3,
        dropout=0.1, attention_heads=8
    ).to(device)
    model.load_state_dict(state["weights"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = state["scaler_mean"], state["scaler_scale"]

    # 2) CSV 로드 & 전처리 ----------------------------------------------------
    df = pd.read_csv(csv_path)
    df["mrdDt"] = pd.to_datetime(df["mrdDt"], format="%Y-%m-%d %H")
    df = df.sort_values("mrdDt").reset_index(drop=True)
    df["date"] = df["mrdDt"].dt.date  # predict_blocks 호환 필드

    # 3) 예측 ---------------------------------------------------------------
    hist_days = 28
    win = hist_days * 24
    label_len = 7 * 24
    pred_len = 24
    predict_blocks(model, df, scaler,
                   win, label_len, pred_len,
                   n_blocks=n_blocks,
                   device=device)


# ──────────────────────────────────────────
# 내일 전력 사용량 예측
# ──────────────────────────────────────────
def predict_tomorrow(member_id: str,
                     hist_csv: Path,
                     model_path: Path,
                     device: str = "cpu"):
    tomorrow_weather = weather_api()
    print(type(tomorrow_weather))
    st = torch.load(model_path, map_location=device)
    mdl = TransformerRevIN(d_model=160, input_features_count=7,
                           num_encoder_layers=3, num_decoder_layers=3,
                           dim_feedforward=160, kernel_size=3,
                           dropout=0.05, attention_heads=8).to(device)
    mdl.load_state_dict(st["weights"]);
    mdl.eval()

    scl = StandardScaler()
    scl.mean_, scl.scale_ = st["scaler_mean"], st["scaler_scale"]

    # ③ 과거 CSV
    df_all = pd.read_csv(hist_csv)

    # ④ 예측 (그래프 자동 출력)
    return forecast_next_day_by_house(mdl, df_all, member_id,
                                      tomorrow_weather, scl, device=device)


# ──────────────────────────────────────────
# forecast_next_day_by_house
# ──────────────────────────────────────────
def predict_tomorrow(member_id: str,
                     hist_csv: Path,
                     model_path: Path,
                     device: str = "cpu"):
    tomorrow_weather = weather_api()
    print(type(tomorrow_weather))
    st = torch.load(model_path, map_location=device)
    mdl = TransformerRevIN(d_model=160, input_features_count=7,
                           num_encoder_layers=3, num_decoder_layers=3,
                           dim_feedforward=160, kernel_size=3,
                           dropout=0.05, attention_heads=8).to(device)
    mdl.load_state_dict(st["weights"]);
    mdl.eval()

    scl = StandardScaler()
    scl.mean_, scl.scale_ = st["scaler_mean"], st["scaler_scale"]

    # ③ 과거 CSV
    df_all = pd.read_csv(hist_csv)

    # ④ 예측 (그래프 자동 출력)
    return forecast_next_day_by_house(mdl, df_all, member_id,
                                      tomorrow_weather, scl, device=device, hist_csv_path=hist_csv)


# ──────────────────────────────────────────
# forecast_next_day_by_house
# ──────────────────────────────────────────
def forecast_next_day_by_house(
        model: TransformerRevIN,
        df_all: pd.DataFrame,
        member_id: str,
        tomorrow_weather: pd.DataFrame,
        scaler: StandardScaler,
        device: str = "cpu",
        plot: bool = True,
        hist_csv_path: Optional[Path] = None
):
    # --- 0) memberID 컬럼 탐색 (대·소문자 무시) --------------------
    col = next((c for c in df_all.columns if c.lower() == "memberid"), None)
    if col is None:
        print("❌  CSV 에 memberID 열이 없습니다.");
        return None

    # --- 1) 문자열 전처리 : strip + zero-padding -------------------
    df_all[col] = df_all[col].astype(str).str.strip()
    mid = str(member_id).strip()

    # (가장 흔한) 앞자리 0 누락 대비 – 길이를 맞춰 패딩
    max_len = max(df_all[col].str.len().max(), len(mid))
    df_all[col] = df_all[col].str.zfill(max_len)
    mid = mid.zfill(max_len)

    # --- 2) 대상 가구 추출 ----------------------------------------
    df = df_all[df_all[col] == mid].copy()
    if df.empty:
        print(f"❌  '{member_id}' 에 해당하는 데이터가 없습니다.");
        return None

    # --- 3) 시간 순 정렬 & 인코더/디코더 구성 ----------------------
    df["mrdDt"] = pd.to_datetime(df["mrdDt"])
    df = df.sort_values("mrdDt").reset_index(drop=True)

    last_ts = df["mrdDt"].iloc[-1]  # ← 이 시점부터는 안전
    next_date = (last_ts + pd.Timedelta(days=1)).date()

    # ─── 1) Encoder / Decoder 입력 만들기 ────────────────────────
    enc_df = df.iloc[-win:]  # 28일
    dec_hist = enc_df.iloc[-label:]  # 7일

    # enc_df 시간 파생 피처 보강
    h = enc_df["mrdDt"].dt.hour.astype(float)
    enc_df.loc[:, "sin_h"] = np.sin(2 * np.pi * h / 24)
    enc_df.loc[:, "cos_h"] = np.cos(2 * np.pi * h / 24)

    # ─── 내일 날씨 → feature 세트 맞추기 ──────────────────────────
    tw = tomorrow_weather.copy()
    tw.rename(columns={"Forecast_time": "mrdDt"}, inplace=True)
    tw["mrdDt"] = pd.to_datetime(tw["mrdDt"])
    tw.loc[:, "sin_h"] = np.sin(2 * np.pi * tw["mrdDt"].dt.hour / 24)
    tw.loc[:, "cos_h"] = np.cos(2 * np.pi * tw["mrdDt"].dt.hour / 24)
    tw["pwrQrt"] = 0.0  # 미래 target 마스킹

    feats = ["pwrQrt", "temperature", "precipitation",
             "windspeed", "humidity", "sin_h", "cos_h"]

    # ★ NaN/Inf → 0.0 치환 ★
    enc_np = np.nan_to_num(
        scaler.transform(enc_df[feats]),
        nan=0.0, posinf=0.0, neginf=0.0)

    dec_np = np.nan_to_num(
        scaler.transform(pd.concat([dec_hist, tw])[feats]),
        nan=0.0, posinf=0.0, neginf=0.0)

    enc = torch.tensor(enc_np, dtype=torch.float32).unsqueeze(0).to(device)
    dec = torch.tensor(dec_np, dtype=torch.float32).unsqueeze(0).to(device)

    mask = causal_mask(label + 24, device)
    with torch.no_grad():
        out = model(enc, dec, tgt_mask=mask)[0, -24:]
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)  # 안전망

    pred_kw = out.cpu().numpy() * scaler.scale_[0] + scaler.mean_[0]
    idx = pd.date_range(next_date, periods=24, freq="1H")
    series = pd.Series(pred_kw, index=idx, name="pred_pwrQrt")

    original_cols = df_all.columns.tolist()
    ts_tsr = series.index.strftime("%Y-%m-%d %H")
    out_df = pd.DataFrame({
        "mrdDt": ts_tsr,
        "memberID": mid,
        "pwrQrt": series.values,
        "temperature": tw["temperature"].values,
        "precipitation": tw["precipitation"].values,
        "windspeed": tw["windspeed"].values,
        "humidity": tw["humidity"].values
    })
    out_df = out_df[original_cols]
    if hist_csv_path is not None:
        out_df.to_csv(hist_csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        print(f"✅ 예측 결과가 '{hist_csv_path}' 에 저장되었습니다.")
    else:
        print("❗️ CSV 저장 경로가 지정되지 않았습니다. 예측 결과는 출력되지 않습니다.")

    # ─── 2) 시각화 ────────────────────────────────────────────────
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(series.index, series.values, marker='o',
                 label="Predicted pwrQrt (kW)")

        # ▶▶ 여기부터 새 코드 ──────────────────────────────────
        # 24 h 눈금(1 h 간격) & HH:MM 포맷
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.set_xlim(series.index[0], series.index[-1])  # 범위를 딱 하루로 고정
        # ─────────────────────────────────────────────────────

        ax1.set_ylabel("kW");
        ax1.set_xlabel("Time")
        ax1.set_title(f"{member_id} – {series.index[0].date()} power forecast")
        ax1.grid(True)

        # (온도 보조축은 그대로)
        if "temperature" in tomorrow_weather.columns:
            ax2 = ax1.twinx()
            ax2.plot(tw["mrdDt"], tw["temperature"],
                     color="tab:red", alpha=.4, label="Temperature (°C)")
            ax2.set_ylabel("°C", color="tab:red")
            ax2.tick_params(axis='y', labelcolor="tab:red")

        fig.tight_layout()
        plt.show()

    return series

# ─────────────────────────────────────────────────────────────
#  예측 ↔ 실측 ① 오차 계산 ② 예측 행 제거 ③ CSV 갱신
# ─────────────────────────────────────────────────────────────
def reconcile_day_duplicates(csv_path: Path,
                             member_id: str) -> float:
    """
    · 마지막 입력 행의 날짜(y_date)를 기준으로
      그 날(00~23시) 안에서
        - 중복(mrdDt)인 모든 예측·실측 쌍을 찾고
        - 맨 앞 = 예측, 맨 뒤 = 실측으로 가정
        - abs_err를 계산·출력
        - 예측행(앞부분) 전부 삭제, 실측만 남김
    · 반환 : 해당 날짜 MAE  (예측이 없는 시각은 제외)
    """
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if c.lower() == "memberid"), None)
    if col is None:
        print("❌ memberID 컬럼이 없습니다."); return 0.0

    df[col] = df[col].astype(str).str.strip()
    member_id = str(member_id).strip()
    df["mrdDt"] = pd.to_datetime(df["mrdDt"])

    # ── 대상 가구 · 마지막 입력 날짜 ─────────────────────────────
    dfi = df[df[col] == member_id]
    if dfi.empty:
        print(f"❌ {member_id} 데이터가 없습니다."); return 0.0

    y_date = dfi.iloc[-1]["mrdDt"].date()        # 마지막 행 날짜
    day_mask = dfi["mrdDt"].dt.date == y_date
    dfd = dfi[day_mask].copy()

    if dfd.empty:
        print(f"⚠️  {y_date} 날짜 데이터가 없습니다."); return 0.0

    abs_err, drop_idx = [], []

    # ── 시각별 예측·실측 비교 ──────────────────────────────────
    for ts, grp in dfd.groupby("mrdDt"):
        if len(grp) < 2:              # 예측만 있거나 실측만 1개 → 스킵
            continue

        pred_row = grp.iloc[0]
        real_row = grp.iloc[-1]

        err = abs(pred_row["pwrQrt"] - real_row["pwrQrt"])
        abs_err.append(err)

        print(f"[{ts:%Y-%m-%d %H:%M}] "
              f"pred={pred_row['pwrQrt']:.4f} | "
              f"real={real_row['pwrQrt']:.4f} → abs_err={err:.4f}")

        # 예측행·중간행 전부 삭제 → 실측(마지막) 한 행만 남김
        drop_idx.extend(grp.index[:-1])

    # ── CSV 갱신 ────────────────────────────────────────────────
    if drop_idx:
        df.drop(index=drop_idx, inplace=True)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"📝 예측 등 {len(drop_idx)}개 행 삭제 후 CSV 저장 완료")

    mae = np.mean(abs_err) if abs_err else 0.0
    print(f"\n🌙 {y_date} 전체 MAE = {mae:.4f} kW")
    return mae


# ════════════════════════════════════════════
# Main (8-day block K-Fold)  ※ 당신 코드 그대로
# ════════════════════════════════════════════
historty_days = 28
win = historty_days * 24  # 28일 history
decoder_days = 7  # 디코더 context 14일
label = decoder_days * 24  # 14일(336h) 레이블
pred = 24  # 1일(24h) 예측
block_days = historty_days + 1  # 14일 + 1일


def main():
    args = parse_args();
    print("device =", args.device)

    if args.yesterday:
        if args.csv is None:
            sys.exit("--yesterday 모드에는 --csv 인자가 필요합니다.")

        df_tmp = pd.read_csv(args.csv)
        member_id = (df_tmp.get("memberID",
                                pd.Series(["unknown"]))
        .iloc[0])
        reconcile_day_duplicates(args.csv, member_id)
        return

    if args.tomorrow:
        if args.csv is None or args.model is None:
            sys.exit("❌ --tomorrow 모드에서는 --csv 와 --model 인자가 필요합니다.")
        if not args.csv.exists():
            sys.exit(f"❌ CSV 파일을 찾을 수 없습니다 → {args.csv}")
        if not args.model.exists():
            sys.exit(f"❌ 모델 파일을 찾을 수 없습니다 → {args.model}")
        # 내일 예측
        series = predict_tomorrow(
            member_id=args.member_id or "unknown",
            hist_csv=args.csv, model_path=args.model,
            device=args.device)
        print(series)
        return

    if args.pretrain_folder is not None:
        args.win = win
        args.label = label
        args.pred = pred
        pretrain_on_folder(
            args.pretrain_folder,
            args
        )
        return
    if args.model and args.csv is not None:
        # 예측 전용 모드: 저장된 모델로 CSV 예측
        predict_from_csv(
            args.csv, args.model, n_blocks=args.n_pred_blocks,
            device=args.device)
        return

    if args.finetune and args.csv is not None:
        # 파인튜닝 전용 모드: 저장된 모델로 CSV 파인튜닝
        if not args.finetune.exists():
            sys.exit(f"저장된 모델이 없습니다 → {args.finetune}")

        state = torch.load(args.finetune, map_location=args.device)
        pretrained_mean = state["scaler_mean"]
        pretrained_scale = state["scaler_scale"]

        model = TransformerRevIN(
            d_model=160, input_features_count=7,
            num_encoder_layers=3, num_decoder_layers=3,
            dim_feedforward=160, kernel_size=3,
            dropout=0.1, attention_heads=8
        ).to(args.device)
        model.load_state_dict(state["weights"])
        model.relu = nn.Identity()  # ReLU 제거

    df = pd.read_csv(args.csv)
    df["mrdDt"] = pd.to_datetime(df["mrdDt"], format="%Y-%m-%d %H")
    df = df.sort_values("mrdDt").reset_index(drop=True)
    df["date"] = df["mrdDt"].dt.date
    member_id = df.get("memberID", pd.Series(["unknown"])).iloc[0]

    # === ADD BEGIN : predict_only 빠른 탈출 ======================
    if args.predict_only:
        ckpt = args.save_dir / f"{args.member_id or member_id}.pt"
        if not ckpt.exists():
            sys.exit(f"저장된 모델이 없습니다 → {ckpt}")
        state = torch.load(ckpt, map_location=args.device)
        model = TransformerRevIN(d_model=160, input_features_count=7,
                                 num_encoder_layers=3, num_decoder_layers=3,
                                 dim_feedforward=160, kernel_size=3,
                                 dropout=0.1, attention_heads=8).to(args.device)
        model.load_state_dict(state["weights"])
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = state["scaler_mean"], state["scaler_scale"]
        predict_blocks(model, df, scaler,
                       win, label, pred, args.n_pred_blocks, args.device)
        return
    # === ADD END =================================================

    # (아래 학습 루프는 당신 코드 그대로 …)

    dates = df["date"].unique()
    blocks = [dates[i:i + block_days]
              for i in range(0, len(dates), block_days)
              if len(dates[i:i + block_days]) == block_days]

    tscv = TimeSeriesSplit(n_splits=args.kfold, test_size=1)
    fold_results = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(blocks), 1):
        train_dates = np.concatenate([blocks[i] for i in tr_idx])
        val_dates = np.concatenate([blocks[i] for i in val_idx])
        next_block = val_idx[-1] + 1
        if next_block < len(blocks):
            test_dates = blocks[next_block]
        else:
            test_dates = blocks[val_idx[-1]]

        tr_df = df[df["date"].isin(train_dates)].drop(columns="date")
        va_df = df[df["date"].isin(val_dates)].drop(columns="date")
        te_df = df[df["date"].isin(test_dates)].drop(columns="date")

        print(f"\n========== Fold {fold}/{args.kfold} ==========")
        print(f"Train {train_dates[0]} ~ {train_dates[-1]}")
        print(f" Val  {val_dates[0]} ~ {val_dates[-1]}")
        print(f" Test {test_dates[0]} ~ {test_dates[-1]}")

        scaler = StandardScaler()
        tr_dl = DataLoader(PowerDataset(tr_df, scaler, win, label, pred, fit=True),
                           batch_size=args.batch, shuffle=True)
        va_dl = DataLoader(PowerDataset(va_df, scaler, win, label, pred),
                           batch_size=args.batch)
        te_dl = DataLoader(PowerDataset(te_df, scaler, win, label, pred),
                           batch_size=args.batch)

        model = TransformerRevIN(d_model=160, input_features_count=7,
                                 num_encoder_layers=3, num_decoder_layers=3,
                                 dim_feedforward=160, kernel_size=3,
                                 dropout=0.05, attention_heads=8).to(args.device)
        model.relu = nn.Identity()

        crit = L1Loss();
        opt = AdamW(model.parameters(), lr=5e-4, eps=1e-9)
        sch = StepLR(opt, step_size=2, gamma=0.1)
        mask = causal_mask(label + pred, args.device)

        best, pat = 1e9, 0
        for ep in range(1, args.epochs + 1):
            tr = run_epoch(model, tr_dl, crit, opt, mask, args.device)
            va = run_epoch(model, va_dl, crit, None, mask, args.device)
            sch.step()
            va_real = eval_epoch_real(model, va_dl, scaler, mask, args.device)
            print(f"[{ep:03d}] train(norm)={tr:.4f}  val(norm)={va:.4f}  val(kW)={va_real:.4f}")
            print(preview_batch(model, va_dl, mask, scaler, args.device), "\n")
            if va < best - 1e-4:
                best, best_state, pat = va, model.state_dict(), 0
            else:
                pat += 1
                if pat > 8: print("Early stop."); break
        model.load_state_dict(best_state)
        test_mae = run_epoch(model, te_dl, crit, None, mask, args.device)
        fold_results.append({"fold": fold, "val": best, "test": test_mae})
        print(f"Fold {fold}  best_val={best:.4f} | test_MAE={test_mae:.4f}")

    print("\n===== K-Fold Summary =====")
    for r in fold_results:
        print(f"Fold {r['fold']}: val={r['val']:.4f} | test={r['test']:.4f}")
    print(f"Avg  val={np.mean([r['val'] for r in fold_results]):.4f}")
    print(f"Avg test={np.mean([r['test'] for r in fold_results]):.4f}")

    # === ADD BEGIN : 모델 저장 ====================================
    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / f"{member_id}.pt"
    torch.save({"weights": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_}, ckpt)
    print(f"\n모델 저장 완료 → {ckpt}")
    # === ADD END ==================================================

    model.load_state_dict(best_state)
    model.eval()

    residuals = []
    with torch.no_grad():
        for enc, dec, tgt in va_dl:
            enc, dec, tgt = enc.to(args.device), dec.to(args.device), tgt.to(args.device)
            out = model(enc, dec, tgt_mask=mask)
            # 디코더 길이 맞추기
            if out.size(1) != tgt.size(1):
                out = out[:, -tgt.size(1):]
            # 원 스케일로 복원
            out = out.cpu().numpy() * scaler.scale_[0] + scaler.mean_[0]
            tgt = tgt.cpu().numpy() * scaler.scale_[0] + scaler.mean_[0]
            residuals.append(np.abs(out - tgt))

    residuals = np.concatenate(residuals).ravel()
    thr_97 = np.percentile(residuals, 97)
    print(f"\n99% 잔차 임계값: {thr_97:.4f} kW")
    '''
    eval_random_10_blocks(model, df, scaler,
                      win, label, pred,
                      device=args.device)
                      '''
    print(f"\n97% 잔차 임계값: {thr_97:.4f} kW")
    # 추가: 즉시 예측 샘플 확인
    predict_blocks(model, df, scaler,
                   win, label, pred,
                   n_blocks=15, device=args.device)



if __name__ == "__main__":
    main()