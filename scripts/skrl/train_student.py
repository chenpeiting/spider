"""
train_student.py — 觀測歷史版
Student 吃過去 N 幀本體感知，隱式推斷地形

用法：
  python scripts/skrl/train_student.py \
    --data teacher_data.pt \
    --history 10 \
    --epochs 300 \
    --output student_policy.pt
"""
import argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--history", type=int, default=10, help="歷史窗口大小 N")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--output", type=str, default="student_policy.pt")
parser.add_argument("--hidden", type=str, default="256,128",
                    help="隱藏層，逗號分隔")
args = parser.parse_args()

STUDENT_DIM = 23
ACTION_DIM = 18


class StudentPolicyWithHistory(nn.Module):
    """
    Student MLP：吃 N 幀歷史的本體感知
    輸入: [batch, N * 52]  →  輸出: [batch, 18]
    """
    def __init__(self, obs_dim, act_dim, history_len, hidden_sizes):
        super().__init__()
        input_dim = obs_dim * history_len
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.history_len = history_len

    def forward(self, x):
        return self.net(x)


def build_history_dataset(obs_seq, act_seq, done_seq, history_len):
    """
    從時序資料建立歷史窗口

    Args:
        obs_seq:  [steps, envs, 52]
        act_seq:  [steps, envs, 18]
        done_seq: [steps, envs] bool
        history_len: N

    Returns:
        obs_windows: [M, N*52]  — M 個有效樣本
        actions:     [M, 18]
    """
    steps, envs, obs_dim = obs_seq.shape
    windows, actions = [], []

    for e in range(envs):
        # 每個 env 獨立處理，遇到 done 就重置歷史
        buffer = torch.zeros(history_len, obs_dim)  # 歷史 buffer

        for t in range(steps):
            # 滾動 buffer：丟掉最舊的，加入最新的
            buffer = torch.roll(buffer, -1, dims=0)
            buffer[-1] = obs_seq[t, e]

            # 前 history_len-1 步還沒填滿，跳過
            if t < history_len - 1:
                if done_seq[t, e]:
                    buffer.zero_()
                continue

            # 存配對
            windows.append(buffer.reshape(-1).clone())  # [N*52]
            actions.append(act_seq[t, e].clone())         # [18]

            # episode 結束 → 重置歷史
            if done_seq[t, e]:
                buffer.zero_()

    return torch.stack(windows, dim=0), torch.stack(actions, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = args.history

    # 載入
    data = torch.load(args.data, weights_only=True)
    obs_seq = data["obs_seq"]    # [steps, envs, 52]
    act_seq = data["act_seq"]    # [steps, envs, 18]
    done_seq = data["done_seq"]  # [steps, envs]
    steps, envs, obs_dim = obs_seq.shape
    print(f"資料: {steps} steps × {envs} envs, obs={obs_dim}, history={H}")

    # 建立歷史窗口
    print("建立歷史窗口...")
    obs_win, actions = build_history_dataset(obs_seq, act_seq, done_seq, H)
    print(f"有效樣本: {obs_win.shape[0]:,}  輸入維度: {obs_win.shape[1]}")

    obs_win = obs_win.to(device)
    actions = actions.to(device)

    # 正規化
    obs_mean, obs_std = obs_win.mean(0), obs_win.std(0).clamp(min=1e-6)
    act_mean, act_std = actions.mean(0), actions.std(0).clamp(min=1e-6)
    obs_n = (obs_win - obs_mean) / obs_std
    act_n = (actions - act_mean) / act_std

    # Train/Val
    ds = TensorDataset(obs_n, act_n)
    n_val = int(len(ds) * 0.1)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # 模型
    hs = [int(x) for x in args.hidden.split(",")]
    model = StudentPolicyWithHistory(STUDENT_DIM, ACTION_DIM, H, hs).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Student: {STUDENT_DIM}×{H}={STUDENT_DIM*H} → {hs} → {ACTION_DIM}  ({params:,} params)")

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        t_loss = 0.0
        for bx, by in train_dl:
            loss = loss_fn(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item() * bx.size(0)
        t_loss /= n_train

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                v_loss += loss_fn(model(bx), by).item() * bx.size(0)
        v_loss /= n_val
        sched.step()

        mark = ""
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " ★"

        if (ep+1) % 20 == 0 or ep == 0:
            print(f"Ep {ep+1:3d}/{args.epochs}  "
                  f"train={t_loss:.6f}  val={v_loss:.6f}{mark}")

    torch.save({
        "model_state_dict": best_state,
        "obs_mean": obs_mean.cpu(), "obs_std": obs_std.cpu(),
        "act_mean": act_mean.cpu(), "act_std": act_std.cpu(),
        "hidden_sizes": hs,
        "obs_dim": STUDENT_DIM, "act_dim": ACTION_DIM,
        "history_len": H,
    }, args.output)
    print(f"\n✅ 儲存: {args.output}  (best val={best_val:.6f})")


if __name__ == "__main__":
    main()