"""
dagger_label.py — DAgger 第二步：Teacher 標註 + 合併 + 重訓
完整 DAgger 流程：

  Step A: Student 跑環境收集 obs
    python scripts/skrl/deploy_student.py \
      --task SpiderHybrid --student student_policy.pt \
      --num_envs 64 --num_steps 2000 \
      --dagger --output dagger_obs.pt --headless

  Step B: Teacher 標註 + 合併 + 重訓（本腳本）
    python scripts/skrl/dagger_label.py \
      --task SpiderHybrid \
      --checkpoint <teacher.pt> \
      --dagger_obs dagger_obs.pt \
      --original_data teacher_data.pt \
      --output student_policy_dagger.pt \
      --headless
"""
import argparse, sys, os, torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="SpiderHybrid")
parser.add_argument("--checkpoint", type=str, required=True, help="Teacher checkpoint")
parser.add_argument("--dagger_obs", type=str, required=True, help="deploy_student 產出的 dagger_obs.pt")
parser.add_argument("--original_data", type=str, required=True, help="原始 teacher_data.pt")
parser.add_argument("--output", type=str, default="student_policy_dagger.pt")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--hidden", type=str, default="128,64")
parser.add_argument("--num_envs", type=int, default=16)  # 標註用，不需多
parser.add_argument("--ml_framework", type=str, default="torch")
parser.add_argument("--algorithm", type=str, default="PPO")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from skrl.utils.runner.torch import Runner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa
import spider.tasks     # noqa

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

STUDENT_DIM = 52
ACTION_DIM = 18

agent_cfg_entry_point = "skrl_cfg_entry_point"


class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    device = torch.device("cuda:0")

    # ══════════════════════════════════════
    # Phase 1: Teacher 標註 DAgger obs
    # ══════════════════════════════════════
    print("=" * 50)
    print("Phase 1: Teacher 標註 DAgger obs")
    print("=" * 50)

    # 載入 DAgger obs（Student 遇到的完整 173 dim obs）
    dagger_data = torch.load(args_cli.dagger_obs, weights_only=True)
    dagger_full_obs = dagger_data["full_obs"].to(device)  # [M, 173]
    n_dagger = dagger_full_obs.shape[0]
    print(f"DAgger obs: {n_dagger:,} 筆")

    # 建環境 + 載入 Teacher（只為了用 agent.act）
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)
    ckpt_path = retrieve_file_path(args_cli.checkpoint)
    print(f"Teacher: {ckpt_path}")
    runner.agent.load(ckpt_path)
    runner.agent.set_running_mode("eval")

    # 批次標註（不需要 env.step，只需要 policy forward）
    teacher_actions = []
    batch_size = 1024
    for i in range(0, n_dagger, batch_size):
        batch_obs = dagger_full_obs[i:i + batch_size]
        with torch.no_grad():
            act = runner.agent.act(
                batch_obs, timestep=0, timesteps=1
            )[0]
        teacher_actions.append(act.cpu())
        if (i // batch_size + 1) % 20 == 0:
            print(f"  標註 {min(i+batch_size, n_dagger):,}/{n_dagger:,}")

    dagger_actions = torch.cat(teacher_actions, dim=0)
    dagger_student_obs = dagger_full_obs[:, :STUDENT_DIM].cpu()
    print(f"✅ 標註完成: {dagger_actions.shape}")

    env.close()

    # ══════════════════════════════════════
    # Phase 2: 合併資料
    # ══════════════════════════════════════
    print("\n" + "=" * 50)
    print("Phase 2: 合併原始 + DAgger 資料")
    print("=" * 50)

    original = torch.load(args_cli.original_data, weights_only=True)
    orig_obs = original["student_obs"]    # [N, 52]
    orig_act = original["teacher_action"] # [N, 18]

    # 合併
    all_obs = torch.cat([orig_obs, dagger_student_obs], dim=0)
    all_act = torch.cat([orig_act, dagger_actions], dim=0)
    print(f"原始: {orig_obs.shape[0]:,}  DAgger: {dagger_student_obs.shape[0]:,}  總計: {all_obs.shape[0]:,}")

    # ══════════════════════════════════════
    # Phase 3: 重新訓練 Student
    # ══════════════════════════════════════
    print("\n" + "=" * 50)
    print("Phase 3: 訓練 Student")
    print("=" * 50)

    all_obs = all_obs.to(device)
    all_act = all_act.to(device)

    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0).clamp(min=1e-6)
    act_mean, act_std = all_act.mean(0), all_act.std(0).clamp(min=1e-6)
    obs_n = (all_obs - obs_mean) / obs_std
    act_n = (all_act - act_mean) / act_std

    ds = TensorDataset(obs_n, act_n)
    n_val = int(len(ds) * 0.1)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=args_cli.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args_cli.batch_size)

    hs = [int(x) for x in args_cli.hidden.split(",")]
    model = StudentPolicy(STUDENT_DIM, ACTION_DIM, hs).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Student: {STUDENT_DIM}→{hs}→{ACTION_DIM}  ({params:,} params)")

    opt = optim.Adam(model.parameters(), lr=args_cli.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args_cli.epochs)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(args_cli.epochs):
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

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"Ep {ep+1:3d}/{args_cli.epochs}  "
                  f"train={t_loss:.6f}  val={v_loss:.6f}{mark}")

    torch.save({
        "model_state_dict": best_state,
        "obs_mean": obs_mean.cpu(), "obs_std": obs_std.cpu(),
        "act_mean": act_mean.cpu(), "act_std": act_std.cpu(),
        "hidden_sizes": hs, "obs_dim": STUDENT_DIM, "act_dim": ACTION_DIM,
    }, args_cli.output)
    print(f"\n✅ DAgger Student 儲存: {args_cli.output}  (best val={best_val:.6f})")


if __name__ == "__main__":
    main()
    simulation_app.close()