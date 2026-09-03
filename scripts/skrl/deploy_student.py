"""
deploy_student.py — 觀測歷史版
維護 rolling buffer，每步餵 N 幀歷史給 Student

用法：
  python scripts/skrl/deploy_student.py \
    --task SpiderHybrid \
    --student student_policy.pt \
    --num_envs 16 --num_steps 1000 --headless
"""
import argparse, sys, torch, torch.nn as nn
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="SpiderHybrid")
parser.add_argument("--student", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import spider.tasks
from isaaclab_rl.skrl import SkrlVecEnvWrapper

STUDENT_DIM = 23


class StudentPolicyWithHistory(nn.Module):
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

# # wrapped_env.step(rl_action) 之前加
# def reorder_sim_to_real(a):  # [envs, 18]
#     out = torch.zeros_like(a)
#     for i in range(6):
#         out[:, i*3+0] = a[:, i]
#         out[:, i*3+1] = a[:, i+6]
#         out[:, i*3+2] = a[:, i+12]
#     return out   
        

def main():
    device = torch.device("cuda:0")

    # 載入 Student
    ckpt = torch.load(args_cli.student, map_location=device, weights_only=True)
    H = ckpt.get("history_len", 1)
    model = StudentPolicyWithHistory(
        ckpt["obs_dim"], ckpt["act_dim"], H, ckpt["hidden_sizes"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    act_mean = ckpt["act_mean"].to(device)
    act_std = ckpt["act_std"].to(device)

    print(f"✅ Student 載入: history={H}, input={STUDENT_DIM}×{H}={STUDENT_DIM*H}")

    # 環境
    import importlib
    mod = importlib.import_module("spider.tasks.manager_based.spider.spider_env_cfg_hybrid")
    env_cfg = mod.SpiderEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # ── 歷史 buffer [envs, H, 52] ──
    history_buf = torch.zeros(args_cli.num_envs, H, STUDENT_DIM, device=device)

    obs, _ = wrapped_env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]

    total_reward = torch.zeros(args_cli.num_envs, device=device)
    episode_count = torch.zeros(args_cli.num_envs, device=device)
    # 初始化
    prev_action = torch.zeros(args_cli.num_envs, 18, device=obs.device)

    # 改成：
    for step in range(args_cli.num_steps):
        # 萃取 22 維觀測
        student_obs = torch.zeros(args_cli.num_envs, 23, device=device)
        student_obs[:, 0:1]   = obs[:, 42:43]    # up_proj (1)
        student_obs[:, 1:19]  = prev_action       # prev_action (18)
        student_obs[:, 19:21] = obs[:, 48:50]    # phase (2)
        student_obs[:, 21:23] = obs[:, 46:48]    # heading_err (2)
        
        history_buf = torch.roll(history_buf, -1, dims=1)
        history_buf[:, -1] = student_obs
        obs_flat = history_buf.reshape(args_cli.num_envs, -1)

        # 正規化
        obs_norm = (obs_flat - obs_mean) / obs_std

        # Student 推理
        with torch.no_grad():
            rl_action_norm = model(obs_norm)
            rl_action = rl_action_norm * act_std + act_mean

        prev_action = rl_action.clone()

        
        # rl_action = reorder_sim_to_real(rl_action)  # ← 加這行

        obs, reward, terminated, truncated, _ = wrapped_env.step(rl_action)
        # 在 env.step 之後加
        # 取得 base angular velocity z
        ang_vel = env.unwrapped.scene["robot"].data.root_ang_vel_w[:, 2]
        yaw_rate = ang_vel.mean().item()
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: avg_reward={total_reward.mean().item():.2f} | yaw_rate={yaw_rate:.3f} rad/s")

        if isinstance(obs, dict):
            obs = obs["policy"]

        total_reward += reward.squeeze()
        done = terminated.squeeze() | truncated.squeeze()
        episode_count += done.float()

        if done.any():
            history_buf[done] = 0.0
            prev_action[done] = 0.0  # 這行也要加

        if (step + 1) % 200 == 0:
            print(f"Step {step+1}: avg_reward≈{total_reward.mean().item():.2f}")

    eps = episode_count.sum().item()
    if eps > 0:
        print(f"\n平均 episode reward: {total_reward.sum().item() / eps:.2f}")
    else:
        print(f"\n總 reward: {total_reward.mean().item():.2f} (未完成任何 episode)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()