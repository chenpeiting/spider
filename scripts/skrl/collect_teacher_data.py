"""
collect_teacher_data.py — 保留時序版
存 [num_steps, num_envs, dim] 以便建立觀測歷史窗口

用法：
  python scripts/skrl/collect_teacher_data.py \
    --task SpiderHybrid \
    --checkpoint <teacher.pt> \
    --num_envs 64 --num_steps 3000 \
    --output teacher_data.pt --headless
"""
import argparse, sys, os, torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="SpiderHybrid")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=3000)
parser.add_argument("--output", type=str, default="teacher_data.pt")
parser.add_argument("--checkpoint", type=str, default=None)
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

STUDENT_DIM = 23

agent_cfg_entry_point = "skrl_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)

    # 載入 checkpoint
    ckpt_path = args_cli.checkpoint
    if ckpt_path is None:
        log_root = os.path.join("logs", "skrl",
                                agent_cfg["agent"]["experiment"]["directory"])
        if os.path.exists(log_root):
            runs = sorted(os.listdir(log_root))
            if runs:
                ckpt_dir = os.path.join(log_root, runs[-1], "checkpoints")
                if os.path.exists(ckpt_dir):
                    pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
                    if pts:
                        ckpt_path = os.path.join(ckpt_dir, pts[-1])

    if ckpt_path is None:
        print("❌ 找不到 checkpoint"); env.close(); return

    ckpt_path = retrieve_file_path(ckpt_path)
    print(f"✅ Teacher: {ckpt_path}")
    runner.agent.load(ckpt_path)
    runner.agent.set_running_mode("eval")

    # ── 收集（保留時序）──
    obs_seq = []   # 會變成 [steps, envs, 52]
    act_seq = []   # 會變成 [steps, envs, 18]
    done_seq = []  # 會變成 [steps, envs]，標記 episode 邊界

    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    print(f"Obs dim: {obs.shape[1]} (Student 取前 {STUDENT_DIM})")

    # 靜止暖機段：zero action，但記錄真實 up_proj
    print("收集靜止初始狀態...")
    prev_action = torch.zeros(args_cli.num_envs, 18, device=obs.device)
    for _ in range(100):
        zero_action = torch.zeros(args_cli.num_envs, 18, device=obs.device)

        s_obs_23 = torch.zeros(args_cli.num_envs, 23, device=obs.device)
        s_obs_23[:, 0:1]   = obs[:, 42:43]    # up_proj (1)
        s_obs_23[:, 1:19]  = prev_action       # prev_action (18)
        s_obs_23[:, 19:21] = obs[:, 48:50]    # phase (2)
        s_obs_23[:, 21:23] = obs[:, 46:48]    # heading_err (2)

        obs_seq.append(s_obs_23.cpu())
        act_seq.append(zero_action.cpu())
        done_seq.append(torch.zeros(args_cli.num_envs, dtype=torch.bool))

        prev_action = zero_action.clone()
        obs, _, _, _, _ = env.step(zero_action)
        if isinstance(obs, dict):
            obs = obs["policy"]

    target_seq = []
    # ── 正常收集走動資料 ──
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    prev_action = torch.zeros(args_cli.num_envs, 18, device=obs.device)
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            action = runner.agent.act(
                obs, timestep=step, timesteps=args_cli.num_steps
            )[0]
        
        # 存 Teacher 的 joint_pos（≈ 實際 joint target）
        target_seq.append(obs[:, 0:18].cpu())
        
        # 萃取 Phase II 22 維觀測
        s_obs_23 = torch.zeros(args_cli.num_envs, 23, device=obs.device)
        s_obs_23[:, 0:1]   = obs[:, 42:43]    # up_proj (1)
        s_obs_23[:, 1:19]  = prev_action       # prev_action (18)
        s_obs_23[:, 19:21] = obs[:, 48:50]    # phase (2)
        s_obs_23[:, 21:23] = obs[:, 46:48]    # heading_err (2)
        
        obs_seq.append(s_obs_23.cpu())
        act_seq.append(action.cpu())
        prev_action = action.clone()
        
        obs, _, terminated, truncated, _ = env.step(action)
        if isinstance(obs, dict):
            obs = obs["policy"]
        
        done = (terminated.squeeze() | truncated.squeeze()).cpu()
        done_seq.append(done)
        
        if (step + 1) % 500 == 0:
            print(f"  {step+1}/{args_cli.num_steps}")

    dataset = {
        "obs_seq": torch.stack(obs_seq, dim=0),    # [steps, envs, 52]
        "act_seq": torch.stack(act_seq, dim=0),    # [steps, envs, 18]
        "done_seq": torch.stack(done_seq, dim=0),  # [steps, envs]
        "target_seq": torch.stack(target_seq),  # 只有 num_steps 長，回放用
        "student_dim": STUDENT_DIM,
    }
    torch.save(dataset, args_cli.output)
    print(f"\n✅ {args_cli.num_steps} steps × {args_cli.num_envs} envs → {args_cli.output}")
    print(f"   obs_seq:  {dataset['obs_seq'].shape}")
    print(f"   act_seq:  {dataset['act_seq'].shape}")
    print(f"   done_seq: {dataset['done_seq'].shape}")
    print(f"   大小: {os.path.getsize(args_cli.output)/1e6:.1f} MB")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()