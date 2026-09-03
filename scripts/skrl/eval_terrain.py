#!/usr/bin/env python3
"""
eval_terrain.py — Single-terrain evaluation with npz output.

每次執行只跑一個 terrain，將 100 個 episode 的資料存成 {out_dir}/{terrain}.npz。
用 run_eval.sh 連跑三次後再加 --plot_only 合併出圖。

用法（單次）:
    python scripts/skrl/eval_terrain.py \\
        --task Isaac-Spider-Hybrid-v0 \\
        --checkpoint logs/.../best_agent.pt \\
        --terrain stairs --num_episodes 100 --mode hybrid --headless

產出比較圖:
    python scripts/skrl/eval_terrain.py \\
        --plot_only --out_dir logs/eval --mode hybrid --task Spider-Hybrid

或直接:
    bash scripts/run_eval.sh \\
        Isaac-Spider-Hybrid-v0 logs/.../best_agent.pt hybrid logs/eval
"""

import sys, os

# ─── plot-only 模式：不需要 Isaac Sim ────────────────────────────────────────
if "--plot_only" in sys.argv:
    import argparse as _ap, numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _p = _ap.ArgumentParser()
    _p.add_argument("--out_dir", required=True)
    _p.add_argument("--mode",      default="pure_rl",
                    choices=["pure_rl", "hybrid", "pure_cpg"])
    _p.add_argument("--vel_cmd",   type=float, default=0.2)
    _p.add_argument("--stair_cross_dist", type=float, default=1.7,
                    help="Min max_x_displacement (m) for stairs success. "
                         "2m tile: full traversal≈1.85m; 0.85m=half. default=1.7")
    _p.add_argument("--task",      default="eval")
    _args, _ = _p.parse_known_args()

    _TERRAIN_COLORS = {"flat": "#2196F3", "noise": "#FF9800", "stairs": "#4CAF50"}
    _TERRAIN_LABELS = {"flat": "平地\n(Flat)", "noise": "噪聲\n(Noise)", "stairs": "階梯\n(Stairs)"}

    def _load_npz(out_dir, terrain):
        path = os.path.join(out_dir, f"{terrain}.npz")
        if not os.path.exists(path):
            print(f"[Plot] {path} not found, skipping")
            return [], 500
        d = np.load(path)
        N = len(d["lengths"])
        T = d["heights"].shape[1]
        eps = []
        for i in range(N):
            L = int(d["lengths"][i])
            eps.append({
                "height":     d["heights"][i, :L],
                "vel":        d["vels"][i, :L],
                "orient":     d["orients"][i, :L],
                "length":     L,
                "x_disp":     float(d["x_disps"][i]),
                # backward compat: old npz may not have max_x_disps
                "max_x_disp": float(d["max_x_disps"][i])
                               if "max_x_disps" in d else float(d["x_disps"][i]),
            })
        return eps, int(d["max_ep_len"])

    def _pad(episodes, key):
        if not episodes: return np.zeros((0, 1))
        T = max(len(e[key]) for e in episodes)
        mat = np.full((len(episodes), T), np.nan, dtype=np.float32)
        for i, ep in enumerate(episodes):
            a = ep[key]; mat[i, :len(a)] = a
        return mat

    def _draw(ax, mat, color, ylabel=None, hline=None, n_traces=20):
        if mat.shape[0] == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); return
        idx = np.random.choice(mat.shape[0], min(n_traces, mat.shape[0]), replace=False)
        for i in idx:
            ax.plot(mat[i], color=color, alpha=0.10, linewidth=0.6)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        t    = np.arange(mat.shape[1])
        ax.plot(mean, color=color, linewidth=2.0, label=f"μ={np.nanmean(mean):.3f}")
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18)
        if hline is not None:
            ax.axhline(hline, color="red", ls="--", lw=1.5, alpha=0.8, label=f"cmd={hline:.2f}")
        if ylabel: ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("step", fontsize=8); ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

    def _savefig(fig, path):
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[Plot] → {path}")

    all_results, max_ep_len = {}, 500
    for _t in ["flat", "noise", "stairs"]:
        eps, mel = _load_npz(_args.out_dir, _t)
        if eps: all_results[_t] = eps; max_ep_len = mel
    terrains = list(all_results.keys())
    if not terrains:
        print("[Plot] No terrain data found. Run terrain evaluations first."); sys.exit(1)

    n_t = len(terrains)
    colors  = [_TERRAIN_COLORS[t] for t in terrains]

    # ── Fig 1: Height ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_t, figsize=(6*n_t, 5), sharey=True)
    if n_t == 1: axes = [axes]
    for ax, t in zip(axes, terrains):
        _draw(ax, _pad(all_results[t], "height"), _TERRAIN_COLORS[t],
              ylabel="body z (m)" if t == terrains[0] else None)
        ax.set_title(_TERRAIN_LABELS[t], fontsize=11)
    fig.suptitle(f"Body Height — {_args.task} [{_args.mode}]", fontsize=12)
    fig.tight_layout(); _savefig(fig, os.path.join(_args.out_dir, "eval_height.png"))

    # ── Fig 2: Velocity (skip CPG) ─────────────────────────────────────
    if _args.mode != "pure_cpg":
        fig, axes = plt.subplots(1, n_t, figsize=(6*n_t, 5), sharey=True)
        if n_t == 1: axes = [axes]
        for ax, t in zip(axes, terrains):
            _draw(ax, _pad(all_results[t], "vel"), _TERRAIN_COLORS[t],
                  ylabel="vx (m/s)" if t == terrains[0] else None, hline=_args.vel_cmd)
            ax.set_title(_TERRAIN_LABELS[t], fontsize=11)
        fig.suptitle(f"Forward Velocity vs {_args.vel_cmd} m/s — {_args.task}", fontsize=12)
        fig.tight_layout(); _savefig(fig, os.path.join(_args.out_dir, "eval_velocity.png"))

    # ── Fig 3: Orientation ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_t, figsize=(6*n_t, 5), sharey=True)
    if n_t == 1: axes = [axes]
    for ax, t in zip(axes, terrains):
        _draw(ax, _pad(all_results[t], "orient"), _TERRAIN_COLORS[t],
              ylabel="|proj_grav_xy| (rad)" if t == terrains[0] else None)
        ax.set_title(_TERRAIN_LABELS[t], fontsize=11)
    fig.suptitle(f"Orientation Stability — {_args.task} [{_args.mode}]", fontsize=12)
    fig.tight_layout(); _savefig(fig, os.path.join(_args.out_dir, "eval_orientation.png"))

    # ── Fig 4: Episode length violin ───────────────────────────────────
    fig, axes = plt.subplots(1, n_t, figsize=(4*n_t, 5), sharey=True)
    if n_t == 1: axes = [axes]
    for ax, t in zip(axes, terrains):
        eps = all_results[t]
        if not eps: ax.set_title(_TERRAIN_LABELS[t]); continue
        lens = [ep["length"] for ep in eps]
        parts = ax.violinplot([lens], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(_TERRAIN_COLORS[t]); pc.set_alpha(0.65)
        ax.axhline(max_ep_len * 0.9, color="red", ls="--", lw=1.2, alpha=0.7, label="90% thr")
        ax.set_xticks([1]); ax.set_xticklabels([_TERRAIN_LABELS[t].replace("\n"," ")], fontsize=8)
        ax.set_title(_TERRAIN_LABELS[t], fontsize=11); ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("episode length (steps)", fontsize=9)
    fig.suptitle(f"Episode Length — {_args.task}", fontsize=12)
    fig.tight_layout(); _savefig(fig, os.path.join(_args.out_dir, "eval_ep_length.png"))

    # ── Fig 5: Summary bar chart ───────────────────────────────────────
    sr_v, ep_v, mh_v, mv_v, mo_v = [], [], [], [], []
    for t in terrains:
        eps = all_results[t]
        if not eps:
            sr_v.append(0); ep_v.append(0); mh_v.append(0); mv_v.append(0); mo_v.append(0); continue
        if t == "stairs":
            # 成功 = episode 中最遠前進距離 >= threshold（穿越整段對稱樓梯）
            # tile 2m，spawn 在 0.15m，完整穿越 ≈ 1.85m，門檻 1.7m
            sr = np.mean([ep["max_x_disp"] >= _args.stair_cross_dist
                          for ep in eps]) * 100
        else:
            sr = np.mean([ep["length"] >= max_ep_len * 0.9 for ep in eps]) * 100
        sr_v.append(sr)
        ep_v.append(np.mean([ep["length"] for ep in eps]))
        mh_v.append(float(np.nanmean(np.concatenate([ep["height"] for ep in eps]))))
        mv_v.append(float(np.nanmean(np.concatenate([ep["vel"]    for ep in eps]))))
        mo_v.append(float(np.nanmean(np.concatenate([ep["orient"] for ep in eps]))))

    xlabels = [_TERRAIN_LABELS[t].replace("\n", " ") for t in terrains]
    n_bars  = 4 if _args.mode == "pure_cpg" else 5
    fig, axes = plt.subplots(1, n_bars, figsize=(4*n_bars, 5))
    if n_bars == 1: axes = [axes]

    def _bar(ax, vals, title, ylabel, hlv=None):
        bars = ax.bar(xlabels, vals, color=colors, edgecolor="white", width=0.5)
        if hlv is not None and _args.mode != "pure_cpg":
            ax.axhline(hlv, color="red", ls="--", lw=1.5, alpha=0.8, label=f"cmd={hlv:.2f}")
            ax.legend(fontsize=7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals + [1e-6]) * 0.02,
                    f"{v:.2f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=0.3)

    _bar(axes[0], sr_v,  "成功率\nSuccess (%)",       "%")
    _bar(axes[1], ep_v,  "存活步數\nSurvival Steps",  "steps")
    _bar(axes[2], mh_v,  "平均高度\nMean Height (m)", "m")
    _bar(axes[3], mo_v,  "姿態誤差\nOrient (rad)",    "rad")
    if _args.mode != "pure_cpg":
        _bar(axes[4], mv_v, "平均速度\nVel (m/s)", "m/s", hlv=_args.vel_cmd)

    fig.suptitle(f"Terrain Comparison — {_args.task} [{_args.mode}]", fontsize=12)
    fig.tight_layout()
    _savefig(fig, os.path.join(_args.out_dir, "eval_summary.png"))

    # Text summary
    txt = os.path.join(_args.out_dir, "eval_summary.txt")
    with open(txt, "w") as f:
        f.write(f"EVAL — {_args.task} [{_args.mode}]\n")
        f.write(f"stair_success: max_x_disp>={_args.stair_cross_dist}m (穿越對稱樓梯)  |  flat/noise survival>=90%ep\n")
        f.write("="*60 + "\n")
        f.write(f"{'Terrain':<10} {'SR%':>7} {'Steps':>8} {'H(m)':>8} {'V':>7} {'O(r)':>8}\n")
        f.write("-"*60 + "\n")
        for i, t in enumerate(terrains):
            f.write(f"{t:<10} {sr_v[i]:>7.1f} {ep_v[i]:>8.1f} "
                    f"{mh_v[i]:>8.4f} {mv_v[i]:>7.4f} {mo_v[i]:>8.4f}\n")
    print("\n" + open(txt).read())
    sys.exit(0)

# ─── normal mode: needs Isaac Sim ────────────────────────────────────────────
import argparse

parser = argparse.ArgumentParser(description="Single-terrain locomotion evaluation")
parser.add_argument("--task",         required=True)
parser.add_argument("--checkpoint",   required=True)
parser.add_argument("--terrain",      default="flat",
                    choices=["flat", "noise", "stairs"],
                    help="Terrain to evaluate (one at a time)")
parser.add_argument("--num_episodes", type=int,   default=100)
parser.add_argument("--vel_cmd",      type=float, default=0.2)
parser.add_argument("--out_dir",      default=None)
parser.add_argument("--mode",         default="pure_rl",
                    choices=["pure_rl", "hybrid", "pure_cpg"])
parser.add_argument("--stair_cross_dist", type=float, default=1.7)
parser.add_argument("--agent",        default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── post-launch imports ─────────────────────────────────────────────────────
import copy
import numpy as np
import torch
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")

import skrl
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import spider.tasks  # noqa


# ─── terrain factory ─────────────────────────────────────────────────────────
def _make_terrain(terrain_type: str) -> TerrainImporterCfg:
    phys = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
        static_friction=1.0, dynamic_friction=1.0,
    )
    if terrain_type == "flat":
        return TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane",
            collision_group=-1, physics_material=phys, debug_vis=False,
        )

    gen_kw = dict(seed=42, horizontal_scale=0.05, vertical_scale=0.005,
                  slope_threshold=0.75, use_cache=False)

    if terrain_type == "noise":
        gen_kw.update(size=(2.0, 2.0), border_width=0.1, num_rows=20, num_cols=20)
        sub = {"noise": HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.06),
            noise_step=0.01, border_width=0.1,
        )}
    else:  # stairs
        # 與訓練完全一致的對稱階梯（hf_up_down_stairs）
        # tile 2m×2m：edge(0m)=height 0 → peak(1m)=0.32m → edge(2m)=0
        # spawn 覆寫到底邊(0.15m)，機器人需穿越整段上坡+下坡
        @height_field_to_mesh
        def _hf_stairs(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
            lp = int(cfg.size[0] / cfg.horizontal_scale)
            wp = int(cfg.size[1] / cfg.horizontal_scale)
            hf = np.zeros((lp, wp), dtype=np.float32)
            sw = max(1, int(0.20 / cfg.horizontal_scale))  # 4 px = 0.2m/step
            sh = 0.08 / cfg.vertical_scale                 # 16 units = 0.08m/step
            for i in range(lp):
                hf[i, :] = (min(i, lp - 1 - i) // sw) * sh  # symmetric up-down
            return np.rint(hf).astype(np.int16)

        @configclass
        class _StairsCfg(HfTerrainBaseCfg):
            function = _hf_stairs
            horizontal_scale: float = 0.05
            vertical_scale:   float = 0.005

        gen_kw.update(size=(2.0, 2.0), border_width=0.2, num_rows=20, num_cols=20)
        sub = {"stairs": _StairsCfg(
            proportion=1.0, size=(2.0, 2.0),
            horizontal_scale=0.05, vertical_scale=0.005, border_width=0.0,
        )}

    return TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(**gen_kw, sub_terrains=sub),
        collision_group=-1, physics_material=phys, debug_vis=False,
    )


# ─── env factory ─────────────────────────────────────────────────────────────
def _make_env(env_cfg, terrain_type, vel_cmd, mode, task_name):
    cfg = copy.deepcopy(env_cfg)
    cfg.scene.num_envs = 1
    cfg.sim.device = getattr(args_cli, "device", None) or cfg.sim.device
    cfg.scene.terrain = _make_terrain(terrain_type)

    if mode != "pure_cpg":
        try:
            r = cfg.commands.base_velocity.ranges
            r.lin_vel_x = (vel_cmd, vel_cmd)
            r.lin_vel_y = (0.0, 0.0)
            r.ang_vel_z = (0.0, 0.0)
        except AttributeError:
            pass

    env = gym.make(task_name, cfg=cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return SkrlVecEnvWrapper(env, ml_framework="torch")


# ─── episode runner ──────────────────────────────────────────────────────────
def run_episodes(env_w, agent, n_episodes: int, terrain_type: str = "flat"):
    isaac_env = (env_w.unwrapped.env
                 if hasattr(env_w.unwrapped, "env")
                 else env_w.unwrapped)
    max_ep_len = int(getattr(isaac_env, "max_episode_length", 500))

    # ── stairs: 覆寫 spawn 到 tile 底邊 ──────────────────────────────
    # tile 2m×2m, center=1m → 底邊 0.15m；shift = 0.85m
    _TILE_HALF    = 1.0   # 2m tile / 2
    _EDGE_MARGIN  = 0.15  # spawn 距底邊
    _SPAWN_SHIFT  = _TILE_HALF - _EDGE_MARGIN  # 0.85m
    if terrain_type == "stairs":
        try:
            # TerrainImporter stores origins in _env_origins tensor
            origs = isaac_env.scene.terrain.env_origins
            origs[0, 0] = origs[0, 0] - _SPAWN_SHIFT
        except Exception as e:
            print(f"[Eval] spawn override failed ({e}), using tile center")

    episodes = []
    obs, _ = env_w.reset()
    robot = isaac_env.scene["robot"]

    ep_h, ep_v, ep_o = [], [], []
    ep_max_x = 0.0       # max forward advancement from spawn this episode
    prev_buf  = None
    x_start   = float(robot.data.root_pos_w[0, 0].item())

    while len(episodes) < n_episodes and simulation_app.is_running():
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env_w.step(actions)

        cur_x = float(robot.data.root_pos_w[0, 0].item())
        ep_max_x = max(ep_max_x, cur_x - x_start)   # signed: +x = forward into stairs

        ep_h.append(float(robot.data.root_pos_w[0, 2].item()))
        ep_v.append(float(robot.data.root_lin_vel_w[0, 0].item()))
        ep_o.append(float(torch.norm(robot.data.projected_gravity_b[0, :2]).item()))

        cur_buf = int(isaac_env.episode_length_buf[0].item())
        if prev_buf is not None and cur_buf < prev_buf and len(ep_h) > 5:
            ep_len = prev_buf
            episodes.append({
                "height":     np.array(ep_h, dtype=np.float32),
                "vel":        np.array(ep_v, dtype=np.float32),
                "orient":     np.array(ep_o, dtype=np.float32),
                "length":     ep_len,
                "x_disp":     abs(float(robot.data.root_pos_w[0, 0].item()) - x_start),
                "max_x_disp": max(ep_max_x, 0.0),   # max forward progress
            })
            n = len(episodes)
            print(f"  ep {n:3d}/{n_episodes}  len={ep_len:4d}  "
                  f"max_x={episodes[-1]['max_x_disp']:.2f}m  "
                  f"z={np.mean(ep_h):.3f}m  orient={np.mean(ep_o):.3f}r")
            ep_h, ep_v, ep_o = [], [], []
            ep_max_x = 0.0
            x_start = float(robot.data.root_pos_w[0, 0].item())
        prev_buf = cur_buf

    return episodes, max_ep_len


# ─── save helpers ─────────────────────────────────────────────────────────────
def _save_npz(out_dir: str, terrain: str, episodes: list, max_ep_len: int):
    """Pad + save episode arrays to {out_dir}/{terrain}.npz."""
    os.makedirs(out_dir, exist_ok=True)
    if not episodes:
        print(f"[Eval] No episodes to save for {terrain}"); return

    max_T = max(len(ep["height"]) for ep in episodes)

    def _pad(key):
        m = np.full((len(episodes), max_T), np.nan, dtype=np.float32)
        for i, ep in enumerate(episodes):
            a = ep[key]; m[i, :len(a)] = a
        return m

    path = os.path.join(out_dir, f"{terrain}.npz")
    np.savez(
        path,
        heights    = _pad("height"),
        vels       = _pad("vel"),
        orients    = _pad("orient"),
        lengths    = np.array([ep["length"]     for ep in episodes], dtype=np.int32),
        x_disps    = np.array([ep["x_disp"]     for ep in episodes], dtype=np.float32),
        max_x_disps= np.array([ep.get("max_x_disp", ep["x_disp"])
                                for ep in episodes], dtype=np.float32),
        max_ep_len = np.array([max_ep_len], dtype=np.int32),
    )
    print(f"[Eval] Saved {len(episodes)} episodes → {path}")


# ─── main ─────────────────────────────────────────────────────────────────────
_agent_entry = args_cli.agent or "skrl_cfg_entry_point"
_algorithm   = ("ppo" if args_cli.agent is None
                else args_cli.agent.split("_cfg")[0].split("skrl_")[-1].lower())


@hydra_task_config(args_cli.task, _agent_entry)
def main(env_cfg: ManagerBasedRLEnvCfg, experiment_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    terrain   = args_cli.terrain

    # resolve checkpoint
    log_root = os.path.abspath(
        os.path.join("logs", "skrl",
                     experiment_cfg["agent"]["experiment"]["directory"])
    )
    resume_path = (os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint
                   else get_checkpoint_path(log_root,
                        run_dir=f".*_{_algorithm}_torch",
                        other_dirs=["checkpoints"]))
    out_dir = args_cli.out_dir or os.path.abspath(
        os.path.join(os.path.dirname(resume_path), "..", "eval_terrain"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Eval] task       : {task_name}")
    print(f"[Eval] terrain    : {terrain}")
    print(f"[Eval] checkpoint : {resume_path}")
    print(f"[Eval] out_dir    : {out_dir}")

    exp_cfg = copy.deepcopy(experiment_cfg)
    exp_cfg["trainer"]["close_environment_at_exit"]           = False
    exp_cfg["agent"]["experiment"]["write_interval"]          = 0
    exp_cfg["agent"]["experiment"]["checkpoint_interval"]     = 0

    print(f"\n{'='*55}")
    print(f"[Eval] Running {args_cli.num_episodes} episodes on: {terrain}")
    print(f"{'='*55}")

    env = _make_env(env_cfg, terrain, args_cli.vel_cmd, args_cli.mode, args_cli.task)
    runner = Runner(env, exp_cfg)
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    episodes, max_ep_len = run_episodes(env, runner.agent, args_cli.num_episodes,
                                         terrain_type=args_cli.terrain)
    _save_npz(out_dir, terrain, episodes, max_ep_len)

    env.close()
    print(f"\n[Eval] Done. To generate plots after all terrains:\n"
          f"  python {sys.argv[0]} --plot_only --out_dir {out_dir} "
          f"--mode {args_cli.mode} --task {task_name}")


if __name__ == "__main__":
    main()
    simulation_app.close()
