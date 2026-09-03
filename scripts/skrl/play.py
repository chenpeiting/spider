# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--metrics_steps", type=int, default=2000, help="Number of steps to collect metrics (0 = unlimited, Ctrl+C to stop).")
parser.add_argument("--metrics_out", type=str, default=None, help="Output directory for metrics CSV / PNG. Defaults to <log_dir>/metrics.")
parser.add_argument("--skip_vel_metrics", action="store_true", default=False,
                    help="Skip lin/ang velocity tracking metrics (use when evaluating open-loop CPG against RL).")
parser.add_argument("--hybrid", action="store_true", default=False,
                    help="Enable Hybrid-specific metrics (rl_contribution_rate).")
parser.add_argument("--eval_terrain", type=str, default=None,
                    choices=["flat", "noise", "stairs"],
                    help="Override env terrain for evaluation without modifying env_cfg.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import spider.tasks  # noqa: F401
from spider.tasks.manager_based.spider import metrics as _spider_metrics


def _make_eval_terrain(terrain_type: str):
    """回傳單一地形的 TerrainImporterCfg，用於 --eval_terrain 覆蓋。"""
    import numpy as np
    from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
    from isaaclab.terrains.height_field.utils import height_field_to_mesh
    import isaaclab.sim as sim_utils

    phys = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
        static_friction=1.0, dynamic_friction=1.0,
    )
    if terrain_type == "flat":
        return TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane",
            collision_group=-1, physics_material=phys, debug_vis=False,
        )

    gen_base = dict(seed=42, size=(2.0, 2.0), border_width=0.2,
                    num_rows=20, num_cols=20,
                    horizontal_scale=0.05, vertical_scale=0.005,
                    slope_threshold=0.75, use_cache=False)

    if terrain_type == "noise":
        # noise 用小 tile 就好
        gen_base.update(size=(2.0, 2.0), border_width=0.1, num_rows=20, num_cols=20)
        sub = {"noise": HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.06),
            noise_step=0.01, border_width=0.1,
        )}
    else:  # stairs：與訓練地形相同難度（2m tile, step=0.2m/0.08m）
           # 前 50% 平地，spawn 在 tile 中心 = 第一步前 0.2m；後 50% 上坡
        @height_field_to_mesh
        def _hf_stairs(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
            lp = int(cfg.size[0] / cfg.horizontal_scale)
            wp = int(cfg.size[1] / cfg.horizontal_scale)
            hf = np.zeros((lp, wp), dtype=np.float32)
            step_w = max(1, int(0.20 / cfg.horizontal_scale))
            step_h = 0.08 / cfg.vertical_scale
            stair_start = lp // 2  # spawn at tile center = height 0, first step 0.2m ahead
            for i in range(stair_start, lp):
                hf[i, :] = ((i - stair_start) // step_w) * step_h
            return np.rint(hf).astype(np.int16)

        from isaaclab.utils import configclass
        @configclass
        class _OneWayStairsCfg(HfTerrainBaseCfg):
            function = _hf_stairs
            horizontal_scale: float = 0.05
            vertical_scale:   float = 0.005

        sub = {"stairs": _OneWayStairsCfg(
            proportion=1.0, size=(2.0, 2.0),
            horizontal_scale=0.05, vertical_scale=0.005, border_width=0.0,
        )}

    return TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(**gen_base, sub_terrains=sub),
        collision_group=-1, physics_material=phys, debug_vis=False,
    )


class PlayMetricsRecorder:
    """播放期間累積 per-step locomotion 指標，結束時輸出 CSV + 統計表 + 折線圖。"""

    # (name, fn, unit, lower_is_better, vel_metric, hybrid_only)
    METRIC_SPECS = [
        ("lin_vel_tracking_mae",  "lin_vel_tracking_error",  "m/s",   True,  True,  False),
        ("ang_vel_tracking_mae",  "ang_vel_tracking_error",  "rad/s", True,  True,  False),
        ("cost_of_transport",     "cost_of_transport",       "-",     True,  False, False),
        ("body_orientation_rms",  "body_orientation_rms",    "rad",   True,  False, False),
        ("base_height_deviation", "base_height_deviation",   "m",     True,  False, False),
        ("relative_height_dev",   "relative_height_dev",     "m",     True,  False, False),
        ("stair_traversal_dist",  "stair_traversal_dist",    "m",     False, False, False),
        ("action_smoothness",     "action_smoothness",       "-",     True,  False, False),
        ("gait_symmetry_std",     "gait_symmetry_std",       "-",     False, False, False),
        ("foot_slip_rate",        "foot_slip_rate",          "m/s",   True,  False, False),
        ("rl_contribution_rate",  "rl_contribution_rate",    "-",     False, False, True),
    ]

    def __init__(self, isaac_env, out_dir: str, task_name: str,
                 skip_vel_metrics: bool = False, hybrid: bool = False):
        import os
        from isaaclab.managers import SceneEntityCfg
        self.env = isaac_env
        self.out_dir = out_dir
        self.task_name = task_name
        os.makedirs(out_dir, exist_ok=True)

        # 過濾：vel_metric + hybrid_only
        self.active_specs = [
            s for s in self.METRIC_SPECS
            if not (skip_vel_metrics and s[4])
            and not (s[5] and not hybrid)
        ]

        # 解析 contact sensor / foot asset 的 body_ids（只做一次）
        self.foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["leg_.*_3"])
        self.foot_sensor_cfg.resolve(isaac_env.scene)
        self.foot_asset_cfg = SceneEntityCfg("robot", body_names=["leg_.*_3"])
        self.foot_asset_cfg.resolve(isaac_env.scene)

        self.history = {spec[0]: [] for spec in self.active_specs}
        self.phase_log = []        # per-step 地形相位
        self.height_log = []       # per-step mean body_z
        self.ep_lens = []          # 每個完成 episode 的存活步數
        self._prev_ep_buf = None   # 上一 step 的 episode_length_buf
        self.step = 0

    def record(self):
        """每個 env.step 後呼叫一次。"""
        import torch
        with torch.inference_mode():
            for name, fn_name, _unit, _lib, _vel, _hyb in self.active_specs:
                fn = getattr(_spider_metrics, fn_name)
                try:
                    if name in ("gait_symmetry_std",):
                        v = fn(self.env, sensor_cfg=self.foot_sensor_cfg)
                    elif name == "foot_slip_rate":
                        v = fn(self.env, sensor_cfg=self.foot_sensor_cfg, asset_cfg=self.foot_asset_cfg)
                    elif name == "relative_height_dev":
                        v = fn(self.env, sensor_cfg=self.foot_sensor_cfg, asset_cfg=self.foot_asset_cfg)
                    else:
                        v = fn(self.env)
                    self.history[name].append(float(v.mean().item()))
                except Exception:
                    self.history[name].append(float("nan"))
            # 地形相位
            try:
                ph = _spider_metrics.terrain_phase_id(self.env)
                self.phase_log.append(float(ph.mean().item()))
            except Exception:
                self.phase_log.append(float("nan"))

            # 高度 log
            try:
                bz = self.env.scene["robot"].data.root_pos_w[:, 2]
                self.height_log.append(float(bz.mean().item()))
            except Exception:
                self.height_log.append(float("nan"))

            # episode 存活長度（偵測 reset）
            try:
                cur = self.env.episode_length_buf.cpu().clone()
                if self._prev_ep_buf is not None:
                    resets = cur < self._prev_ep_buf
                    if resets.any():
                        self.ep_lens.extend(self._prev_ep_buf[resets].tolist())
                self._prev_ep_buf = cur
            except Exception:
                pass
        self.step += 1

    def summary(self) -> dict:
        """回傳每個指標的 mean / std / min / max。"""
        import numpy as np
        out = {}
        for name, _fn, unit, lower_better, _vel, _hyb in self.active_specs:
            arr = np.asarray(self.history[name], dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            out[name] = {
                "mean": float(arr.mean()),
                "std":  float(arr.std()),
                "min":  float(arr.min()),
                "max":  float(arr.max()),
                "unit": unit,
                "lower_is_better": lower_better,
            }
        return out

    def save(self):
        """輸出 CSV、summary.txt、PNG 折線圖。"""
        import os, csv, numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. CSV (per-step)
        csv_path = os.path.join(self.out_dir, "play_metrics.csv")
        keys = [s[0] for s in self.active_specs]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + keys)
            n = max((len(self.history[k]) for k in keys), default=0)
            for i in range(n):
                row = [i] + [self.history[k][i] if i < len(self.history[k]) else "" for k in keys]
                w.writerow(row)

        # 2. summary.txt
        summ = self.summary()
        summ_path = os.path.join(self.out_dir, "play_metrics_summary.txt")
        with open(summ_path, "w") as f:
            f.write(f"Task: {self.task_name}   Steps collected: {self.step}\n")
            f.write("=" * 78 + "\n")
            f.write(f"{'metric':<24}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}  unit  better\n")
            f.write("-" * 78 + "\n")
            for name, s in summ.items():
                arrow = "↓" if s["lower_is_better"] else "—"
                f.write(f"{name:<24}{s['mean']:>12.4f}{s['std']:>12.4f}"
                        f"{s['min']:>12.4f}{s['max']:>12.4f}  {s['unit']:<4}  {arrow}\n")

        # 3. PNG grid
        png_path = os.path.join(self.out_dir, "play_metrics.png")
        ncols = 2
        nrows = (len(keys) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
        axes = np.atleast_2d(axes)
        for idx, (name, _fn, unit, _, _v, _h) in enumerate(self.active_specs):
            ax = axes[idx // ncols, idx % ncols]
            data = np.asarray(self.history[name], dtype=float)
            if data.size == 0:
                ax.set_title(f"{name}  (no data)")
                continue
            ax.plot(data, linewidth=1.0)
            mean = np.nanmean(data)
            ax.axhline(mean, color="red", linestyle="--", alpha=0.6, label=f"mean={mean:.3f}")
            ax.set_title(f"{name} [{unit}]")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        for idx in range(len(keys), nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")
        fig.suptitle(f"Play Metrics — {self.task_name}", fontsize=14)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"\n[Metrics] CSV     : {csv_path}")
        print(f"[Metrics] Summary : {summ_path}")
        print(f"[Metrics] Plot    : {png_path}")
        print("\n" + open(summ_path).read())

        # ── 地形相位獨立輸出 ──
        if self.phase_log:
            import numpy as np
            ph = np.asarray(self.phase_log)
            ph = ph[~np.isnan(ph)]
            labels = ["flat(0)", "ascending(1)", "descending(2)"]
            counts = [(ph == i).sum() for i in range(3)]
            total = max(ph.size, 1)

            # phase_summary.txt
            ph_txt = os.path.join(self.out_dir, "terrain_phase_summary.txt")
            with open(ph_txt, "w") as f:
                f.write(f"Terrain Phase Distribution — {self.task_name}\n")
                f.write(f"Total steps: {ph.size}\n")
                f.write("-" * 40 + "\n")
                for lbl, cnt in zip(labels, counts):
                    f.write(f"  {lbl:<16}: {cnt:5d} steps  ({cnt/total*100:.1f}%)\n")

            # phase_plot.png (bar + timeline)
            ph_png = os.path.join(self.out_dir, "terrain_phase_plot.png")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            ax1.bar(labels, [c / total * 100 for c in counts],
                    color=["steelblue", "tomato", "seagreen"])
            ax1.set_ylabel("% of steps"); ax1.set_title("Phase Distribution")
            ax1.grid(axis="y", alpha=0.3)
            ax2.plot(ph, linewidth=0.6, alpha=0.8)
            ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels(labels)
            ax2.set_xlabel("step"); ax2.set_title("Phase Timeline")
            ax2.grid(alpha=0.3)
            fig.suptitle(f"Terrain Phase — {self.task_name}", fontsize=13)
            fig.tight_layout()
            fig.savefig(ph_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"[Metrics] Phase   : {ph_png}")
            print(open(ph_txt).read())

        # ── 高度時間曲線 ──
        if self.height_log:
            import numpy as np
            hz = np.asarray(self.height_log, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(hz, linewidth=0.8, color="steelblue")
            ax.set_xlabel("step"); ax.set_ylabel("mean body_z (m)")
            ax.set_title(f"Body Height Profile — {self.task_name}")
            ax.axhline(np.nanmean(hz), color="red", linestyle="--", alpha=0.5,
                       label=f"mean={np.nanmean(hz):.3f}m")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            h_png = os.path.join(self.out_dir, "height_profile.png")
            fig.savefig(h_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Metrics] Height  : {h_png}")

        # ── Episode 存活長度分布 ──
        if self.ep_lens:
            import numpy as np
            ep = np.asarray(self.ep_lens, dtype=float)
            max_ep = self.env.max_episode_length if hasattr(self.env, "max_episode_length") else ep.max()
            success_rate = float((ep >= max_ep * 0.95).mean()) * 100
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(ep, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(ep.mean(), color="red", linestyle="--",
                       label=f"mean={ep.mean():.0f} steps")
            ax.set_xlabel("episode length (steps)"); ax.set_ylabel("count")
            ax.set_title(f"Episode Survival — {self.task_name}  "
                         f"(success≥95% ep_len: {success_rate:.1f}%)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            ep_png = os.path.join(self.out_dir, "episode_survival.png")
            fig.savefig(ep_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            ep_txt = os.path.join(self.out_dir, "episode_survival_summary.txt")
            with open(ep_txt, "w") as f:
                f.write(f"Episodes recorded: {len(ep):.0f}\n"
                        f"Mean survival    : {ep.mean():.1f} steps\n"
                        f"Median           : {np.median(ep):.1f} steps\n"
                        f"Success rate     : {success_rate:.1f}%\n")
            print(f"[Metrics] Survival: {ep_png}")
            print(open(ep_txt).read())

        # ── RL vs CPG 幅值貢獻率堆疊圖（Hybrid 專用）──
        try:
            term = self.env.action_manager.get_term("joint_action")
            buf = getattr(term, "log_buffer", {})
            pm = np.asarray(buf.get("prior_magnitude", []), dtype=float)
            rm = np.asarray(buf.get("rl_magnitude",    []), dtype=float)
            if pm.size > 1 and rm.size > 1:
                w = 15  # 平滑窗口（≈1 步態週期）
                def _roll(arr):
                    # min_periods=1 避免首尾零填充造成的邊界下陷
                    import pandas as pd
                    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values
                x = np.arange(len(pm)) * 10  # 轉為實際步數

                fig, ax = plt.subplots(figsize=(12, 4))
                # 原始值（淡色）
                ax.plot(x, pm, color="steelblue", alpha=0.25, linewidth=0.7)
                ax.plot(x, rm, color="tomato",    alpha=0.25, linewidth=0.7)
                # 平滑後（實線）
                ax.plot(x, _roll(pm), color="steelblue", linewidth=2.0,
                        label=f"CPG ‖a_cpg‖  avg={pm.mean():.3f}")
                ax.plot(x, _roll(rm), color="tomato",    linewidth=2.0,
                        label=f"RL  ‖a_rl‖   avg={rm.mean():.3f}")
                ax.set_ylabel("Action L2 magnitude")
                ax.set_xlabel("step")
                ax.set_title(f"RL vs CPG Output Magnitude — {self.task_name}")
                ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.2)
                fig.tight_layout()
                rl_png = os.path.join(self.out_dir, "rl_cpg_contribution.png")
                fig.savefig(rl_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[Metrics] RL/CPG  : {rl_png}")
                print(f"  avg RL%={rl_pct.mean():.1f}%  avg CPG%={cpg_pct.mean():.1f}%")
        except Exception:
            pass

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 地形覆蓋（--eval_terrain）
    if args_cli.eval_terrain:
        print(f"[INFO] Overriding terrain to: {args_cli.eval_terrain}")
        env_cfg.scene.terrain = _make_eval_terrain(args_cli.eval_terrain)

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # ── 建立 metrics recorder ──
    isaac_env = env.unwrapped.env if hasattr(env.unwrapped, "env") else env.unwrapped
    metrics_out = args_cli.metrics_out or os.path.join(log_dir, "metrics")
    recorder = PlayMetricsRecorder(isaac_env, metrics_out, task_name=train_task_name,
                                    skip_vel_metrics=args_cli.skip_vel_metrics,
                                    hybrid=args_cli.hybrid)

    # Ctrl+C 也能存檔
    import signal
    def _save_on_exit(sig, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[Metrics] Ctrl+C 捕捉到，輸出當前指標...")
        try:
            recorder.save()
        except Exception as e:
            print(f"[Metrics] 存檔失敗: {e}")
        finally:
            os._exit(0)
    signal.signal(signal.SIGINT, _save_on_exit)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # 紀錄 metrics（跳過 video-only 模式的 warmup）
        recorder.record()

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        elif args_cli.metrics_steps > 0 and recorder.step >= args_cli.metrics_steps:
            print(f"[Metrics] 已蒐集 {recorder.step} 步，輸出結果並結束。")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 輸出 metrics
    try:
        recorder.save()
    except Exception as e:
        print(f"[Metrics] 存檔失敗: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
