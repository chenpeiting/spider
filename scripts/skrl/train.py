# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
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
import logging
import os
import random
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # ✅ 新增

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import spider.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# ════════════════════════════════════════════════════════════════════
# ✅ 新增：RL-CPG 監控回調類（必須在 main() 函數之外定義）
# ════════════════════════════════════════════════════════════════════
class RLCPGMonitoringCallback:
    """RL-CPG 監控回調函數"""

    def __init__(self, env, tb_writer, log_interval=100):
        """
        參數:
            env: Isaac Lab 環境
            tb_writer: Tensorboard SummaryWriter
            log_interval: 每多少步記錄一次
        """
        self.env = env
        self.tb_writer = tb_writer
        self.log_interval = log_interval
        self.step_count = 0

    def __call__(self, timestep, timesteps):
        """skrl 會在每個訓練步調用這個函數"""
        self.step_count += 1

        # 每 log_interval 步記錄一次
        if self.step_count % self.log_interval == 0:
            try:
                # 獲取 action manager
                unwrapped_env = self.env.unwrapped

                # 檢查是否是 wrapped 環境
                if hasattr(unwrapped_env, 'env'):
                    isaac_env = unwrapped_env.env
                else:
                    isaac_env = unwrapped_env

                # 獲取 action term
                action_term = isaac_env.action_manager.get_term("joint_action")

                # 獲取監控數據
                log_buffer = action_term.get_log_buffer()

                if log_buffer['correlation']:  # 確保有數據
                    # 取最新的指標
                    latest_corr = log_buffer['correlation'][-1]
                    latest_contrib = log_buffer['rl_contribution'][-1]
                    latest_prior_mag = log_buffer['prior_magnitude'][-1]
                    latest_rl_mag = log_buffer['rl_magnitude'][-1]

                    # 記錄到 Tensorboard
                    self.tb_writer.add_scalar('RL-CPG/correlation', latest_corr, self.step_count)
                    self.tb_writer.add_scalar('RL-CPG/rl_contribution', latest_contrib, self.step_count)
                    self.tb_writer.add_scalar('RL-CPG/prior_magnitude', latest_prior_mag, self.step_count)
                    self.tb_writer.add_scalar('RL-CPG/rl_magnitude', latest_rl_mag, self.step_count)
                    self.tb_writer.add_scalar('RL-CPG/magnitude_ratio',
                                             latest_rl_mag / (latest_prior_mag + 1e-8),
                                             self.step_count)
                    # 原本的 add_scalar 後面加：
                    if 'terrain_level' in log_buffer and log_buffer['terrain_level']:
                        self.tb_writer.add_scalar(
                            'RL-CPG/terrain_level',
                            log_buffer['terrain_level'][-1],
                            self.step_count
                        )
                        # 把 rl_contribution 對 terrain level 作圖（scatter 概念）
                        self.tb_writer.add_scalar(
                            'RL-CPG/contribution_per_level',
                            latest_contrib / (log_buffer['terrain_level'][-1] + 0.1),
                            self.step_count
                        )

                    # 可選：每 1000 步打印一次
                    if self.step_count % 1000 == 0:
                        print(f"[RL-CPG Monitor] Step {self.step_count}: "
                              f"Correlation={latest_corr:.3f}, "
                              f"Contribution={latest_contrib:.3f}")

            except Exception as e:
                # 靜默處理錯誤（訓練初期可能還沒有數據）
                pass
# ════════════════════════════════════════════════════════════════════


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ════════════════════════════════════════════════════════════
    # ✅ 新增：創建 Tensorboard Writer
    # ════════════════════════════════════════════════════════════
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"[INFO] Tensorboard logging to: {tensorboard_dir}")
    print(f"[INFO] 啟動 Tensorboard: tensorboard --logdir={tensorboard_dir}")
    # ════════════════════════════════════════════════════════════

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    
    # ════════════════════════════════════════════════════════════
    # ✅ 新增：設定監控回調
    # ════════════════════════════════════════════════════════════
    monitoring_callback = RLCPGMonitoringCallback(
        env=env, 
        tb_writer=tb_writer, 
        log_interval=100  # 每 100 步記錄一次
    )
    
    # 保存原始的 post_interaction
    original_post_interaction = runner.agent.post_interaction
    
    def custom_post_interaction(timestep, timesteps):
        """自定義的 post_interaction，加入監控"""
        # 調用原始的 post_interaction
        if original_post_interaction is not None:
            original_post_interaction(timestep, timesteps)
        # 調用監控回調
        monitoring_callback(timestep, timesteps)
    
    # 替換 agent 的 post_interaction
    runner.agent.post_interaction = custom_post_interaction
    # ════════════════════════════════════════════════════════════

    import signal

    def save_on_exit(sig, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # ← 先移除 handler，避免重複觸發
        print("\n[INFO] 捕捉到 Ctrl+C，正在保存圖表...")
        try:
            isaac_env = env.unwrapped.env if hasattr(env.unwrapped, 'env') else env.unwrapped
            action_term = isaac_env.action_manager.get_term("joint_action")
            action_term.save_monitoring_plot(os.path.join(log_dir, 'rl_cpg_interrupted.png'))
            print("[INFO] 圖表已保存！")
        except Exception as e:
            print(f"[WARNING] 無法保存: {e}")
        finally:
            tb_writer.close()  # ← 加這行，確保 TensorBoard 資料也寫入
            os._exit(0)  # ← 強制退出，不走 Python cleanup 流程

    signal.signal(signal.SIGINT, save_on_exit)

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # ════════════════════════════════════════════════════════════
    # ✅ 新增：訓練結束後生成最終圖表
    # ════════════════════════════════════════════════════════════
    try:
        isaac_env = env.unwrapped.env if hasattr(env.unwrapped, 'env') else env.unwrapped
        action_term = isaac_env.action_manager.get_term("joint_action")
        final_plot_path = os.path.join(log_dir, 'rl_cpg_final_analysis.png')
        action_term.save_monitoring_plot(final_plot_path)
        print(f"[INFO] 最終分析圖表已保存: {final_plot_path}")
    except Exception as e:
        print(f"[WARNING] 無法生成最終圖表: {e}")
    
    # 關閉 Tensorboard writer
    tb_writer.close()
    # ════════════════════════════════════════════════════════════

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()