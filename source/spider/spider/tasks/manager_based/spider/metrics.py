"""
量化評估指標 (Locomotion RL Benchmarks)
=========================================

這些指標是比較 Pure-RL、Pure-CPG、Hybrid 三種控制策略的「標準語言」，
設計參考自近年多足機器人 RL 論文 (Rudin et al. 2022, ANYmal; Margolis et al.,
Rapid Locomotion; MIT Cheetah; Hwangbo et al., Sim-to-Real)。

每個指標都是 per-env、per-step 的 `torch.Tensor [N]`，可直接掛成 `RewTerm`
讓 skrl / Isaac Lab 的 reward_manager 自動產生
`Episode_Reward/<term_name>` 的 tensorboard scalar。

建議使用方式
------------
在 `RewardsCfg` 中加入 (weight=1e-10 即可被 reward_manager 計算並記錄，
但數值上不會干擾訓練)：

    metric_lin_vel_mae = RewTerm(
        func=metrics.lin_vel_tracking_error,
        weight=1e-10,
        params={"command_name": "base_velocity"},
    )

-- 指標對應關係 --
| 面向          | 指標                    | Pure-RL 預期  | Hybrid 預期   |
|--------------|------------------------|--------------|--------------|
| 任務達成      | lin/ang_vel_tracking   | 慢達到最佳   | 前期快        |
| 能量效率      | cost_of_transport      | 通常較高      | 較低 (CPG 韻律) |
| 平衡穩定性    | body_stability_rms     | 依 reward 調節 | CPG 提供先驗較穩 |
| 步態規律性    | gait_symmetry_std      | 可能不對稱    | CPG 保證對稱   |
| 腳底滑動      | foot_slip_rate         | 初期高        | 低            |
| 動作平滑度    | action_smoothness      | 高頻抖動風險   | CPG 先驗平滑   |
| 存活 / 韌性   | survival_rate (play)   | 需夠長訓練    | 初期較佳      |
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────
# 1. 指令追蹤誤差 (MAE) — 越小越好
# ──────────────────────────────────────────────────────────────────────

def lin_vel_tracking_error(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """線速度追蹤 MAE (m/s)。取 body-frame 的 xy 誤差範數。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = asset.data.root_lin_vel_b[:, :2]
    return torch.norm(cmd - vel, dim=-1)


def ang_vel_tracking_error(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw 角速度追蹤 MAE (rad/s)。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    return torch.abs(cmd_yaw - ang_vel_z)


# ──────────────────────────────────────────────────────────────────────
# 2. 能量效率 — Cost of Transport，越小越好
# ──────────────────────────────────────────────────────────────────────

def cost_of_transport(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_speed: float = 0.1,
    gravity: float = 9.81,
) -> torch.Tensor:
    r"""
    Cost of Transport (無量綱)：
        CoT = \sum_j |\tau_j \cdot \dot q_j| / (m g \|v\|)
    低速時 (≤ min_speed) 以 min_speed 夾住避免分母為 0 / 爆炸。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque  # [N, J]
    qd = asset.data.joint_vel         # [N, J]
    power = torch.sum(torch.abs(tau * qd), dim=-1)  # [N]

    mass = asset.data.default_mass.sum(dim=-1).to(env.device)  # [N]
    speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    speed = torch.clamp(speed, min=min_speed)
    return power / (mass * gravity * speed + 1e-6)


# ──────────────────────────────────────────────────────────────────────
# 3. 姿態穩定性 — RMS of pitch/roll，越小越好
# ──────────────────────────────────────────────────────────────────────

def body_orientation_rms(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    身體水平度偏差 (rad)：直接用 projected_gravity_b 的 xy 範數。
    站直時為 0；越大代表身體越傾斜。對 noise 地形很關鍵。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)


def base_height_deviation(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.14,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """身體高度誤差 (m)：|z - target_z|。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height)


# ──────────────────────────────────────────────────────────────────────
# 4. 步態規律性 — 六腳 duty factor 的標準差
# ──────────────────────────────────────────────────────────────────────

def gait_symmetry_std(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    六隻腳「目前是否接觸地面」的 std。
    - 三腳步態 (tripod) 期望永遠有 3 接觸 / 3 懸空 → std ≈ 0.5 (對稱)
    - 亂走 / 拖腳 → std 可能接近 0 (全部都接地) 或不穩
    把這個值視為「步態結構性」指標（數值本身不代表好壞，但與標稱值偏離大 = 不規則）。
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # [N, H, F, 3]
    contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0                     # [N, F]
    return contact.float().std(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# 5. 腳底滑動率 — 接觸中的腳側向速度，越小越好
# ──────────────────────────────────────────────────────────────────────

def foot_slip_rate(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    在 foot-down 的瞬間，腳在地面上的 xy 速度大小 (m/s)。等同於 Isaac Lab
    的 feet_slide，但回傳未加權值供作評估。
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contact, dim=-1)


# ──────────────────────────────────────────────────────────────────────
# 6. 動作平滑度 — 二階動作差分
# ──────────────────────────────────────────────────────────────────────

def action_smoothness(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Δ²a = a_t - 2 a_{t-1} + a_{t-2} 的 L2 範數。
    比 action_rate (一階) 更敏感於高頻抖動，常用於衡量是否能直接部署到硬體。
    """
    am = env.action_manager
    a0 = am.action
    a1 = am.prev_action
    # 部分版本沒有 prev_prev；退化成 |Δa|
    a2 = getattr(am, "prev_prev_action", a1)
    return torch.norm(a0 - 2.0 * a1 + a2, dim=-1)


# ──────────────────────────────────────────────────────────────────────
# 7. RL 實際貢獻率 — Hybrid 專用，量化 RL 修正量佔總輸出的比例
# ──────────────────────────────────────────────────────────────────────

def rl_contribution_rate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    從 Hybrid action term 的 log_buffer 讀取最新的 RL 貢獻率。
    定義：Var(rl_part) / Var(combined)，範圍 [0, 1]。
    - 接近 0：CPG 主導，RL 幾乎沒有修正
    - 接近 1：RL 完全主導，CPG prior 幾乎被蓋過
    非 Hybrid 環境會回傳全 0（不會 crash）。
    """
    try:
        term = env.action_manager.get_term("joint_action")
        buf = getattr(term, "log_buffer", {})
        history = buf.get("rl_contribution", [])
        val = history[-1] if history else 0.0
        return torch.full((env.num_envs,), val, dtype=torch.float32, device=env.device)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────────────────────────────
# 8. 地形相位辨識 — 每 step 回傳 0=平地 / 1=爬坡 / 2=下坡
# ──────────────────────────────────────────────────────────────────────

def terrain_phase_id(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vz_threshold: float = 0.03,
) -> torch.Tensor:
    """
    依 body 垂直速度判斷當前地形相位（per-env）：
      0 = flat    |vz| ≤ threshold
      1 = ascending  vz > threshold
      2 = descending vz < -threshold
    搭配 play.py 的 phase log 可輸出三段各自的指標分布圖。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    phase = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    phase[vz >  vz_threshold] = 1.0
    phase[vz < -vz_threshold] = 2.0
    return phase


# ──────────────────────────────────────────────────────────────────────
# 9. 爬樓梯位移 — 每 step 在階梯方向（+x）的前進距離，越大越好
# ──────────────────────────────────────────────────────────────────────

def stair_traversal_dist(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    每 step body 在 world +x 方向的位移（即 |vx| * dt）。
    階梯沿 x 軸展開，累加後代表爬過多遠。
    兩者都有 RL 速度指令時可用 vx；純 CPG 不追蹤指令也能量化實際前進量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_w[:, 0]   # world frame x 速度
    return torch.abs(vx) * env.step_dt


# ──────────────────────────────────────────────────────────────────────
# 8. 相對地形高度偏差 — body z 相對於當前支撐腳的高度，越穩定越好
# ──────────────────────────────────────────────────────────────────────

def relative_height_dev(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_clearance: float = 0.14,
) -> torch.Tensor:
    """
    body z − 接觸腳 z 的平均值，再減去目標淨空（clearance）。
    在平地時等同於 base_height_deviation；在階梯上能反映機器人相對地形的高度。
    若完全沒有腳接地（跌倒），退化為 |body_z|。
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :] \
                .norm(dim=-1).max(dim=1)[0] > 1.0            # [N, F]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] # [N, F]

    # 有接地腳時取平均，否則用 0 作為地面高度
    n_contact = contact.float().sum(dim=-1).clamp(min=1.0)
    terrain_z = (foot_z * contact.float()).sum(dim=-1) / n_contact

    body_z = asset.data.root_pos_w[:, 2]
    return torch.abs((body_z - terrain_z) - target_clearance)


# ──────────────────────────────────────────────────────────────────────
# 9. 存活指示 — 每 step 回傳 1 (episode_sum_avg / T 會收斂到 survival_rate)
# ──────────────────────────────────────────────────────────────────────

def survival_step(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """每步回傳 1。Episode_Reward/metric_survival ≈ 存活率 × dt_ratio。"""
    return torch.ones(env.num_envs, device=env.device)
