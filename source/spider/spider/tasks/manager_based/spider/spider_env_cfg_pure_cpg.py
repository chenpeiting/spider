# 完整修复版 - 解决绕圈问题
import torch
import math
import torch.nn.functional as F
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import RayCasterCfg, patterns

# 使用官方的 velocity_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from . import mdp

from isaaclab.terrains.height_field import (
    HfRandomUniformTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
)

import numpy as np
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh

from .spider_ik_cpg import IKCPGActionCfg


@height_field_to_mesh
def hf_up_down_stairs(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
    length_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    width_pixels  = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((length_pixels, width_pixels), dtype=np.float32)
    step_width_pixels = max(1, int(0.20 / cfg.horizontal_scale))
    step_height_units = 0.08 / cfg.vertical_scale
    for i in range(length_pixels):
        dist = min(i, length_pixels - 1 - i)
        hf_raw[i, :] = (dist // step_width_pixels) * step_height_units
    return np.rint(hf_raw).astype(np.int16)


@configclass
class OneWayStairsCfg(HfTerrainBaseCfg):
    function = hf_up_down_stairs
    horizontal_scale: float = 0.05
    vertical_scale:   float = 0.005

# ════════════════════════════════════════════════════════════════════
# ✅ 辅助函数：解决绕圈问题
# ════════════════════════════════════════════════════════════════════

def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """
    从四元数提取 Yaw 角度
    
    Args:
        quat: 四元数 [N, 4]，格式为 [w, x, y, z]
    
    Returns:
        yaw: Yaw 角度 [N]，范围 [-π, π]
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return yaw


def heading_projection(env, command_name: str = "base_velocity") -> torch.Tensor:
    """
    计算航向误差（机器人朝向与目标方向的夹角）
    
    Returns:
        [sin(Δθ), cos(Δθ)] - 使用 sin/cos 编码避免不连续性
    """
    # 获取速度指令
    command = env.command_manager.get_command(command_name)
    
    # 计算目标航向角（从速度指令）
    cmd_heading = torch.atan2(command[:, 1], command[:, 0])  # [N]
    
    # 获取机器人当前朝向
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w  # [N, 4]
    robot_heading = quat_to_yaw(quat)  # [N]
    
    # 计算航向误差
    heading_error = cmd_heading - robot_heading
    
    # 使用 sin/cos 编码（避免 ±π 不连续性）
    return torch.stack([
        torch.sin(heading_error),
        torch.cos(heading_error)
    ], dim=-1)


def projected_velocity_commands(env, command_name: str = "base_velocity") -> torch.Tensor:
    """
    将世界坐标系的速度指令投影到机器人本体坐标系
    
    这样机器人就知道：相对于自己当前的朝向，应该往哪个方向移动
    """
    # 获取世界坐标系的指令
    command = env.command_manager.get_command(command_name)  # [N, 3]
    cmd_world = command[:, :2]  # [N, 2] (v_x, v_y)
    
    # 获取机器人当前朝向
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w
    yaw = quat_to_yaw(quat)
    
    # 旋转矩阵（世界坐标系 → 本体坐标系）
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)
    
    # 投影到本体坐标系
    cmd_body_x = cmd_world[:, 0] * cos_yaw - cmd_world[:, 1] * sin_yaw
    cmd_body_y = cmd_world[:, 0] * sin_yaw + cmd_world[:, 1] * cos_yaw
    
    return torch.stack([cmd_body_x, cmd_body_y, command[:, 2]], dim=-1)


def gait_phase(env, action_name: str = "joint_action") -> torch.Tensor:
    """返回步态相位（sin 和 cos）"""
    action_term = env.action_manager.get_term(action_name)
    return torch.cat([action_term.sin_phase, action_term.cos_phase], dim=-1)


class TripodActionWithLogging(JointPositionAction):
    """带监控功能的三脚步态控制器"""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._isaac_env = env  # ← 直接存，不依賴父類屬性名
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.dt = env.step_dt
        self.freq = 2.0
        
        # 必须定义（gait_phase 会用到）
        self.sin_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.cos_phase = torch.ones(self.num_envs, 1, device=self.device)
        
        # 监控缓冲区
        self.log_buffer = {
            'prior_magnitude': [],
            'rl_magnitude': [],
            'correlation': [],
            'rl_contribution': [],
            'step_count': 0
        }
        
        # RL 权重（必须与 process_actions 中的一致）
        self.rl_weight = 0.0
        
    def process_actions(self, actions: torch.Tensor):
        self.phase += 2 * math.pi * self.freq * self.dt
        self.phase = torch.fmod(self.phase, 2 * math.pi)
        
        self.sin_phase[:, 0] = torch.sin(self.phase)
        self.cos_phase[:, 0] = torch.cos(self.phase)

        prior = torch.zeros_like(actions)
        
        # Hopf oscillator 的 y 輸出 ≈ sin(phase)
        y_A = torch.sin(self.phase)       # A 組
        y_B = torch.sin(self.phase + math.pi)  # B 組
        
        # ẏ > 0 = swing phase
        dy_A = torch.cos(self.phase)
        dy_B = torch.cos(self.phase + math.pi)
        
        A1 = 0.3  # hip amplitude 0.3 - 15
        A2 = 0.6 # femur amplitude  0.6 - 0.3
        A3 = 0.4  # tibia amplitude 0.4 - 0.2
        
        # 論文映射函數
        def mapping(y, dy):
            hip   = A1 * y
            femur = A2 * (1 - y**2) * (dy >= 0).float()
            tibia = -A3 * femur
            return hip, femur, tibia
        
        hip_A, fem_A, tib_A = mapping(y_A, dy_A)
        hip_B, fem_B, tib_B = mapping(y_B, dy_B)

        # A 組（leg0, leg2, leg5）
        prior[:, 0]  =-hip_A;  prior[:, 6]  = fem_A;  prior[:, 12]  = tib_A
        prior[:, 2]  =-hip_A;  prior[:, 8]  = fem_A;  prior[:, 14]  = tib_A
        prior[:, 5]  = hip_A;  prior[:, 11] = fem_A;  prior[:, 17]  = tib_A

        # B 組（leg1, leg3, leg4）
        prior[:, 1]  =-hip_B;  prior[:, 7]  = fem_B;  prior[:, 13] = tib_B
        prior[:, 3]  = hip_B;  prior[:, 9]  = fem_B;  prior[:, 15] = tib_B
        prior[:, 4]  = hip_B;  prior[:, 10] = fem_B;  prior[:, 16] = tib_B
                
        # 合并
        rl_part = actions * self.rl_weight
        combined_actions = rl_part + prior
        
        # 监控
        self.log_buffer['step_count'] += 1
        if self.log_buffer['step_count'] % 10 == 0:
            self._log_metrics(prior, rl_part, actions)
        
        super().process_actions(combined_actions)

    # def process_actions(self, actions: torch.Tensor):
    #     self.phase += 2 * math.pi * self.freq * self.dt
    #     self.phase = torch.fmod(self.phase, 2 * math.pi)
    #     self.sin_phase[:, 0] = torch.sin(self.phase)
    #     self.cos_phase[:, 0] = torch.cos(self.phase)
        
    #     # ═══ 診斷模式：固定正值，不擺動 ═══
    #     prior = torch.zeros_like(actions)
        
    #     # 一次只開一行，其他保持註解
    #     # prior[:, 0] = 0.5    # leg0 hip：正值 0.5 rad
    #     # prior[:, 1] = 0.5  # leg0 femur
    #     # prior[:, 9] = 0.5  # leg3 hip
    #     # prior[:, 10] = 0.5 # leg3 femur
    #     # prior[:, 15] = 0.5 # leg5 hip
    #     # prior[:, 16] = 0.5 # leg5 femur
        
    #     combined_actions = prior + actions * self.rl_weight
    #     super().process_actions(combined_actions)
    
    def _log_metrics(self, prior, rl_part, raw_actions):
        with torch.no_grad():
            prior_mag = torch.norm(prior, dim=1).mean().item()
            rl_mag = torch.norm(rl_part, dim=1).mean().item()
            
            prior_flat = prior.reshape(prior.shape[0], -1)
            rl_flat = rl_part.reshape(rl_part.shape[0], -1)
            cos_sim = F.cosine_similarity(prior_flat, rl_flat, dim=1).mean().item()
            
            combined = rl_part + prior
            # ✅ 換成這兩行
            rl_var_per_env = torch.var(rl_part, dim=1)
            combined_var_per_env = torch.var(combined, dim=1)
            rl_contribution = (rl_var_per_env / (combined_var_per_env + 1e-8)).mean().item()
            
            self.log_buffer['prior_magnitude'].append(prior_mag)
            self.log_buffer['rl_magnitude'].append(rl_mag)
            self.log_buffer['correlation'].append(cos_sim)
            self.log_buffer['rl_contribution'].append(rl_contribution)
            # ★ 新增：記錄當前 terrain level（如果環境有這個資訊）
            # ✓ 改成
            if hasattr(self, '_isaac_env') and hasattr(self._isaac_env, 'scene'):
                try:
                    terrain = self._isaac_env.scene.terrain
                    if hasattr(terrain, 'terrain_levels'):
                        avg_level = terrain.terrain_levels.float().mean().item()
                        self.log_buffer.setdefault('terrain_level', []).append(avg_level)
                except Exception:
                    pass
    
    def get_log_buffer(self):
        return self.log_buffer
    
    def save_monitoring_plot(self, save_path='rl_cpg_analysis.png'):
        from .monitoring_plot import plot_rl_cpg_analysis
        plot_rl_cpg_analysis(self.log_buffer, self.rl_weight, save_path)
    
    # ✓ 修正
    def reset(self, env_ids: torch.Tensor | None = None):
        super().reset(env_ids)
        if env_ids is None:
            self.phase[:] = 0.0
        else:
            self.phase[env_ids] = 0.0


@configclass
class TripodActionCfg(JointPositionActionCfg):
    class_type: type = TripodActionWithLogging




# ════════════════════════════════════════════════════════════════════
# 场景配置
# ════════════════════════════════════════════════════════════════════

@configclass
class SpiderSceneCfg(InteractiveSceneCfg):
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.001)
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(2.0, 2.0),
            border_width=0.01,
            num_rows=32,
            num_cols=32,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": HfRandomUniformTerrainCfg(
                    proportion=0.1,
                    noise_range=(0.0, 0.01),
                    noise_step=0.005,
                    border_width=0.1,
                ),
                "rough": HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(0.02, 0.06),
                    noise_step=0.01,
                    border_width=0.1,
                ),
                "stairs": OneWayStairsCfg(
                    proportion=0.5,
                    size=(2.0, 2.0),
                    horizontal_scale=0.05,
                    vertical_scale=0.005,
                    border_width=0.1,
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    
    # 加入高度掃描感測器
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    # dome_light, contact_forces, robot 不動
    # 只改 robot 的初始高度：
    # pos=(0.0, 0.0, 0.18)  ← 0.12 → 0.18

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=1500.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/.*",
        history_length=3,
        track_air_time=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"assets/spider_mesh_6legs/spider_mesh_6legs.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,   # False → True
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18),
            rot=(1.0, 0.0, 0.0, 0.0), #原本1, 0, 0, 0 ///0602
            joint_pos={
                "body_leg_0": 0.0, "leg_0_1_2": 0.0, "leg_0_2_3": 0.0,
                "body_leg_1": 0.0, "leg_1_1_2": 0.0, "leg_1_2_3": 0.0,
                "body_leg_2": -0.8, "leg_2_1_2": 0.0, "leg_2_2_3": 0.0,
                "body_leg_3": 0.8, "leg_3_1_2": 0.0, "leg_3_2_3": 0.0,
                "body_leg_4": 0.0, "leg_4_1_2": 0.0, "leg_4_2_3": 0.0,
                "body_leg_5": 0.0, "leg_5_1_2": 0.0, "leg_5_2_3": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=15.0,
                stiffness=60.0,
                damping=2.0,
                velocity_limit_sim=42.0
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


@configclass
class ActionsCfg:
    joint_action = TripodActionCfg(
        asset_name="robot",
        joint_names=[
            "body_leg_0", "leg_0_1_2", "leg_0_2_3",
            "body_leg_1", "leg_1_1_2", "leg_1_2_3",
            "body_leg_2", "leg_2_1_2", "leg_2_2_3",
            "body_leg_3", "leg_3_1_2", "leg_3_2_3",
            "body_leg_4", "leg_4_1_2", "leg_4_2_3",
            "body_leg_5", "leg_5_1_2", "leg_5_2_3",       
        ],
        scale=1.0,
    )



@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), 
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,               
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


# ════════════════════════════════════════════════════════════════════
# ✅ 修改：添加航向信息，解决绕圈问题
# ════════════════════════════════════════════════════════════════════

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 原有观测
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        
        # ✅ 修改 1: 使用投影后的速度指令（本体坐标系）
        velocity_commands = ObsTerm(
            func=projected_velocity_commands,
            params={"command_name": "base_velocity"}
        )
        
        # ✅ 修改 2: 添加航向误差（让机器人知道自己的朝向）
        heading_error = ObsTerm(
            func=heading_projection,
            params={"command_name": "base_velocity"}
        )
        
        # 步态相位
        phase = ObsTerm(func=gait_phase, params={"action_name": "joint_action"})

        # ✓ 改成
        height_scan = ObsTerm(
            func=velocity_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=velocity_mdp.track_lin_vel_xy_exp,
        weight=1.0,          # 2.0 → 1.0，降低「衝速度」的誘因
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    flat_orientation_l2 = RewTerm(func=velocity_mdp.flat_orientation_l2, weight=-0.5)# -1.0 → -2.0，身體傾斜懲罰加倍 -2.0 to -0.5 (stairs)
    ang_vel_xy_l2 = RewTerm(func=velocity_mdp.ang_vel_xy_l2, weight=-0.05)
    base_height_l2 = RewTerm(
        func=velocity_mdp.base_height_l2, weight=-0.2, #-1.0 to -0.2
        params={"target_height": 0.14, "asset_cfg": SceneEntityCfg("robot")}
    )
    undesired_contacts = RewTerm(
        func=velocity_mdp.undesired_contacts, weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), "threshold": 1.0}
    )
    action_rate_l2 = RewTerm(func=velocity_mdp.action_rate_l2, weight=-0.005) #-0.01 to -0.005
    joint_vel_l2 = RewTerm(func=velocity_mdp.joint_vel_l2, weight=-2e-5) #-5e-5 to -2e-5
    joint_acc_l2 = RewTerm(func=velocity_mdp.joint_acc_l2, weight=-1e-7) #-2.5e-7 to -1e-7
    lin_vel_z_l2 = RewTerm(func=velocity_mdp.lin_vel_z_l2, weight=-0.2) #-0.5 to -0.2
    joint_torques_l2 = RewTerm(func=velocity_mdp.joint_torques_l2, weight=-1e-5)
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time, weight=1.5,  # 0.5 → 1.5，最關鍵的修改
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_.*_3"]),
            "command_name": "base_velocity",
            "threshold": 0.4,# 0.5 → 0.4，稍微放寬抬腳時間要求
        }
    )
    # 在 RewardsCfg 最後加這一項
    joint_deviation_l2 = RewTerm(
        func=velocity_mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_2_3"])}
        # 只針對 tibia（小腿）關節，這是最容易卡在奇怪角度的關節
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})

@configclass
class CurriculumCfg:
    pass
    # terrain_levels = CurrTerm(
    #     func=velocity_mdp.terrain_levels_vel,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

@configclass
class SpiderEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpiderSceneCfg = SpiderSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()  # ← 新增這行

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0