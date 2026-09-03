# 快速修復版 - 移除需要自定義 mdp 函數的部分
import torch
import math
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

# ✅ 使用官方的 velocity_mdp（包含所有需要的函數）
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from . import mdp

class TripodAction(JointPositionAction):
    """修正版三腳步態控制器"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.dt = env.step_dt
        self.freq = 2.0
        
        self.sin_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.cos_phase = torch.ones(self.num_envs, 1, device=self.device)
        
    def process_actions(self, actions: torch.Tensor):
        self.phase += 2 * math.pi * self.freq * self.dt
        self.phase = torch.fmod(self.phase, 2 * math.pi)

        self.sin_phase[:, 0] = torch.sin(self.phase)
        self.cos_phase[:, 0] = torch.cos(self.phase)

        prior = torch.zeros_like(actions)
        phase_A = self.phase
        phase_B = self.phase + math.pi

        swing_amp = 0.3
        knee_amp  = 0.6
        ankle_amp = 0.4

        # A 組 signals（左後 leg0、左前 leg2、右中 leg5）
        val_swing_A  =  swing_amp * torch.cos(phase_A)
        val_knee_A   =  knee_amp  * torch.clamp(torch.sin(phase_A), min=0.0)
        val_ankle_A  = -ankle_amp * torch.clamp(torch.sin(phase_A), min=0.0)  # ← 修正符號

        prior[:, 0]  = -val_swing_A   # leg0 hip  (左 → 取負)
        prior[:, 1]  =  val_knee_A    # leg0 femur
        prior[:, 2]  =  val_ankle_A   # leg0 tibia

        prior[:, 6]  = -val_swing_A   # leg2 hip  (左 → 取負)
        prior[:, 7]  =  val_knee_A    # leg2 femur
        prior[:, 8]  =  val_ankle_A   # leg2 tibia

        prior[:, 15] =  val_swing_A   # leg5 hip  (右 → 取正)
        prior[:, 16] =  val_knee_A    # leg5 femur
        prior[:, 17] =  val_ankle_A   # leg5 tibia

        # B 組 signals（左中 leg1、右前 leg3、右後 leg4）
        val_swing_B  =  swing_amp * torch.cos(phase_B)
        val_knee_B   =  knee_amp  * torch.clamp(torch.sin(phase_B), min=0.0)
        val_ankle_B  = -ankle_amp * torch.clamp(torch.sin(phase_B), min=0.0)  # ← 修正符號

        prior[:, 3]  = -val_swing_B   # leg1 hip  (左 → 取負)
        prior[:, 4]  =  val_knee_B    # leg1 femur
        prior[:, 5]  =  val_ankle_B   # leg1 tibia

        prior[:, 9]  =  val_swing_B   # leg3 hip  (右 → 取正)
        prior[:, 10] =  val_knee_B    # leg3 femur
        prior[:, 11] =  val_ankle_B   # leg3 tibia

        prior[:, 12] =  val_swing_B   # leg4 hip  (右 → 取正)
        prior[:, 13] =  val_knee_B    # leg4 femur
        prior[:, 14] =  val_ankle_B   # leg4 tibia

        combined_actions = (actions * 0.1) + prior
        super().process_actions(combined_actions)

    def reset(self, env_ids: torch.Tensor | None = None):
        super().reset(env_ids)
        if env_ids is not None:
            self.phase[env_ids] = 0.0

@configclass
class TripodActionCfg(JointPositionActionCfg):
    class_type: type = TripodAction

def gait_phase(env, action_name: str = "joint_action") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return torch.cat([action_term.sin_phase, action_term.cos_phase], dim=-1)

# 加在 gait_phase 函數旁邊
def lin_vel_y_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """懲罰側向速度"""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


@configclass
class SpiderSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
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
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(1.0, 0.0, 0.0, 0.0),
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
        scale=0.3,
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

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        phase = ObsTerm(func=gait_phase, params={"action_name": "joint_action"})

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class RewardsCfg:
    """✅ 使用官方 velocity_mdp 函數"""
    
    # 核心：追蹤速度
    track_lin_vel_xy = RewTerm(
        func=velocity_mdp.track_lin_vel_xy_exp,  # ✅ 使用 velocity_mdp
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # 姿態穩定
    flat_orientation_l2 = RewTerm(
        func=velocity_mdp.flat_orientation_l2,  # ✅ 使用 velocity_mdp
        weight=-1.0
    )
    
    ang_vel_xy_l2 = RewTerm(
        func=velocity_mdp.ang_vel_xy_l2,  # ✅ 使用 velocity_mdp
        weight=-0.2 # 從 -0.05 改成 -0.2
    )
    
    base_height_l2 = RewTerm(
        func=velocity_mdp.base_height_l2,  # ✅ 使用 velocity_mdp
        weight=-0.5,
        params={"target_height": 0.12, "asset_cfg": SceneEntityCfg("robot")}
    )
    
    # 碰撞懲罰
    undesired_contacts = RewTerm(
        func=velocity_mdp.undesired_contacts,  # ✅ 使用 velocity_mdp
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), 
            "threshold": 1.0
        }
    )
    
    # 動作平滑
    action_rate_l2 = RewTerm(
        func=velocity_mdp.action_rate_l2,  # ✅ 使用 velocity_mdp
        weight=-0.01
    )
    
    joint_vel_l2 = RewTerm(
        func=velocity_mdp.joint_vel_l2,  # ✅ 使用 velocity_mdp
        weight=-5e-5
    )
    
    joint_acc_l2 = RewTerm(
        func=velocity_mdp.joint_acc_l2,  # ✅ 使用 velocity_mdp
        weight=-2.5e-7
    )
    
    lin_vel_z_l2 = RewTerm(
        func=velocity_mdp.lin_vel_z_l2,  # ✅ 使用 velocity_mdp
        weight=-0.5
    )
    
    joint_torques_l2 = RewTerm(
        func=velocity_mdp.joint_torques_l2,  # ✅ 使用 velocity_mdp
        weight=-1e-5
    )
    
    # ✅ 使用官方的 feet_air_time
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time,  # ✅ 使用 velocity_mdp
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_.*_3"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        }
    )

    # # ✅ 加這兩個，直走問題會大幅改善
    # track_ang_vel_z = RewTerm(
    #     func=velocity_mdp.track_ang_vel_z_exp,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": 0.25}
    # )

    # lin_vel_y_l2 = RewTerm(
    #     func=lin_vel_y_l2,
    #     weight=-2.0
    # )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})

@configclass
class SpiderEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpiderSceneCfg = SpiderSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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