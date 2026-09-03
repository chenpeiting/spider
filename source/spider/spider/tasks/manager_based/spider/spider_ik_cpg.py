"""
IK-based CPG Controller for Spider 6-legged Robot
==================================================
基於使用者 MATLAB 模擬 (`SpiderBot_White_Theme.m`) 的 IK 公式，
向量化實作於 GPU 上。支援 Tripod / Wave 兩種步態。

幾何（取自 MATLAB sim）
  L1 = 0.045 m   (coxa)
  L2 = 0.090 m   (femur)
  L3 = 0.125 m   (effective tibia，URDF tibia link 內建 45° 預角)
  R0 = 0.130 m   (站立時足端到 mount 的水平距離)
  Z_HIP = -0.100 m (站立時足端在 leg-local frame 的 z)

URDF 對齊（根據 init_state 反推）
  body_leg_2 init = -0.8 rad  → leg2 站立時 hip 應收斂到 ≈ -π/4
  body_leg_3 init = +0.8 rad  → leg3 站立時 hip 應收斂到 ≈ +π/4
  其他 4 隻腳 init = 0         → hip 站立時應為 0

Convention（與 MATLAB 一致）
  leg-local +x = outward direction (where leg points when hip joint = 0)
  β[i] = leg-local +x 在 body frame 的角度
  IK 輸出 theta1/2 直接送 URDF；theta3 經 sign-flip + 45° 預角補償。
"""

import math
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


# ═══════════════════════════════════════════════════
# 機器人幾何常數
# ═══════════════════════════════════════════════════

L1 = 0.045
L2 = 0.090
# L3 = 0.100：URDF tibia link offset = (0, 0.0884, -0.0884)，
# 在 IK 直桿近似下「水平投影 ≈ 0.0884」、「斜邊 = 0.125」。
# 用斜邊 0.125 會讓 IK 算出來的 r 在 cos law 落在退化區，造成
# tripod 兩組站起來不一致（一組 r 過大被 clamp、一組 r 在合理範圍）。
# 用直桿近似 0.100 比較穩。
L3 = 0.100
TIBIA_PRE_ANGLE = math.pi / 4   # URDF tibia link 內建 -45° 向下預角

# 各腳髖關節在 body frame 的掛載點 (x, y, z)
MOUNT_POS = [
    (-0.119,  0.0585, -0.025),   # leg 0  rear-left
    ( 0.000,  0.0915, -0.025),   # leg 1  mid-left
    ( 0.119,  0.0585, -0.025),   # leg 2  front-left
    ( 0.119, -0.0585, -0.025),   # leg 3  front-right
    (-0.119, -0.0585, -0.025),   # leg 4  rear-right
    ( 0.000, -0.0915, -0.025),   # leg 5  mid-right
]

# β[i] = 該腳 leg-local +x 軸在 body frame 的角度
#   legs 0/1/4/5: URDF init hip=0 → +x 對齊「外展」方向
#   legs 2/3:     URDF init hip=∓0.8 → +x 沿 body ±Y，再由 hip 旋 ±π/4 到前角
BETA = [
     3 * math.pi / 4,   # leg 0
         math.pi / 2,   # leg 1
         math.pi / 2,   # leg 2
        -math.pi / 2,   # leg 3
    -3 * math.pi / 4,   # leg 4
        -math.pi / 2,   # leg 5
]

# 站立時足端外展方向（對稱六角，前/後角 ±45°、中側 ±90°）
OUT_ANGLE = [
     3 * math.pi / 4,   # leg 0  rear-left
         math.pi / 2,   # leg 1  mid-left
         math.pi / 4,   # leg 2  front-left
        -math.pi / 4,   # leg 3  front-right
    -3 * math.pi / 4,   # leg 4  rear-right
        -math.pi / 2,   # leg 5  mid-right
]

R0 = 0.130        # 站立時足端到 mount 的水平距離
Z_HIP = -0.100    # 站立時足端在 leg-local frame 的 z（mount 下方）

# 站立 home pose（body frame）
LEG_HOME = [
    (mx + math.cos(out) * R0, my + math.sin(out) * R0, mz + Z_HIP)
    for (mx, my, mz), out in zip(MOUNT_POS, OUT_ANGLE)
]


# ═══════════════════════════════════════════════════
# 步態相位偏移
# ═══════════════════════════════════════════════════

# Tripod：A 組 {0,2,5} 與 B 組 {1,3,4} 交替
TRIPOD_OFFSET = [0.0, math.pi, 0.0, math.pi, math.pi, 0.0]

# Wave：抬腳順序 5→3→2→1→0→4（對應作者 i→j→k→l→m→n）
WAVE_OFFSET = [
    2 * math.pi / 3,   # leg 0
        math.pi,       # leg 1
    4 * math.pi / 3,   # leg 2
    5 * math.pi / 3,   # leg 3
        math.pi / 3,   # leg 4
    0.0,               # leg 5
]


# ═══════════════════════════════════════════════════
# IK / Action 類
# ═══════════════════════════════════════════════════

class IKCPGAction(JointPositionAction):
    """貝茲足端軌跡 + MATLAB 風格 IK，全向量化。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._isaac_env = env

        d = self.device
        self.phase       = torch.zeros(self.num_envs, device=d)
        self.dt          = env.step_dt
        self.freq        = cfg.gait_freq
        self.gait        = cfg.gait
        self.step_len    = cfg.step_len
        self.step_height = cfg.step_height
        self.rl_weight   = cfg.rl_weight
        self._static_stand = cfg.static_stand
        self._femur_sign = -1.0 if cfg.flip_femur else 1.0
        self._tibia_sign = -1.0 if cfg.flip_tibia else 1.0

        # static_stand 模式下直接送出的 18 維 joint vector
        # = URDF init hip + 可調 femur/tibia（試 sign 用）
        urdf_init_hip = [0.0, 0.0, -0.8, 0.8, 0.0, 0.0]
        debug_target = []
        for h in urdf_init_hip:
            debug_target.extend([h, cfg.debug_femur, cfg.debug_tibia])
        self._debug_target = torch.tensor(debug_target, dtype=torch.float32, device=d)

        # 常數張量（搬到 device）
        mount_t = torch.tensor(MOUNT_POS, dtype=torch.float32, device=d)        # [6, 3]
        self._mx = mount_t[:, 0]
        self._my = mount_t[:, 1]
        self._mz = mount_t[:, 2]

        beta_t = torch.tensor(BETA, dtype=torch.float32, device=d)              # [6]
        self._cos_neg_beta = torch.cos(-beta_t)                                  # [6]
        self._sin_neg_beta = torch.sin(-beta_t)

        home_t = torch.tensor(LEG_HOME, dtype=torch.float32, device=d)           # [6, 3]
        self._hx = home_t[:, 0]
        self._hy = home_t[:, 1]
        self._hz = home_t[:, 2]

        offsets = TRIPOD_OFFSET if self.gait == 'tripod' else WAVE_OFFSET
        self._phase_off = torch.tensor(offsets, dtype=torch.float32, device=d)   # [6]

        # 給 gait_phase observation 用
        self.sin_phase = torch.zeros(self.num_envs, 1, device=d)
        self.cos_phase = torch.ones(self.num_envs,  1, device=d)

        self.log_buffer = {'step_count': 0, 'prior_magnitude': [], 'rl_magnitude': []}

    # ── 主邏輯 ──────────────────────────────────────
    def process_actions(self, actions: torch.Tensor):
        # static_stand：完全跳過 IK，直接送 URDF init hip + 固定 femur/tibia
        # 用這個鎖定 URDF 軸方向（試 debug_femur / debug_tibia sign）
        if self._static_stand:
            self.sin_phase[:, 0] = 0.0
            self.cos_phase[:, 0] = 1.0
            target = self._debug_target.unsqueeze(0).expand(self.num_envs, 18)
            super().process_actions(target)
            return

        # 1. 相位推進
        self.phase = torch.fmod(self.phase + 2 * math.pi * self.freq * self.dt,
                                2 * math.pi)
        self.sin_phase[:, 0] = torch.sin(self.phase)
        self.cos_phase[:, 0] = torch.cos(self.phase)

        # 2. 每隻腳的相位 [N, 6]
        lp = torch.fmod(self.phase.unsqueeze(1) + self._phase_off.unsqueeze(0),
                        2 * math.pi)
        swing = lp < math.pi                            # [N, 6]
        t_sw = lp / math.pi
        t_st = (lp - math.pi) / math.pi

        # 3. body frame 足端目標 [N, 6]
        #    swing：x 線性前進，z 半正弦抬腳；stance：x 線性後退，z 不變
        dx_sw = self.step_len * (t_sw - 0.5)
        dz_sw = self.step_height * torch.sin(math.pi * t_sw)
        dx_st = self.step_len * (0.5 - t_st)

        dx = torch.where(swing, dx_sw, dx_st)
        dz = torch.where(swing, dz_sw, torch.zeros_like(dz_sw))

        fx = self._hx.unsqueeze(0) + dx                 # [N, 6]
        fy = self._hy.unsqueeze(0).expand_as(fx)
        fz = self._hz.unsqueeze(0) + dz

        # 4. body → leg-local 轉換
        dpx = fx - self._mx.unsqueeze(0)
        dpy = fy - self._my.unsqueeze(0)
        dpz = fz - self._mz.unsqueeze(0)

        cb = self._cos_neg_beta.unsqueeze(0)
        sb = self._sin_neg_beta.unsqueeze(0)
        lx = cb * dpx - sb * dpy                        # leg-local x（外展）
        ly = sb * dpx + cb * dpy                        # leg-local y（側向）
        lz = dpz                                         # leg-local z

        # 5. MATLAB IK（6 行版本）
        theta1 = torch.atan2(ly, lx)
        w = torch.sqrt(lx * lx + ly * ly) - L1
        r = torch.clamp(torch.sqrt(w * w + lz * lz),
                        min=abs(L2 - L3) + 1e-3,
                        max=L2 + L3 - 1e-3)
        alpha = torch.atan2(lz, w)
        cos_g = torch.clamp((L2 * L2 + r * r - L3 * L3) / (2.0 * L2 * r),
                            -1.0 + 1e-6, 1.0 - 1e-6)
        cos_k = torch.clamp((L2 * L2 + L3 * L3 - r * r) / (2.0 * L2 * L3),
                            -1.0 + 1e-6, 1.0 - 1e-6)
        theta2 = alpha + torch.acos(cos_g)
        theta3_matlab = -(math.pi - torch.acos(cos_k))   # ≤ 0

        # 6. URDF 補償：tibia 軸方向相反 + 45° 預角
        theta3_urdf = -theta3_matlab - TIBIA_PRE_ANGLE

        # 7. 軸方向 sign-flip（URDF femur/tibia 軸跟 MATLAB 不一致時）
        theta2 = theta2 * self._femur_sign
        theta3_urdf = theta3_urdf * self._tibia_sign

        # 7. 排成 ActionsCfg.joint_names 順序：(leg0_θ1, θ2, θ3, leg1_θ1, ...)
        prior = torch.stack([theta1, theta2, theta3_urdf], dim=-1) \
                     .reshape(self.num_envs, 18)

        # 8. 混合 RL（rl_weight=0 → 純 CPG）
        rl_part = actions * self.rl_weight
        combined = prior + rl_part

        # 9. 監控
        self.log_buffer['step_count'] += 1
        if self.log_buffer['step_count'] % 10 == 0:
            self.log_buffer['prior_magnitude'].append(
                torch.norm(prior, dim=1).mean().item())
            self.log_buffer['rl_magnitude'].append(
                torch.norm(rl_part, dim=1).mean().item())

        super().process_actions(combined)

    def reset(self, env_ids: torch.Tensor | None = None):
        super().reset(env_ids)
        if env_ids is None:
            self.phase[:] = 0.0
        else:
            self.phase[env_ids] = 0.0

    def get_log_buffer(self):
        return self.log_buffer


@configclass
class IKCPGActionCfg(JointPositionActionCfg):
    class_type: type = IKCPGAction

    gait:         str   = 'tripod'   # 'tripod' | 'wave'
    gait_freq:    float = 1.5        # 步頻 [Hz]
    step_len:     float = 0.02       # 步幅 [m]
    step_height:  float = 0.010      # 抬腳高度 [m]（先用小值確認站起來）
    rl_weight:    float = 0.0        # 0 = 純 CPG；> 0 = Hybrid
    # debug：所有腳保持 home，不跑 gait — 用這個先確認 IK 站姿正確
    static_stand: bool  = False
    # URDF femur/tibia 軸方向跟 MATLAB IK 不一致時，set True 把 sign 翻過來
    flip_femur:   bool  = True
    flip_tibia:   bool  = True
    # static_stand 模式下用來鎖定 URDF 軸方向的 femur/tibia 固定值
    # 試 4 組合：(±0.5, ±1.0)，找到能站立的就知道 URDF 軸方向
    debug_femur:  float = -0.5
    debug_tibia:  float = -1.0


# ═══════════════════════════════════════════════════
# 站立姿勢自我驗證（standalone）
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    """快速驗算：站立時 6 隻腳的 IK 解。"""
    names = ['rear-L', 'mid-L ', 'front-L', 'front-R', 'rear-R', 'mid-R ']
    print(f"{'leg':<10}{'hip°':>10}{'femur°':>10}{'tibia°':>10}")
    for i in range(6):
        mx, my, mz = MOUNT_POS[i]
        hx, hy, hz = LEG_HOME[i]
        b = BETA[i]
        dpx, dpy, dpz = hx - mx, hy - my, hz - mz
        lx = math.cos(-b) * dpx - math.sin(-b) * dpy
        ly = math.sin(-b) * dpx + math.cos(-b) * dpy
        lz = dpz
        t1 = math.atan2(ly, lx)
        w = math.sqrt(lx*lx + ly*ly) - L1
        r = math.sqrt(w*w + lz*lz)
        alpha = math.atan2(lz, w)
        t2 = alpha + math.acos((L2*L2 + r*r - L3*L3) / (2*L2*r))
        t3m = -(math.pi - math.acos((L2*L2 + L3*L3 - r*r) / (2*L2*L3)))
        t3 = -t3m - TIBIA_PRE_ANGLE
        print(f"leg{i} {names[i]:<6}{math.degrees(t1):+10.2f}"
              f"{math.degrees(t2):+10.2f}{math.degrees(t3):+10.2f}")
