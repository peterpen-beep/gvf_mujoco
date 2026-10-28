"""原生 MuJoCo 的 XG 四足部署环境（严格对齐 code/ 里的训练配置）。

本文件是全工程的"运动层 + 物理层"，不涉及导航算法。
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

# ===========================================================================================
# 一、与训练严格对齐的常量（出处已注明；改这里前请先核对 code/src）
# ===========================================================================================
XML_PATH = r"D:\workpalce\code\assets\bot_model\xg\xg.xml"

DT_SIM = 0.005          # cfg/xg_config.py:127  config.sim.dt（覆盖 xg.xml 的 0.002）
DECIMATION = 4          # cfg/xg_config.py:79   → 策略控制周期 0.02 s = 50 Hz
KP, KD = 20.0, 0.5      # cfg/xg_config.py:76-77  control.stiffness / damping
ACTION_SCALE = 0.5      # cfg/xg_config.py:78
ACTION_CLIP = 1.0       # cfg/xg_config.py:81
TORQUE_LIMIT = 28.0     # robots/xg.py:55
NUM_JOINTS = 12
OBS_DIM = 48

# 关节顺序（robots/xg.py:10-23，与 MJCF 一致，mjx_to_policy_idx = identity）
JOINT_NAMES = (
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT",
)
# 默认站姿（robots/xg.py:28-41）
DEFAULT_POSE = np.array([0.0, 0.8, -1.5] * 4, dtype=np.float64)
# 关节限位（robots/xg.py:42-43）
JOINT_LOWER = np.array([-0.4887, -1.152, -2.723] * 4, dtype=np.float64)
JOINT_UPPER = np.array([0.4887, 2.967, -0.602] * 4, dtype=np.float64)

POLICY_DT = DT_SIM * DECIMATION


# ===========================================================================================
# 二、四元数工具（与 brax.math.rotate 同一约定：rotate(v, q) = R(q)·v）
# ===========================================================================================
def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵。"""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """把向量 v 用四元数 q 旋转（与 brax.math.rotate 一致）。"""
    return quat_to_mat(q) @ np.asarray(v, dtype=np.float64)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


# ===========================================================================================
# 三、建模：xg.xml + 运行时添加障碍物/目标（不改 xg.xml）
# ===========================================================================================
def build_model(obstacles=(), with_target: bool = True, timestep: float = DT_SIM):
    """构建场景。

    obstacles: 序列，每项 (x, y, radius, height)，生成立在地面上的圆柱。
    with_target: 是否加一个 mocap 小球表示目标（纯视觉，不参与碰撞）。
    """
    spec = mujoco.MjSpec.from_file(XML_PATH)
    spec.option.timestep = timestep

    for i, ob in enumerate(obstacles):
        x, y, r, h = float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3])
        body = spec.worldbody.add_body(name=f"obstacle_{i}", pos=[x, y, h / 2.0])
        geom = body.add_geom(name=f"obstacle_geom_{i}")
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [r, h / 2.0, 0.0]
        geom.rgba = [0.58, 0.59, 0.64, 1.0]

    if with_target:
        tb = spec.worldbody.add_body(name="target_marker", pos=[10.0, 10.0, 0.45], mocap=True)
        tg = tb.add_geom(name="target_geom")
        tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
        tg.size = [0.20, 0.0, 0.0]
        tg.rgba = [1.0, 0.76, 0.10, 1.0]
        tg.contype = 0
        tg.conaffinity = 0

    model = spec.compile()

    # —— 与训练 generic_env.py:61-68 的 use_custom_pd 分支一致 ——
    model.dof_damping[6:] = 0.0                    # 关节阻尼清零（改用 PD 力矩里的 Kd）
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 0.0         # actuator 无效化，力矩走 qfrc_applied
        model.actuator_biasprm[i, :] = 0.0
    return model


# ===========================================================================================
# 四、部署环境
# ===========================================================================================
class XGDeploy:
    """XG 四足的环境封装：reset / step(action) / obs(command) / measure()。"""

    def __init__(self, obstacles=(), with_target: bool = True, settle_time: float = 0.2,
                 odom_zero: bool = True):
        self.obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 4)
        self.model = build_model(self.obstacles, with_target)
        self.data = mujoco.MjData(self.model)

        # ⚠️ 关键部署细节：训练（ppo.py:248）与 play.py:373 都把 `odom_data` 传成 zeros(3)，
        # 而 observations.py:80 是 `obs_lin_vel = info["odom_data"]` →
        # 也就是说 **obs[0:3] 在训练时恒为 0**（"里程计占位"没接）。部署必须一致，
        # 否则喂真实速度会变成分布外输入（实测会让速度跟踪从 ~0.4 掉到 0.29）。
        self.odom_zero = bool(odom_zero)

        self.obstacle_xy = self.obstacles[:, :2].copy()
        self.obstacle_rho = self.obstacles[:, 2].copy()

        self._mocap_ids = [i for i in range(self.model.nmocap)]
        self._target_body = None
        for bid in range(self.model.nbody):
            if self.model.body(bid).name == "target_marker":
                self._target_body = bid
                break
        # 外力扰动用：躯干 body id
        self._torso_body = next((b for b in range(self.model.nbody)
                                 if self.model.body(b).name == "torso"), None)

        self.t = 0.0
        self.last_action = np.zeros(NUM_JOINTS)
        self.reset()

        if settle_time > 0:
            self._settle(settle_time)

    # ------------------------------------------------------------------ 状态
    def reset(self, pos=(0.0, 0.0, 0.40), yaw: float = 0.0):
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = [pos[0], pos[1], pos[2]]
        d.qpos[3:7] = quat_from_yaw(yaw)
        d.qpos[7:19] = DEFAULT_POSE
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, d)
        self.t = 0.0
        self.last_action = np.zeros(NUM_JOINTS)

    # ------------------------------------------------------------------ 外力扰动
    def set_push(self, force_xy=(0.0, 0.0), torque_z: float = 0.0):
        """在世界系躯干上施加恒定外力/力矩（用于复现论文的"外力扰动"实验）。

        注意：xfrc_applied 会被 mj_step 持续使用，不会自动清零，用完要 clear_push()。
        """
        if self._torso_body is None:
            return
        self.data.xfrc_applied[self._torso_body, :] = [
            float(force_xy[0]), float(force_xy[1]), 0.0, 0.0, 0.0, float(torque_z)]

    def clear_push(self):
        self.data.xfrc_applied[:] = 0.0

    def _settle(self, seconds: float):
        """起步前用默认站姿 PD 稳住（让脚落地），不属于算法部分。"""
        n = max(1, int(round(seconds / DT_SIM)))
        q_des = DEFAULT_POSE.copy()
        for _ in range(n):
            self._substep(q_des)

    # ------------------------------------------------------------------ 物理
    def _substep(self, q_des: np.ndarray):
        """一个物理子步：算 PD 力矩 → qfrc_applied → mj_step（与训练 compute_pd_torques 等价）。"""
        d, m = self.data, self.model
        q = d.qpos[7:19]
        qd = d.qvel[6:18]
        tau = KP * (q_des - q) - KD * qd
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[6:18] = tau
        mujoco.mj_step(m, d)
        self.t += m.opt.timestep

    def apply_action(self, action: np.ndarray):
        """执行一个策略步（decimation 个物理子步）。"""
        a = np.clip(np.asarray(action, dtype=np.float64), -ACTION_CLIP, ACTION_CLIP)
        q_des = DEFAULT_POSE + ACTION_SCALE * a
        q_des = np.clip(q_des, JOINT_LOWER, JOINT_UPPER)
        for _ in range(DECIMATION):
            self._substep(q_des)
        self.last_action = a

    # ------------------------------------------------------------------ 观测
    def obs(self, command: np.ndarray) -> np.ndarray:
        """48 维 actor 观测（顺序见 observations.py:24-32）。

        注意 obs[0:3]：按训练/play 的口径喂 **zeros**（见 __init__ 里的说明）。
        """
        d = self.data
        quat = d.qpos[3:7].copy()
        inv = quat_inv(quat)
        if self.odom_zero:
            lin_b = np.zeros(3)
        else:
            lin_b = rotate(d.qvel[0:3], inv)
        ang_b = rotate(d.qvel[3:6], inv)
        grav_b = rotate(np.array([0.0, 0.0, -1.0]), inv)
        q = d.qpos[7:19].copy()
        qd = d.qvel[6:18].copy()
        return np.concatenate([
            lin_b, ang_b, grav_b,
            q - DEFAULT_POSE, qd,
            self.last_action,
            np.asarray(command, dtype=np.float64),
        ])

    def measure(self) -> dict:
        """导航层需要的量：平面位姿 + 机体系速度 + 偏航角速度。"""
        d = self.data
        quat = d.qpos[3:7].copy()
        inv = quat_inv(quat)
        v_b = rotate(d.qvel[0:3], inv)
        w_b = rotate(d.qvel[3:6], inv)
        return dict(
            x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
            yaw=yaw_from_quat(quat), quat=quat,
            vx=float(v_b[0]), vy=float(v_b[1]),
            wz=float(w_b[2]),
            roll=float(np.arctan2(quat_to_mat(quat)[2, 1], quat_to_mat(quat)[2, 2])),
            pitch=float(-np.arcsin(np.clip(quat_to_mat(quat)[2, 0], -1.0, 1.0))),
        )

    # ------------------------------------------------------------------ 目标显示
    def set_target_marker(self, xy):
        if self._target_body is None:
            return
        mid = self.model.body_mocapid[self._target_body]
        if mid >= 0:
            self.data.mocap_pos[mid] = [float(xy[0]), float(xy[1]), 0.45]
            self.data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------ 诊断
    def min_clearance(self) -> float:
        """机器人中心到最近障碍物【边界】的距离（与 2D 仿真的 clear 定义一致）。"""
        if len(self.obstacle_xy) == 0:
            return float("inf")
        p = self.data.qpos[0:2]
        dists = np.linalg.norm(self.obstacle_xy - p[None, :], axis=1) - self.obstacle_rho
        return float(np.min(dists))
