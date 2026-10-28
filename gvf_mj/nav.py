"""导航层：把 gvf_tracking/src/mgvfdc 的论文算法接到四足策略上。

三个关键设计（都是"具身化"必须做的适配，不是论文内容）
---------------------------------------------------------------------------
1. **速度标定**：策略的 "指令 → 实际速度" 并不等于 1:1，所以导航层算出的 (u, r) 是
   【期望的实际速度】，要经标定表反解成策略指令。
   · v1（原版权重 iteration_3000）：0.4 指令只跑出 0.28，低速达成率 30%
   · v2（微调权重 iteration_4500，2026-10-28）：0.4 指令跑出 0.36，低速达成率 82%
2. **指令包络**：v2 策略训练范围 vx∈[-1.2,1.2]、wz∈[-1.0,1.0]，
   所以指令限幅用这个范围；导航层的【实际速度】仍限 u≤1.0、|r|≤0.5（按你要求未提 u_max）。
3. **GPIO 吃"指令值"**：控制器内部把【下发的指令】当作已知输入，
   于是"策略实际跑出来的速度 − 指令"就变成观测器要估的扰动 d_w。
   这正是论文方法的用武之地 —— 腿足执行层的跟踪误差被当成未知扰动补偿掉。
"""
from __future__ import annotations

import sys

import numpy as np

GVF_SRC = r"D:\workpalce\gvf_tracking\src"
if GVF_SRC not in sys.path:
    sys.path.insert(0, GVF_SRC)

from mgvfdc.config import GPIOConfig, GuidanceConfig, GVFConfig   # noqa: E402
from mgvfdc.controllers import MGVFDCController                   # noqa: E402
from mgvfdc.plant import wrap_angle                               # noqa: E402

# --- 速度标定表：sweep_policy.py 实测（微调权重 iteration_4500，2026-10-28，原生 MuJoCo，6 s 平均）---
#   cmd  0.0  0.2    0.4    0.6    0.8    1.0    1.2
#   act  0.0  0.163  0.361  0.533  0.742  0.942  1.130
# 旧表（v1，iteration_3000）留在 outputs_archive/20261024_v1_policy_iter3000/ 的说明里备查
CALIB_CMD = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
CALIB_ACT = np.array([0.0, 0.163, 0.361, 0.533, 0.742, 0.942, 1.130])
VX_CMD_LIMIT = 1.2      # 新策略训练包络 ±1.2
WZ_CMD_LIMIT = 1.0      # 新策略训练包络 ±1.0

# --- 防卡死参数（v2 新增，v2.1 收紧）---
# v2.1 教训：v2 的判据太松，被外力推得短暂失速就会被误判成"卡死"，
# 于是 u 在 +1.0/−0.5 之间反复跳 → 形成极限环、把机器人带离目标（有扰动实验里暴露）。
# 现在要求：① 必须贴近障碍（才是真"顶住了"）② 要求更久的停滞 ③ 触发后冷却 2 s。
JAM_V_THRESH = 0.06     # [m/s] 实测速度低于此值视为"没动"
JAM_TIME = 1.2          # [s] 持续这么久触发自救
JAM_NEAR_CLEAR = 1.8    # [m] 只有离障碍边界这么近时才允许触发
ESCAPE_TIME = 0.6       # [s] 自救倒车时长
ESCAPE_U = -0.50        # [m/s] 自救倒车速度
ESCAPE_COOLDOWN = 2.0   # [s] 触发后的冷却时间，避免连发


def vx_actual_to_cmd(u_actual: float) -> float:
    """期望的实际速度 → 策略 vx 指令（线性内插反解，带符号）。"""
    sign = -1.0 if u_actual < 0 else 1.0
    a = min(abs(u_actual), 1.10)
    return float(sign * np.interp(a, CALIB_ACT, CALIB_CMD))


class NavLayer:
    """MG VFDC 导航层 + 策略指令映射。"""

    def __init__(self,
                 obstacle_xy: np.ndarray | None = None,
                 obstacle_rho: np.ndarray | None = None,
                 *,
                 compensation: bool = True,
                 u_max: float = 1.0,        # 期望实际速度上限（按用户要求暂不提，策略实际能到 1.13）
                 u_min: float = -0.6,       # 允许小幅倒车：解决 u≥0 + den<0 的静摩擦死锁
                 r_max: float = 0.5,        # 导航层期望偏航角速度上限（实际速度口径）
                 wz_scale: float = 1.20,    # 偏航标定：v2 实测 cmd 0.4→0.32, 0.6→0.51，约 1.2 倍
                 omega_yaw: float = 3.0,    # 观测器带宽（导航 50 Hz ⇒ ω·dt=0.06，离散稳定）
                 omega_s: float = 2.5,
                 k1: float = 1.5, k2: float = 0.6,
                 k3: float = 1.0, k4: float = 0.5,
                 eps_tanh: float = 0.01, lam: float = 1.0, psi_T: float = 0.45,
                 cmd_tau: float = 0.15,     # 指令一阶低通时间常数 [s]（实测标定，见 REPORT）
                 cmd_rate_max: float = 6.0):  # 指令变化率上限 [1/s]（相对量程）
        # 注：τ=0.25/rate=3 过于迟钝，会让"倒车脱困"机动慢半拍而跑飞（clean 单次 6.1 m）；
        #     rate=8 又会因指令突变导致摔倒。τ=0.15/rate=6 是实测折中点。
        self.obstacle_xy = (np.zeros((0, 2)) if obstacle_xy is None
                            else np.asarray(obstacle_xy, dtype=float).reshape(-1, 2))
        self.obstacle_rho = (np.zeros(0) if obstacle_rho is None
                             else np.asarray(obstacle_rho, dtype=float).ravel())
        self.wz_scale = wz_scale
        self.u_max = float(u_max)
        self.u_min = float(u_min)
        self.psi_T = float(psi_T)
        self._jam_t = 0.0        # 卡死计时
        self._escape_t = 0.0     # 自救倒车剩余时间
        self._escape_cool = 0.0  # 自救冷却计时
        # 【v2.1 关键修正】连续航向：GPIO 的 err = x̂ − x_meas 不做角度归一化，
        # 机器人绕目标转圈时航向跨 ±π 会让测量瞬间跳 2π，观测器被打爆
        # （实测 d̂_w3 从 0 冲到 +15，随后控制律乱转、丢目标）。
        # 这里在导航层内部累积"不折返"的航向再喂给控制器（cos/sin、e_ψ 都是周期量，安全）。
        self._psi_cont: float | None = None
        self._psi_last: float | None = None
        # 指令低通/限速：策略是在"指令每 5.5 s 才变一次"的条件下训练的，
        # 50 Hz 抖动的指令属于分布外输入 —— 实测会直接把机器人晃摔（z_min=0.11 m）。
        self.cmd_tau = float(cmd_tau)
        self.cmd_rate_max = float(cmd_rate_max)
        self._vx_f = 0.0
        self._wz_f = 0.0

        gvf_cfg = GVFConfig(
            rho_b=1.0, rho_a=2.1, rho_s=1.1, chi=0.25,
            att_mode="variable", normalize_fields=True, blend="product",
            rep_sense_rule="side",
        )
        ctrl_cfg = GuidanceConfig(
            k1=k1, k2=k2, k3=k3, k4=k4, lam=lam,
            eps_tanh=eps_tanh, psi_T=psi_T,
            u_max=u_max, u_min=u_min, r_max=r_max,
            den_guard=0.12, compensation=compensation, psi_r_dot_max=6.0,
            yaw_obs=GPIOConfig(order=3, omega_o=omega_yaw),
            s_obs=GPIOConfig(order=3, omega_o=omega_s),
        )
        self.ctrl = MGVFDCController(gvf_cfg, ctrl_cfg)
        self.cfg = ctrl_cfg

    # ------------------------------------------------------------------
    def reset(self, xy, psi: float = 0.0):
        self.ctrl.reset(np.asarray(xy, dtype=float), psi, self.obstacle_xy)
        self._vx_f = 0.0
        self._wz_f = 0.0
        self._jam_t = 0.0
        self._escape_t = 0.0
        self._escape_cool = 0.0
        self._psi_cont = float(psi)
        self._psi_last = float(psi)

    # ------------------------------------------------------------------
    def _shape(self, vx_t: float, wz_t: float, dt: float) -> tuple[float, float]:
        """指令低通 + 变化率限制（防止 50 Hz 抖动变成策略的分布外输入）。"""
        a = float(np.clip(dt / max(self.cmd_tau, 1e-6), 0.0, 1.0))
        vx_f = self._vx_f + a * (vx_t - self._vx_f)
        wz_f = self._wz_f + a * (wz_t - self._wz_f)
        # 变化率上限（按指令量程归一）
        dvx_max = self.cmd_rate_max * VX_CMD_LIMIT * dt
        dwz_max = self.cmd_rate_max * WZ_CMD_LIMIT * dt
        vx_f = float(np.clip(vx_f, self._vx_f - dvx_max, self._vx_f + dvx_max))
        wz_f = float(np.clip(wz_f, self._wz_f - dwz_max, self._wz_f + dwz_max))
        self._vx_f, self._wz_f = vx_f, wz_f
        return vx_f, wz_f

    # ------------------------------------------------------------------
    def step(self, xy, psi: float, psi_dot: float, target_xy, dt: float,
             v_meas: float = 0.0) -> dict:
        """一步制导。输入：位姿、测量偏航角速度、目标位置、控制周期、实测前向速度。

        返回导航量 + 要下发给策略的速度指令 command=[vx_cmd, vy, wz_cmd]。

        v2 的两处工程修正（v1 策略太慢、没暴露）：
        · **转向优先改成"绝对上限"**：论文式(32)(33) 的 exp(ψ_T−|e_ψ|) 本意是"没对准就别冲"。
          v1 里它只是把 u_raw 乘一个系数，而 u_raw 被滑模项撑得很大（几十），乘完仍饱和，
          于是机器人以满速顶进障碍圆柱、卡死（实测 z_min 正常但位置冻结、e_ψ=−1.58）。
          改成 `|u| ≤ u_max·exp(ψ_T−|e_ψ|)` 后才真正"先转向、再前进"。
        · **卡死自救**：指令前进但实测速度 ≈0 持续 0.8 s → 倒车 0.6 s 脱困。
        """
        # 连续航向累计（用测量ψ的增量，避免跨 ±π 跳变）——必须在控制步之前推进
        step_dpsi = wrap_angle(float(psi) - float(self._psi_last))
        self._psi_cont += step_dpsi
        self._psi_last = float(psi)

        out = self.ctrl.step(
            np.asarray(xy, dtype=float), float(self._psi_cont), float(psi_dot),
            np.asarray(target_xy, dtype=float),
            self.obstacle_xy, self.obstacle_rho, dt,
        )
        # ---- ① 转向优先：绝对速度上限 ----
        e_abs = abs(out.e_psi)
        u_prio = self.u_max * (np.exp(self.psi_T - e_abs) if e_abs > self.psi_T else 1.0)
        u_eff = float(np.sign(out.u) * min(abs(out.u), u_prio))

        # ---- ② 卡死自救：只在"贴近障碍 + 前进指令却完全不动"时触发 ----
        if self._escape_cool > 0.0:
            self._escape_cool -= dt
        near_obs = False
        if len(self.obstacle_xy):
            xy_arr = np.asarray(xy, dtype=float)[None, :]
            near_obs = bool(np.min(np.linalg.norm(self.obstacle_xy - xy_arr, axis=1)
                                   - self.obstacle_rho) < JAM_NEAR_CLEAR)
        if (abs(u_eff) > 0.4 and abs(v_meas) < JAM_V_THRESH
                and near_obs and self._escape_cool <= 0.0):
            self._jam_t += dt
        else:
            self._jam_t = 0.0
        if self._jam_t > JAM_TIME:
            self._escape_t = ESCAPE_TIME
            self._escape_cool = ESCAPE_COOLDOWN
            self._jam_t = 0.0
        if self._escape_t > 0.0:
            self._escape_t -= dt
            u_eff = ESCAPE_U

        vx_t = float(np.clip(vx_actual_to_cmd(u_eff), -VX_CMD_LIMIT, VX_CMD_LIMIT))
        wz_t = float(np.clip(out.r * self.wz_scale, -WZ_CMD_LIMIT, WZ_CMD_LIMIT))
        vx_cmd, wz_cmd = self._shape(vx_t, wz_t, dt)
        return dict(
            u=u_eff, r=out.r, psi_r=out.psi_r, s=out.s, e_psi=out.e_psi,
            d_w3_hat=out.d_w3_hat, d_p_hat=out.d_p_hat,
            sigma_min=out.sigma_min, priority_gain=out.priority_gain,
            vx_cmd=vx_cmd, wz_cmd=wz_cmd,
            command=np.array([vx_cmd, 0.0, wz_cmd], dtype=np.float64),
        )
