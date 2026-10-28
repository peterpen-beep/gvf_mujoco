"""3D 闭环主程序：论文 MGVFDC 导航层 + XG 四足策略（原生 MuJoCo，只用 CPU）。

场景与 2D 复现完全一致：
    · 起点 (0,0)，航向 0
    · 3 个圆柱障碍：(3.0,1.6,r0.6) / (5.6,6.9,r0.8) / (8.2,2.6,r0.6)
    · 目标在 (10,10) 半径 2 m 的圆上以 0.4 m/s（ω=0.2 rad/s）运动
    · ρ_b = 1.0 m，ρ_a = 2.1，ρ_s = 1.1，χ = 0.25，∏σ 融合

实验模式
--------
1) 单次出图：   -m gvf_mj.run_sim --clean --compare
2) 多种子统计： -m gvf_mj.run_sim --seeds 5 --compare
3) 外力扰动：   -m gvf_mj.run_sim --seeds 3 --compare --push
（--push 会在 t=20 s 给躯干施加 60 N、0.25 s 的侧向脉冲）

指标口径（比第一版更严谨）
--------------------------
· 到达时间 arrival_t ：首次 dist < 0.6 m 的时刻
· 稳态 RMSE ss_rmse  ：最后 15 s 的 RMSE（避开追赶段）
· 速度跟踪误差 vx_err ：|实测 vx − 导航层期望 u| 的后半段均值（量化"具身差距"）
· 外力扰动：push_max_dist（扰动后 5 s 内最大距离）、push_recover_t（恢复到 <0.8 m 用时）
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .env import POLICY_DT, XGDeploy
from .nav import NavLayer
from .policy import ActorMLP

# ---- 场景（与 2D 完全一致）----
OBSTACLES_3D = [                      # (x, y, 半径, 高度)
    (3.0, 1.6, 0.6, 1.0),
    (5.6, 6.9, 0.8, 1.2),
    (8.2, 2.6, 0.6, 1.0),
]
TGT_CENTER = np.array([10.0, 10.0])
TGT_RADIUS = 2.0
TGT_OMEGA = 0.2                       # → 切向速度 0.4 m/s
OUT_DIR = os.path.join(r"D:\workpalce\gvf_mujoco", "outputs")

# 测量噪声（模拟真实感知；论文 SimConfig 也有类似设置）
NOISE_POS = 0.01          # [m]
NOISE_YAW = 0.005         # [rad]
NOISE_WZ = 0.02           # [rad/s]


def target_position(t: float, phase: float = 0.0, omega: float | None = None) -> np.ndarray:
    a = (TGT_OMEGA if omega is None else float(omega)) * t + phase
    return TGT_CENTER + TGT_RADIUS * np.array([np.sin(a), np.cos(a)])


# ===========================================================================================
def run(seed: int = 0, duration: float = 40.0, compensation: bool = True,
        obstacles: bool = True, noise: bool = True, jitter: bool = True,
        push: dict | None = None, nav_kwargs: dict | None = None,
        ckpt: str | None = None, odom_zero: bool = True) -> dict:
    """跑一次闭环，返回记录数组。

    ckpt      : 策略权重路径（None=用 code/ 里的 iteration_3000）
    odom_zero : 观测 obs[0:3] 是否喂 0。
                · 原策略（code/ 里的）训练时该位恒为 0 → True
                · 微调后的策略（code_finetune/）接了真实机体速度 → False
    """
    rng = np.random.RandomState(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi)) if jitter else 0.0
    jx, jy = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)) if jitter else (0.0, 0.0)
    jyaw = float(rng.uniform(-0.06, 0.06)) if jitter else 0.0

    scene = OBSTACLES_3D if obstacles else []
    env = XGDeploy(obstacles=scene, settle_time=0.2, odom_zero=odom_zero)
    net = ActorMLP(ckpt) if ckpt else ActorMLP()

    ob_xy = np.array([[o[0], o[1]] for o in scene], dtype=float).reshape(-1, 2)
    ob_rho = np.array([o[2] for o in scene], dtype=float)
    nav = NavLayer(ob_xy, ob_rho, compensation=compensation, **(nav_kwargs or {}))

    env.reset(pos=(jx, jy, 0.40), yaw=jyaw)
    m = env.measure()
    nav.reset((m["x"], m["y"]), m["yaw"])

    n = int(round(duration / POLICY_DT))
    keys = ("t", "x", "y", "z", "yaw", "vx", "vy", "wz",
            "u", "r", "vx_cmd", "wz_cmd", "psi_r", "e_psi", "s",
            "d_w3_hat", "d_p_hat", "sigma_min", "dist", "clear",
            "tgt_x", "tgt_y", "vx_ach", "wz_ach", "push")
    rec = {k: np.zeros(n) for k in keys}
    actions = np.zeros((n, 12))
    qpos_hist = np.zeros((n, env.model.nq))

    for k in range(n):
        t = k * POLICY_DT
        tgt = target_position(t, phase)
        env.set_target_marker(tgt)

        # ---- 外力扰动窗口 ----
        pushing = False
        if push is not None and push["t"] <= t < push["t"] + push["dur"]:
            ang = np.deg2rad(push.get("dir_deg", 90.0))
            env.set_push((push["force"] * np.cos(ang), push["force"] * np.sin(ang)),
                         push.get("torque_z", 0.0))
            pushing = True
        elif push is not None:
            env.clear_push()

        # ---- 感知（带噪声）----
        m = env.measure()
        meas_xy = np.array([m["x"], m["y"]])
        meas_yaw = m["yaw"]
        meas_wz = m["wz"]
        if noise:
            meas_xy = meas_xy + rng.normal(0.0, NOISE_POS, 2)
            meas_yaw = meas_yaw + rng.normal(0.0, NOISE_YAW)
            meas_wz = meas_wz + rng.normal(0.0, NOISE_WZ)

        nav_out = nav.step(meas_xy, meas_yaw, meas_wz, tgt, POLICY_DT, v_meas=m["vx"])
        cmd = nav_out["command"]
        action = net(env.obs(cmd))
        actions[k] = action
        qpos_hist[k] = env.data.qpos
        env.apply_action(action)

        clear = env.min_clearance()
        vals = dict(t=t, x=m["x"], y=m["y"], z=m["z"], yaw=m["yaw"],
                    vx=m["vx"], vy=m["vy"], wz=m["wz"],
                    u=nav_out["u"], r=nav_out["r"],
                    vx_cmd=nav_out["vx_cmd"], wz_cmd=nav_out["wz_cmd"],
                    psi_r=nav_out["psi_r"], e_psi=nav_out["e_psi"], s=nav_out["s"],
                    d_w3_hat=nav_out["d_w3_hat"], d_p_hat=nav_out["d_p_hat"],
                    sigma_min=nav_out["sigma_min"], clear=clear,
                    tgt_x=tgt[0], tgt_y=tgt[1], push=float(pushing))
        for kk, vv in vals.items():
            rec[kk][k] = vv
        rec["dist"][k] = float(np.hypot(m["x"] - tgt[0], m["y"] - tgt[1]))
        m2 = env.measure()
        rec["vx_ach"][k] = m2["vx"]
        rec["wz_ach"][k] = m2["wz"]

    env.clear_push()
    rec["actions"] = actions
    rec["qpos"] = qpos_hist
    return rec


# ===========================================================================================
def metrics(rec: dict, push: dict | None = None) -> dict:
    """一套量化指标。"""
    t, dist = rec["t"], rec["dist"]
    n = len(t)
    tail = t >= t[-1] - 15.0
    dt = float(t[1] - t[0])
    half = slice(n // 2, None)

    ok = np.where(dist < 0.6)[0]
    arrival_t = float(t[ok[0]]) if len(ok) else float("nan")

    # 到达后的"纯跟踪段"指标（追赶段不该算进跟踪精度里）
    track_rmse = track_mae = float("nan")
    if np.isfinite(arrival_t) and t[-1] - arrival_t > 4.0:
        win = t >= arrival_t + 2.0
        track_rmse = float(np.sqrt(np.mean(dist[win] ** 2)))
        track_mae = float(np.mean(dist[win]))

    out = dict(
        final_dist=float(dist[-1]),
        ss_rmse=float(np.sqrt(np.mean(dist[tail] ** 2))),
        track_rmse=track_rmse,
        track_mae=track_mae,
        ss_mae=float(np.mean(dist[tail])),
        arrival_t=arrival_t,
        min_clear=float(rec["clear"].min()),
        path_len=float(np.sum(np.hypot(np.diff(rec["x"]), np.diff(rec["y"])))),
        mean_abs_u=float(np.mean(np.abs(rec["u"]))),
        vx_err=float(np.mean(np.abs(rec["vx_ach"][half] - rec["u"][half]))),
        wz_err=float(np.mean(np.abs(rec["wz_ach"][half] - rec["r"][half]))),
        z_min=float(rec["z"].min()),
        fell=bool(rec["z"].min() < 0.20),
        d_p_mean=float(np.mean(rec["d_p_hat"][half])),
    )
    # 外力扰动指标
    if push is not None:
        t0 = push["t"]
        win = (t >= t0) & (t < t0 + 5.0)
        out["push_max_dist"] = float(dist[win].max()) if win.any() else float("nan")
        rec_ok = np.where((t > t0 + 0.05) & (dist < 0.8))[0]
        out["push_recover_t"] = (float(t[rec_ok[0]] - t0) if len(rec_ok) else float("nan"))
        out["push_yaw_swing"] = float(np.max(np.abs(rec["yaw"][win] - rec["yaw"][np.argmin(np.abs(t - t0))]))) \
            if win.any() else float("nan")
    return out


KEY_ROWS = [("final_dist", "末端距目标[m]"), ("ss_rmse", "末15s RMSE[m]"),
            ("arrival_t", "到达用时[s]"), ("min_clear", "最小裕度[m]"),
            ("path_len", "路径长[m]"), ("vx_err", "速度跟踪误差[m/s]"),
            ("z_min", "躯干最低z[m]")]


def print_table(results: dict, push: dict | None = None):
    """results: {控制器名: [每 seed 的 metrics]}"""
    print("\n" + "=" * 108)
    rows = KEY_ROWS + ([("push_max_dist", "扰动后最大距[m]"), ("push_recover_t", "恢复用时[s]"),
                        ("push_yaw_swing", "航向摆幅[rad]")] if push else [])
    head = f"{'控制器':<10}{'seed':>5}" + "".join(f"{h:>18}" for _, h in rows)
    print(head)
    print("-" * 108)
    for name, ms in results.items():
        for i, m in enumerate(ms):
            line = f"{name:<10}{i:>5}"
            for key, _ in rows:
                v = m[key]
                line += f"{(f'{v:.2f}' if np.isfinite(v) else '   —  '):>18}"
            print(line)
        line = f"{name+' 均值':<10}{'':>5}"
        for key, _ in rows:
            vals = np.array([m[key] for m in ms], dtype=float)
            finite = vals[np.isfinite(vals)]
            line += f"{(f'{finite.mean():.2f}' if len(finite) else '—'):>18}"
        print(line)
        line = f"{name+' 标准差':<10}{'':>5}"
        for key, _ in rows:
            vals = np.array([m[key] for m in ms], dtype=float)
            finite = vals[np.isfinite(vals)]
            line += f"{(f'{finite.std():.2f}' if len(finite) > 1 else '—'):>18}"
        print(line)
        print("-" * 108)


# ===========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--compare", action="store_true", help="同时跑 MGVF（关补偿）作对照")
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--clean", action="store_true", help="无噪声/无初始抖动/目标相位 0（出图用）")
    ap.add_argument("--push", action="store_true", help="加外力脉冲扰动")
    ap.add_argument("--push-time", type=float, default=20.0)
    ap.add_argument("--push-force", type=float, default=60.0, help="[N]")
    ap.add_argument("--push-dur", type=float, default=0.25, help="[s]")
    ap.add_argument("--push-dir", type=float, default=90.0, help="世界系方向 deg（90=侧向 +y）")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="策略权重路径（默认用 code/ 里的 iteration_3000）")
    ap.add_argument("--no-odom-zero", action="store_true",
                    help="obs[0:3] 喂真实机体速度（微调后的策略需要；原策略不要加）")
    a = ap.parse_args()

    push = (dict(t=a.push_time, force=a.push_force, dur=a.push_dur, dir_deg=a.push_dir)
            if a.push else None)

    print("=" * 108)
    print("3D 闭环：MGVFDC 导航 + XG 四足策略（原生 MuJoCo，CPU）")
    print(f"  时长 {a.duration:.0f} s | seeds={a.seeds} | "
          f"障碍 {'有' if not a.no_obstacles else '无'} | "
          f"噪声 {'关(clean)' if a.clean else '开'} | "
          f"扰动 {f'{a.push_force}N/{a.push_dur}s @ t={a.push_time}s' if push else '无'}")
    print("=" * 108)

    combos = [("MGVFDC", True)] + ([("MGVF", False)] if a.compare else [])
    results: dict[str, list[dict]] = {}
    keep: dict[str, dict] = {}

    for name, comp in combos:
        ms = []
        for s in range(a.seeds):
            rec = run(seed=s, duration=a.duration, compensation=comp,
                      obstacles=not a.no_obstacles,
                      noise=not a.clean, jitter=not a.clean, push=push,
                      ckpt=a.ckpt, odom_zero=not a.no_odom_zero)
            m = metrics(rec, push)
            ms.append(m)
            if s == 0:
                keep[name] = rec
            print(f"  {name} seed={s}: 末端 {m['final_dist']:.2f} m | "
                  f"末15s RMSE {m['ss_rmse']:.2f} | 到达 {m['arrival_t']:.1f}s | "
                  f"裕度 {m['min_clear']:.2f} | 速度跟踪误差 {m['vx_err']:.2f}"
                  + (f" | 扰动后最大 {m['push_max_dist']:.2f} m" if push else ""))
        results[name] = ms

    print_table(results, push)

    # 保存：出图用的 npz + 指标 json/csv
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rec in keep.items():
        tag = "mgvfdc" if name == "MGVFDC" else "mgvf"
        if a.push:
            tag += "_push"
        np.savez_compressed(os.path.join(OUT_DIR, f"run3d_{tag}.npz"),
                            **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)})
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({n: ms for n, ms in results.items()}, f, ensure_ascii=False, indent=2)

    rows = ["controller,seed," + ",".join(k for k, _ in KEY_ROWS)
            + (",push_max_dist,push_recover_t,push_yaw_swing" if push else "")]
    for name, ms in results.items():
        for i, m in enumerate(ms):
            rows.append(f"{name},{i}," + ",".join(f"{m[k]:.4f}" for k, _ in KEY_ROWS)
                        + (f",{m['push_max_dist']:.4f},{m['push_recover_t']:.4f},"
                           f"{m['push_yaw_swing']:.4f}" if push else ""))
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", encoding="utf-8-sig") as f:
        f.write("\n".join(rows) + "\n")

    print(f"\n已保存：outputs/run3d_*.npz、outputs/metrics.json、outputs/metrics.csv")


if __name__ == "__main__":
    main()
