"""有扰动版 3D 闭环实验：常值侧风 + 定时冲击推，检验新策略 + GPIO 补偿的鲁棒性。

与 run_sim.py 的区别只有"外力"：
  · wind   : 作用在躯干上的**常值**外力（世界系，[N]）——模拟持续侧风/坡面
  · pushes : 定时**冲击**外力（世界系方向 dir_deg、力 force N、持续 dur s）
导航层、策略、场景、指标口径全部与 run_sim 一致，便于和无扰动版对比。

用法：
  $py -m gvf_mj.run_disturbed --ckpt D:\\workpalce\\ft\\ft3\\iteration_4500.pkl --no-odom-zero --clean
  $py -m gvf_mj.run_disturbed --ckpt ... --no-odom-zero --seeds 5 --compare
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .env import POLICY_DT, XGDeploy
from .nav import NavLayer
from .policy import ActorMLP
from .run_sim import OBSTACLES_3D, OUT_DIR, TGT_CENTER, TGT_RADIUS, target_position, NOISE_POS, NOISE_YAW, NOISE_WZ


# ===========================================================================================
def run_once(seed: int = 0, duration: float = 60.0, compensation: bool = True,
             wind=(0.0, 12.0), pushes=None, noise: bool = False, jitter: bool = False,
             ckpt: str | None = None, odom_zero: bool = False) -> dict:
    """跑一次带外力的闭环，返回记录。"""
    pushes = pushes or []
    rng = np.random.RandomState(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi)) if jitter else 0.0
    jx, jy = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)) if jitter else (0.0, 0.0)
    jyaw = float(rng.uniform(-0.06, 0.06)) if jitter else 0.0

    env = XGDeploy(obstacles=OBSTACLES_3D, settle_time=0.2, odom_zero=odom_zero)
    net = ActorMLP(ckpt) if ckpt else ActorMLP()
    ob_xy = np.array([[o[0], o[1]] for o in OBSTACLES_3D])
    ob_rho = np.array([o[2] for o in OBSTACLES_3D])
    nav = NavLayer(ob_xy, ob_rho, compensation=compensation)

    env.reset(pos=(jx, jy, 0.40), yaw=jyaw)
    m = env.measure()
    nav.reset((m["x"], m["y"]), m["yaw"])

    n = int(round(duration / POLICY_DT))
    keys = ("t", "x", "y", "z", "yaw", "vx", "vy", "wz", "u", "r",
            "vx_cmd", "wz_cmd", "psi_r", "e_psi", "s", "d_w3_hat", "d_p_hat",
            "sigma_min", "dist", "clear", "tgt_x", "tgt_y", "vx_ach", "wz_ach",
            "f_ext", "pushing")
    rec = {k: np.zeros(n) for k in keys}
    qpos_hist = np.zeros((n, env.model.nq))

    for k in range(n):
        t = k * POLICY_DT
        tgt = target_position(t, phase)
        env.set_target_marker(tgt)

        # ---- 外力：常值风 + 冲击 ----
        fx, fy, tz = float(wind[0]), float(wind[1]), 0.0
        pushing = False
        for p in pushes:
            if p["t"] <= t < p["t"] + p["dur"]:
                ang = np.deg2rad(p.get("dir_deg", 90.0))
                fx += p["force"] * np.cos(ang)
                fy += p["force"] * np.sin(ang)
                tz += p.get("torque_z", 0.0)
                pushing = True
        if abs(fx) > 1e-9 or abs(fy) > 1e-9 or abs(tz) > 1e-9:
            env.set_push((fx, fy), tz)
        else:
            env.clear_push()

        # ---- 感知（可带噪声）----
        m = env.measure()
        meas_xy = np.array([m["x"], m["y"]])
        meas_yaw, meas_wz = m["yaw"], m["wz"]
        if noise:
            meas_xy = meas_xy + rng.normal(0.0, NOISE_POS, 2)
            meas_yaw = meas_yaw + rng.normal(0.0, NOISE_YAW)
            meas_wz = meas_wz + rng.normal(0.0, NOISE_WZ)

        nav_out = nav.step(meas_xy, meas_yaw, meas_wz, tgt, POLICY_DT, v_meas=m["vx"])
        action = net(env.obs(nav_out["command"]))
        qpos_hist[k] = env.data.qpos
        env.apply_action(action)

        vals = dict(t=t, x=m["x"], y=m["y"], z=m["z"], yaw=m["yaw"],
                    vx=m["vx"], vy=m["vy"], wz=m["wz"],
                    u=nav_out["u"], r=nav_out["r"],
                    vx_cmd=nav_out["vx_cmd"], wz_cmd=nav_out["wz_cmd"],
                    psi_r=nav_out["psi_r"], e_psi=nav_out["e_psi"], s=nav_out["s"],
                    d_w3_hat=nav_out["d_w3_hat"], d_p_hat=nav_out["d_p_hat"],
                    sigma_min=nav_out["sigma_min"], clear=env.min_clearance(),
                    tgt_x=tgt[0], tgt_y=tgt[1],
                    f_ext=float(np.hypot(fx, fy)), pushing=float(pushing))
        for kk, vv in vals.items():
            rec[kk][k] = vv
        rec["dist"][k] = float(np.hypot(m["x"] - tgt[0], m["y"] - tgt[1]))
        m2 = env.measure()
        rec["vx_ach"][k] = m2["vx"]
        rec["wz_ach"][k] = m2["wz"]

    env.clear_push()
    rec["qpos"] = qpos_hist
    return rec


# ===========================================================================================
def summarize(rec: dict, pushes) -> dict:
    t, dist = rec["t"], rec["dist"]
    tail = t >= t[-1] - 15.0
    half = slice(len(t) // 2, None)
    ok = np.where(dist < 0.6)[0]
    arrival = float(t[ok[0]]) if len(ok) else float("nan")
    track = float("nan")
    if np.isfinite(arrival) and t[-1] - arrival > 4.0:
        track = float(np.sqrt(np.mean(dist[t >= arrival + 2.0] ** 2)))

    out = dict(final=float(dist[-1]), ss_rmse=float(np.sqrt(np.mean(dist[tail] ** 2))),
               track_rmse=track, arrival=arrival, min_clear=float(rec["clear"].min()),
               path=float(np.sum(np.hypot(np.diff(rec["x"]), np.diff(rec["y"])))),
               vx_err=float(np.mean(np.abs(rec["vx_ach"][half] - rec["u"][half]))),
               z_min=float(rec["z"].min()), fell=bool(rec["z"].min() < 0.20),
               d_p_mean=float(np.mean(rec["d_p_hat"][half])),
               yaw_max=float(np.max(np.abs(rec["yaw"] - rec["yaw"][0]))))

    # 每次冲击后的偏离与恢复
    devs, recs, swing = [], [], []
    for p in pushes:
        t0 = p["t"]
        win = (t >= t0) & (t < t0 + 5.0)
        if not win.any():
            continue
        devs.append(float(dist[win].max()))
        i0 = int(np.argmin(np.abs(t - t0)))
        swing.append(float(np.max(np.abs(rec["yaw"][win] - rec["yaw"][i0]))))
        back = np.where((t > t0 + 0.05) & (dist < 0.8))[0]
        recs.append(float(t[back[0]] - t0) if len(back) else float("nan"))
    out["push_dev_max"] = float(np.nanmax(devs)) if devs else float("nan")
    out["push_recover"] = float(np.nanmean(recs)) if recs else float("nan")
    out["push_yaw_swing"] = float(np.nanmax(swing)) if swing else float("nan")
    return out


KEY_ROWS = [("final", "末端距[m]"), ("ss_rmse", "末15sRMSE[m]"), ("track_rmse", "跟踪RMSE[m]"),
            ("arrival", "到达[s]"), ("min_clear", "最小裕度[m]"), ("vx_err", "速度误差[m/s]"),
            ("push_dev_max", "冲击最大偏离[m]"), ("push_recover", "恢复[s]"),
            ("push_yaw_swing", "航向摆幅[rad]"), ("z_min", "最低z[m]")]


def print_table(results: dict):
    head = f"{'控制器':<10}{'seed':>5}" + "".join(f"{h:>16}" for _, h in KEY_ROWS)
    print(head)
    print("-" * len(head))
    for name, ms in results.items():
        for i, m in enumerate(ms):
            print(f"{name:<10}{i:>5}" + "".join(
                f"{(f'{m[k]:.2f}' if np.isfinite(m[k]) else '  —  '):>16}" for k, _ in KEY_ROWS))
        for stat, fn in (("均值", np.nanmean), ("标准差", np.nanstd)):
            print(f"{name+' '+stat:<10}{'':>5}" + "".join(
                f"{fn(np.array([m[k] for m in ms], float)):>16.2f}" for k, _ in KEY_ROWS))
        print("-" * len(head))


# ===========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--no-odom-zero", action="store_true", default=True,
                    help="默认开启：新策略观测含真实速度")
    ap.add_argument("--odom-zero", action="store_true", help="老策略用（观测喂 0）")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--compare", action="store_true", help="同时跑 MGVF（关补偿）")
    ap.add_argument("--clean", action="store_true", default=True,
                    help="默认无噪声（出图/视频用）；--noise 打开噪声")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--wind-x", type=float, default=0.0)
    ap.add_argument("--wind-y", type=float, default=12.0, help="侧风 [N]（+y 方向）")
    ap.add_argument("--push-times", type=str, default="15,32", help="冲击时刻，逗号分隔")
    ap.add_argument("--push-force", type=float, default=60.0, help="[N]")
    ap.add_argument("--push-dur", type=float, default=0.25, help="[s]")
    ap.add_argument("--push-dir", type=float, default=90.0, help="世界系方向 deg")
    ap.add_argument("--out", type=str, default=None, help="单 seed 时保存 npz 的路径")
    a = ap.parse_args()

    times = [float(s) for s in a.push_times.split(",") if s.strip()]
    pushes = [dict(t=tt, force=a.push_force, dur=a.push_dur, dir_deg=a.push_dir) for tt in times]
    odom_zero = bool(a.odom_zero)
    noise = bool(a.noise)

    print("=" * 110)
    print("有扰动 3D 闭环：常值侧风 + 冲击推")
    print(f"  风 = ({a.wind_x:+.0f}, {a.wind_y:+.0f}) N | 冲击 {a.push_force:.0f} N × {a.push_dur}s "
          f"@ t={times} (方向 {a.push_dir:.0f}°) | 时长 {a.duration:.0f}s | seeds={a.seeds}")
    print(f"  噪声 {'开' if noise else '关(clean)'} | 策略 {a.ckpt or '默认(code/ iteration_3000)'} | "
          f"odom_zero={odom_zero}")
    print("=" * 110)

    combos = [("MGVFDC", True)] + ([("MGVF", False)] if a.compare else [])
    results, keep = {}, {}
    for name, comp in combos:
        ms = []
        for s in range(a.seeds):
            rec = run_once(seed=s, duration=a.duration, compensation=comp, wind=(a.wind_x, a.wind_y),
                           pushes=pushes, noise=noise, jitter=noise, ckpt=a.ckpt, odom_zero=odom_zero)
            m = summarize(rec, pushes)
            ms.append(m)
            if s == 0:
                keep[name] = rec
            print(f"  {name} seed={s}: 末端 {m['final']:5.2f} | 跟踪RMSE {m['track_rmse']:5.2f} | "
                  f"冲击偏离 {m['push_dev_max']:5.2f} | 恢复 {m['push_recover']:5.2f}s | "
                  f"裕度 {m['min_clear']:4.2f} | 速度误差 {m['vx_err']:4.2f}"
                  + (" | ⚠️摔倒" if m["fell"] else ""))
        results[name] = ms
    print()
    print_table(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rec in keep.items():
        tag = "mgvfdc" if name == "MGVFDC" else "mgvf"
        path = a.out if (a.out and len(keep) == 1) else os.path.join(OUT_DIR, f"run3d_{tag}_dist.npz")
        np.savez_compressed(path, **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)})
        print(f"  已保存 {path}")
    with open(os.path.join(OUT_DIR, "metrics_disturbed.json"), "w", encoding="utf-8") as f:
        json.dump({n: ms for n, ms in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"  指标：{os.path.join(OUT_DIR, 'metrics_disturbed.json')}")


if __name__ == "__main__":
    main()
