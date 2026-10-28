"""离线渲染：从 .npz 轨迹数据重建 3D 场景，出图 / 出视频。

设计要点（解决"GPU 服务器没有显示器"）
--------------------------------------
仿真（run_sim.py）与渲染彻底解耦：服务器只需要跑仿真存 .npz，渲染可以回到任何有 GL 的机器上做。
本脚本只依赖 .npz 里的 qpos 序列 + 场景参数，不重跑仿真。

用法：
  D:\\anaconda\\envs\\MJX\\python.exe -m gvf_mj.render --npz outputs/run3d_mgvfdc.npz
  ... --no-video   只出图不出视频
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .run_sim import OBSTACLES_3D, TGT_CENTER, TGT_RADIUS

OUT_DIR = os.path.join(r"D:\workpalce\gvf_mujoco", "outputs")
RHO_B = 1.0


# ===========================================================================================
# 一、2D 理想执行对照（用 3D 实际下发的同一串指令，喂给理想的单轨质点）
# ===========================================================================================
def replay_2d_from_commands(rec: dict) -> dict:
    """把 3D 运行中导航层输出的 (u, r) 序列，喂给"完美执行"的单轨质点模型。

    这样 3D 与 2D 的差别**只来自执行层**（腿足 vs 理想质点），是干净的"具身差距"对比；
    如果两边分别重跑导航环，轨迹分岔会混入控制器对初值的敏感性，无法归因。
    """
    n = len(rec["t"])
    dt = float(rec["t"][1] - rec["t"][0])
    out = {k: np.zeros(n) for k in ("t", "x", "y", "yaw", "dist")}
    x, y, psi = float(rec["x"][0]), float(rec["y"][0]), float(rec["yaw"][0])
    for k in range(n):
        out["t"][k] = rec["t"][k]
        out["x"][k], out["y"][k], out["yaw"][k] = x, y, psi
        out["dist"][k] = float(np.hypot(x - rec["tgt_x"][k], y - rec["tgt_y"][k]))
        u, r = float(rec["u"][k]), float(rec["r"][k])
        x += u * np.cos(psi) * dt
        y += u * np.sin(psi) * dt
        psi += r * dt
    return out


# ===========================================================================================
# 二、俯视轨迹图（3D 实体 vs 2D 理想执行）
# ===========================================================================================
def fig_trajectory(rec: dict, rec_alt: dict | None, out_path: str, title: str):
    """主图：MGVFDC（含补偿）vs MGVF（无补偿）在同一 3D plant 下的轨迹对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.25,
    })
    fig, ax = plt.subplots(figsize=(8.8, 7.8))

    for ob in OBSTACLES_3D:
        ax.add_patch(Circle((ob[0], ob[1]), ob[2], fc="#c9c9c9", ec="#555555", lw=1.2, zorder=3))
        ax.add_patch(Circle((ob[0], ob[1]), ob[2] + RHO_B, fc="none", ec="#d62728",
                            ls="--", lw=1.1, alpha=0.85, zorder=3))

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(TGT_CENTER[0] + TGT_RADIUS * np.sin(th), TGT_CENTER[1] + TGT_RADIUS * np.cos(th),
            "--", color="#FFB300", lw=1.6, label="目标轨迹")

    if rec_alt is not None:
        ax.plot(rec_alt["x"], rec_alt["y"], "-", color="#d62728", lw=1.8, alpha=0.9,
                label="MGVF（关掉 GPIO 补偿）")
    ax.plot(rec["x"], rec["y"], "-", color="#1f77b4", lw=2.3,
            label="MGVFDC（含 GPIO 补偿）")
    ax.plot(rec["x"][0], rec["y"][0], "s", ms=8, color="#0d47a1", mec="k", zorder=9,
            label="起点")
    ax.plot(rec["tgt_x"][-1], rec["tgt_y"][-1], "*", ms=19, color="#FFB300",
            mec="#7a5200", zorder=9, label="目标（末时刻）")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    n = len(rec["t"])
    tail = slice(int(0.7 * n), None)
    info = (f"MGVFDC：末端 {rec['dist'][-1]:.2f} m，RMSE "
            f"{float(np.sqrt(np.mean(rec['dist'][tail]**2))):.2f} m，"
            f"最小裕度 {rec['clear'].min():.2f} m\n"
            f"躯干最低 z {rec['z'].min():.2f} m")
    if rec_alt is not None:
        info += (f"\nMGVF ：末端 {rec_alt['dist'][-1]:.2f} m，RMSE "
                 f"{float(np.sqrt(np.mean(rec_alt['dist'][tail]**2))):.2f} m，"
                 f"最小裕度 {rec_alt['clear'].min():.2f} m")
    ax.text(0.015, 0.985, info, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.9, pad=3.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_signals(rec: dict, out_path: str, push_times=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.25,
    })
    t = rec["t"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2))
    axes[0, 0].plot(t, rec["dist"], color="#1f77b4")
    axes[0, 0].set_ylabel("到目标的距离 [m]")
    axes[0, 0].set_title("跟踪误差")
    # 冲击时刻标注（有扰动实验）
    for i, tp in enumerate(push_times):
        for ax in (axes[0, 0], axes[0, 1]):
            ax.axvline(tp, color="#d62728", ls="--", lw=1.0, alpha=0.7,
                       label="外力冲击" if i == 0 else None)
        if "f_ext" in rec:
            j = int(np.argmin(np.abs(t - tp)))
            axes[0, 0].annotate("冲击", (tp, rec["dist"][j]), xytext=(4, 6),
                                textcoords="offset points", fontsize=8, color="#d62728")

    axes[0, 1].plot(t, rec["u"], color="#7f7f7f", lw=1.2, label="导航层期望 u")
    axes[0, 1].plot(t, rec["vx_ach"], color="#1f77b4", lw=1.2, alpha=0.85, label="实测 vx")
    axes[0, 1].set_ylabel("速度 [m/s]")
    axes[0, 1].set_title("速度：期望 vs 实测（差值就是腿足跟踪误差）")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, rec["d_w3_hat"], color="#d62728", lw=1.2)
    axes[1, 0].set_ylabel(r"$\hat d_{w3}$ [rad/s]")
    axes[1, 0].set_xlabel("t [s]")
    axes[1, 0].set_title("偏航扰动估计（GPIO）")

    axes[1, 1].plot(t, rec["d_p_hat"], color="#2ca02c", lw=1.2)
    axes[1, 1].set_ylabel(r"$\hat d_{p}$ [m/s]")
    axes[1, 1].set_xlabel("t [s]")
    axes[1, 1].set_title("趋近通道扰动估计（GPIO）")
    fig.suptitle("3D 具身闭环的关键信号", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================================
# 三、3D 视频（MuJoCo 离屏渲染）
# ===========================================================================================
def make_video(rec: dict, out_path: str, stride: int = 2,
               width: int = 1280, height: int = 720) -> str | None:
    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 无法 import mujoco：{e}")
        return None

    from .env import build_model
    from .run_sim import POLICY_DT

    model = build_model(OBSTACLES_3D, with_target=True)
    data = mujoco.MjData(model)
    tgt_bid = next((b for b in range(model.nbody) if model.body(b).name == "target_marker"), None)

    qpos = rec["qpos"]
    n = len(qpos)
    cam = mujoco.MjvCamera()
    cam.distance = 6.0
    cam.azimuth = 135.0
    cam.elevation = -30.0

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 离屏渲染器创建失败（无 GL？）：{e}")
        return None

    frames = []
    for k in range(0, n, stride):
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(model, data)
        if tgt_bid is not None:
            mid = model.body_mocapid[tgt_bid]
            if mid >= 0:
                data.mocap_pos[mid] = [rec["tgt_x"][k], rec["tgt_y"][k], 0.45]
        # 跟随相机：主要盯机器狗，略微偏向目标（两者尽量同框）
        rx, ry = float(data.qpos[0]), float(data.qpos[1])
        cam.lookat = np.array([0.85 * rx + 0.15 * rec["tgt_x"][k],
                               0.85 * ry + 0.15 * rec["tgt_y"][k], 0.30])
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()

    import imageio.v2 as imageio
    fps = max(1, int(round(1.0 / (POLICY_DT * stride))))
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"  视频：{out_path}（{len(frames)} 帧，{width}×{height}，{fps} fps）")
    return out_path


# ===========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=os.path.join(OUT_DIR, "run3d_mgvfdc.npz"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-alt", action="store_true", help="不画 MGVF 消融对照")
    ap.add_argument("--alt", type=str, default=None, help="MGVF 消融的 npz 路径")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tag", type=str, default="", help="产物文件名后缀，如 _disturbed（不覆盖无扰动版）")
    ap.add_argument("--title-suffix", type=str, default="", help="标题里的场景说明，如 （有扰动：侧风+冲击）")
    ap.add_argument("--push-times", type=str, default="", help="冲击时刻，逗号分隔（用于在图上标注）")
    a = ap.parse_args()
    push_times = [float(s) for s in a.push_times.split(",") if s.strip()]

    d = np.load(a.npz)
    rec = {k: d[k] for k in d.files}
    print(f"载入 {a.npz}：{len(rec['t'])} 帧，{float(rec['t'][-1]) + 0.02:.1f} s")

    rec_alt = None
    alt_path = a.alt or os.path.join(OUT_DIR, "run3d_mgvf.npz")
    if not a.no_alt and os.path.exists(alt_path):
        da = np.load(alt_path)
        rec_alt = {k: da[k] for k in da.files}
        print(f"载入对照 {alt_path}（MGVF 无补偿）")

    p1 = fig_trajectory(rec, rec_alt, os.path.join(OUT_DIR, f"fig_traj_3d{a.tag}.png"),
                        f"3D 具身闭环{a.title_suffix}：MGVFDC（含补偿） vs MGVF（无补偿）")
    print(f"  图1：{p1}")
    p2 = fig_signals(rec, os.path.join(OUT_DIR, f"fig_signals_3d{a.tag}.png"), push_times=push_times)
    print(f"  图2：{p2}")

    if not a.no_video:
        make_video(rec, os.path.join(OUT_DIR, f"gvf_mujoco_3d{a.tag}.mp4"), stride=a.stride)


if __name__ == "__main__":
    main()
