"""只读对照：直接调用 code/ 自己的 play.py（MJX 环境），量出策略在【训练环境】里的真实速度跟踪。

目的：判断"我的原生 MuJoCo 部署"和"训练环境"是否一致。
做法：把 viewer/渲染打桩成空操作（不改 code/ 任何文件），然后走他们原来的 play 流程。

用法：D:\\anaconda\\envs\\MJX\\python.exe ref_play_check.py --vx 0.4 --steps 600
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\workpalce\code\src")

import play                                        # noqa: E402  他们的脚本
from envs.backends.generic_env import GenericEnv   # noqa: E402

CKPT = (r"D:\workpalce\code\checkpoints\xg_velocity"
        r"\checkpoints_20260912_151104\iteration_3000.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()

    # —— 打桩：不弹 viewer、不渲染（其余流程完全走他们的代码）——
    GenericEnv.enable_viewer = lambda self, *x, **k: None      # type: ignore[assignment]
    GenericEnv.render = lambda self, *x, **k: None             # type: ignore[assignment]
    GenericEnv.render_frame = lambda self, *x, **k: None       # type: ignore[assignment]

    ns = argparse.Namespace(
        exp="xg_velocity", checkpoint=CKPT, random_policy=False, device="cpu",
        num_envs=1, episode_length=a.steps, seed=0, deterministic=True,
        cmd_vel_x=a.vx, cmd_vel_y=0.0, cmd_vel_yaw=a.wz,
        num_render_envs=1, render_every=2, width=320, height=240, fps=None,
        save_video=False, video_path=None,
    )
    print(f">>> 用 code/ 的 play.py 跑指令 vx={a.vx}, wz={a.wz}, {a.steps} 步（1 个环境，CPU）")
    play.play(ns)


if __name__ == "__main__":
    main()
