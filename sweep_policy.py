"""策略速度响应标定 / 候选权重的离线评估（原生 MuJoCo，纯 CPU，不需要 GPU）。

用途
----
1) 标定"指令 → 实际速度"曲线（导航层速度标定表就是这里产出的）；
2) 微调后比较多个候选 checkpoint，挑最好的那个（**不要只取最后一个**）。

用法
----
  # 评估 code/ 里的默认权重
  D:\\anaconda\\envs\\MJX\\python.exe sweep_policy.py

  # 评估微调产出的权重（可一次给多个，做横向对比）
  ... --ckpt D:\\ft\\iteration_3500.pkl D:\\ft\\iteration_4000.pkl D:\\ft\\iteration_4500.pkl

  # 微调后的权重观测里接了真实速度 → 必须加 --no-odom-zero
  ... --ckpt ...\\iteration_4000.pkl --no-odom-zero

输出：每个权重一张表（vx 0.2~1.2 的达成率、wz 0.2~0.8 的达成率、摔倒）
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, r"D:\workpalce\gvf_mujoco")

from gvf_mj.env import POLICY_DT, XGDeploy
from gvf_mj.policy import DEFAULT_CKPT, ActorMLP

VX_CMDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
WZ_CMDS = (0.2, 0.4, 0.6, 0.8)


def measure(env: XGDeploy, net: ActorMLP, cmd, seconds: float, skip: float):
    """固定指令跑一段，返回 (平均 vx, vx 标准差, 平均 wz, 最低躯干高度)。"""
    n = int(round(seconds / POLICY_DT))
    n_skip = int(round(skip / POLICY_DT))
    cmd = np.asarray(cmd, dtype=np.float64)
    vxs, wzs, zs = [], [], []
    for i in range(n):
        action = net(env.obs(cmd))
        env.apply_action(action)
        m = env.measure()
        zs.append(m["z"])
        if i >= n_skip:
            vxs.append(m["vx"])
            wzs.append(m["wz"])
    return float(np.mean(vxs)), float(np.std(vxs)), float(np.mean(wzs)), float(np.min(zs))


def evaluate(ckpt: str, odom_zero: bool, seconds: float = 6.0):
    net = ActorMLP(ckpt)
    env = XGDeploy(obstacles=(), odom_zero=odom_zero)
    print(f"\n[{ckpt}]")
    print(f"  {net.describe()}")
    print(f"  {'指令 vx':>8}{'实测 vx':>9}{'std':>7}{'达成率':>8}{'最低z':>8}")
    rows = []
    for c in VX_CMDS:
        env.reset()
        vx, sd, _, zmin = measure(env, net, [c, 0, 0], seconds, 1.5)
        rows.append((c, vx))
        print(f"  {c:8.2f}{vx:9.3f}{sd:7.3f}{vx / c * 100:7.1f}%{zmin:8.3f}")
    print(f"  {'指令 wz':>8}{'实测 wz':>9}{'std':>7}{'达成率':>8}{'最低z':>8}")
    for c in WZ_CMDS:
        env.reset()
        _, _, wz, zmin = measure(env, net, [0, 0, c], 5.0, 1.5)
        print(f"  {c:8.2f}{wz:9.3f}{'':>7}{wz / c * 100:7.1f}%{zmin:8.3f}")

    cs = np.array([r[0] for r in rows])
    vs = np.array([r[1] for r in rows])
    print("  反解（想让实际速度达到 target，需要下多大指令）:")
    for target in (0.2, 0.3, 0.4, 0.6):
        if vs.max() > target:
            need = float(np.interp(target, vs, cs))
            print(f"    实际 {target:.1f} m/s → 指令 vx ≈ {need:.2f}"
                  f"{'（包络内 ✓）' if need <= 1.2 else '（超出包络 ✗）'}")
        else:
            print(f"    实际 {target:.1f} m/s → 达不到（策略上限 ≈ {vs.max():.2f} m/s）")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="*", default=None,
                    help="一个或多个 .pkl；缺省=code/ 里的 iteration_3000")
    ap.add_argument("--no-odom-zero", action="store_true",
                    help="obs[0:3] 喂真实速度（微调后的策略必须加）")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()

    ckpts = a.ckpt or [DEFAULT_CKPT]
    for c in ckpts:
        evaluate(c, odom_zero=not a.no_odom_zero, seconds=a.seconds)


if __name__ == "__main__":
    main()
