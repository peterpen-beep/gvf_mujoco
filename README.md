# gvf_mujoco —— 把论文的 GVF 抗扰制导算法接到强化学习四足策略上（3D 闭环）

> 论文：S. Kong, J. Sun, A. Luo, et al., *"A Collision-Free Target Tracking Controller With
> Uncertain Disturbance Rejection for Quadruped Robots"*, IEEE Trans. Intell. Veh., 2024.
> DOI: 10.1109/TIV.2023.3296669

本仓库把**论文的制导算法**（改进 GVF + GPIO 扰动观测器 + 滑模趋近）作为**上层导航**，
把**训练好的 12 自由度四足步态策略**作为**下层执行**，在**原生 MuJoCo**里做 3D 闭环：
机器狗绕开静态障碍、追上并跟随运动目标；支持外力扰动实验（侧风 + 冲击）。

**全程 CPU 运行（45~90× 实时），不需要 GPU。**

---

## 1. 分层结构

```
导航层 @50 Hz（论文方法，纯 numpy）
  MGVF 场 → ψ_r → 转向律(式22) → 趋近律(式27) → (u, r)
  双 GPIO 观测器（式23/28）估计 d_w3、d_p
        ↓  [vx_cmd, 0, wz_cmd]（4 步具身适配，见 §4）
运动层 @50 Hz（RL 策略，numpy 前向）
  48 维观测 → MLP(512-256-128) → 12 维动作
  → q_des = 默认站姿 + 0.5×动作 → PD 力矩 τ = Kp(q_des−q) − Kd·q̇
物理层 @200 Hz（MuJoCo）
  dt=0.005 × 4 子步；力矩经 qfrc_applied 施加；真实接触/摩擦求解
```

## 2. 运行前需要准备的两样外部资产

本仓库**不包含**以下文件（体积/版权考虑），运行时需要放到指定位置：

| 资产 | 用途 | 代码中的路径常量 |
|---|---|---|
| `xg.xml` + `meshes/`（XG 四足的 MJCF 模型） | MuJoCo 场景 | `gvf_mj/env.py` 的 `XML_PATH` |
| `iteration_*.pkl`（训练好的步态策略权重，约 4.5 MB） | 策略推理 | `gvf_mj/policy.py` 的 `DEFAULT_CKPT` |
| `gvf_tracking/src/`（论文算法的 2D 复现代码） | GVF + MGPIO 实现 | `gvf_mj/nav.py` 的 `GVF_SRC` |

> 三个路径目前是作者本机的绝对路径；在你机器上运行时改成自己的路径即可。

## 3. 快速开始

```powershell
# 依赖：python 3.10 + mujoco + numpy + matplotlib + imageio（视频）
pip install mujoco numpy matplotlib imageio imageio-ffmpeg

# ① 部署自检（策略能否站立/直走/转向）
python check_deploy.py

# ② 策略速度标定（生成"指令→实际速度"表，导航层要用）
python sweep_policy.py --ckpt <你的 iteration_4500.pkl> --no-odom-zero

# ③ 3D 闭环（无扰动）
python -m gvf_mj.run_sim --ckpt <权重> --no-odom-zero --clean --compare

# ④ 出图 + 视频
python -m gvf_mj.render --npz outputs/run3d_mgvfdc.npz

# ⑤ 有扰动实验（8 N 侧风 + t=15/32 s 各 50 N 冲击）
python -m gvf_mj.run_disturbed --ckpt <权重> --clean --compare --wind-y 8 --push-times "15,32" --push-force 50 --duration 60
python -m gvf_mj.render --npz outputs/run3d_mgvfdc_dist.npz --alt outputs/run3d_mgvf_dist.npz --tag _disturbed --push-times "15,32"
```

## 4. 送进策略之前的 4 步具身适配（论文之外，必需）

| 步骤 | 内容 | 为什么 |
|---|---|---|
| ① 转向优先**绝对上限** | `|u| ≤ u_max·exp(ψ_T−|e_ψ|)` | 论文是"乘法"，滑模项饱和时失效 → 机器人满速顶死障碍 |
| ② 卡死自救 | 贴障碍且 1.2 s 不动 → 倒车 0.6 s | 兜底脱困 |
| ③ 速度/偏航**标定** | 查表把"期望实际速度"反解成指令；偏航 ×1.2 | RL 策略系统性跑不到标称速度（指令 0.4 实际 0.36） |
| ④ 一阶低通 + 变化率限制 | τ=0.15 s | 50 Hz 指令抖动会让策略失稳（实测会摔） |

## 5. 结果摘要（当前策略 iteration_4500，同场景）

| 指标 | 无扰动(clean) | 有扰动(clean) | 无扰动(5 seeds) | 有扰动(5 seeds) |
|---|---|---|---|---|
| 末端距目标 | 0.13 m | 0.20 m | 2.71 m | 0.93 m |
| 跟踪 RMSE | 0.43 m | 0.19 m | 1.60 m | 1.38 m |
| 最小安全裕度（判据 >1.0 m） | 1.50 m | 1.54 m | 1.32 m | 1.38 m |
| 是否摔倒 | 否 | 否 | 否 | 否 |

> 扰动设置：8 N 世界系 +y 恒定风力 + t=15/32 s 各 50 N、0.25 s 冲击。
> 策略速度跟踪：vx 达成率 82~96%（旧策略 30~70%）。

## 6. 与论文不一致的地方（必须声明）

1. **式(32)(33) 转向优先**：由乘法改为绝对上限（理由见 §4）。
2. **允许小幅倒车** `u_min=−0.6`（论文默认 u≥0；为回避"仅前进 + 分母<0"的死锁）。
3. **观测器已知输入**用指令值而非实测速度 → 策略的执行误差成为被估的 `d_w`。
4. 论文的 **Case 2（动态障碍）**、**真机实验**、**GVFPID 基线对比**未做。

## 7. 文件结构

```
gvf_mj/
  env.py           原生 MuJoCo 部署环境（场景/PD/48 维观测/外力）
  policy.py        .pkl → numpy MLP
  nav.py           论文算法 + 速度标定 + 包络限幅 + 指令低通 + 防卡死
  run_sim.py       3D 闭环主程序（多种子/扰动/指标/存 npz）
  run_disturbed.py 有扰动闭环（常值风 + 定时冲击）
  render.py        离线渲染（出图/视频，与仿真解耦）
check_deploy.py    部署自检（含 numpy vs flax 交叉验证）
sweep_policy.py    策略速度标定 / 多权重评估
ref_play_check.py  只读调用上游 MJX play.py 做对照
REPORT.md          阶段报告
WORK_LOG.md        工作日志（版本演进、改动、FAQ、踩坑记录）
```

## 8. 已知限制

- 观测是**真值**（无编码器噪声、无 IMU 偏置、无观测/控制延迟）；
- 未建模电机非线性、背隙、质量/惯量偏差、地面不平；
- 统计样本 n=5，部分指标方差较大（REPORT 里已标注哪些结论稳健/不稳健）；
- 策略速度上限约 1.13 m/s，目标速度 0.4 m/s 时 5-seed 跟踪 RMSE 仍在 1.4~2.2 m。

详见 [`REPORT.md`](REPORT.md) 与 [`WORK_LOG.md`](WORK_LOG.md)。
