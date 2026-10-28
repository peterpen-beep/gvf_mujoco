# 【已归档】v1 版本产物说明（策略 = iteration_3000 原版权重）

> 归档时间：2026-10-28
> 归档原因：**微调策略后（新权重 iteration_4500）重新生成了全部图/视频/数据**，
> 为避免新旧混淆，旧产物整体移入本目录保留备查。
> 新产物在 `gvf_mujoco/outputs/`。

---

## 1. 本目录产物对应的版本

| 项 | v1（本目录） | v2（当前 `outputs/`） |
|---|---|---|
| 策略权重 | `code/checkpoints/xg_velocity/checkpoints_20260912_151104/iteration_3000.pkl`（原版） | `code_finetune` 微调后的 `iteration_4500.pkl` |
| 部署观测 | `obs[0:3]` **恒为 0**（原训练口径，`odom_zero=True`） | `obs[0:3]` = 真实机体速度（`--no-odom-zero`） |
| 导航层速度标定 | 按老模型：cmd 0.4→实际 0.28 m/s | 按新模型：cmd 0.44→实际 0.40 m/s |
| 指令低通 | τ=0.15 s / 变化率 6.0 | 同左（本次未改） |
| 生成时间 | 2026-10-24 | 2026-10-28 |

## 2. 各文件是什么

| 文件 | 内容 |
|---|---|
| `run3d_mgvfdc.npz` | **确定性单次**闭环（MGVFDC 含补偿），出图/视频用的数据。末端距目标 0.08 m，最小安全裕度 1.12 m |
| `run3d_mgvf.npz` | 同上但 **MGVF（关掉 GPIO 补偿）**；⚠️ 也是带噪 seed-0（末端 5.93 m 那次已被覆盖） |
| `run3d_mgvfdc_push.npz` / `run3d_mgvf_push.npz` | 外力扰动实验（t=32 s 施 60 N 侧向脉冲 0.25 s），60 s 时长 |
| `run3d_noobs.npz` | 无障碍参考（旧的 40 s clean 版） |
| `fig_traj_3d.png` | 俯视轨迹对比图（**clean 运行**：MGVFDC 末端 0.08 m / MGVF 5.48 m） |
| `fig_signals_3d.png` | 关键信号（距离 / 期望 vs 实测速度 / d̂_w3 / d̂_p），clean 运行 |
| `snapshot_3d.png` | 3D 渲染抽帧（clean 运行） |
| `gvf_mujoco_3d.mp4` | 3D 演示视频（**clean 运行**，40 s，1280×720，25 fps） |
| `metrics.json` / `metrics.csv` | ⚠️ 已被后续运行覆盖，不代表 v1 的 5-seed 统计 |

> ⚠️ **数据一致性更正（2026-10-28）**：本目录的 `gvf_mujoco_3d.mp4` 与 `fig_*.png` 是用 **clean
> （无噪声、确定性）**那次运行渲染的（末端 0.08 m）；但 `run3d_*.npz` 在渲染之后又被 `--seeds 5`
> 那次运行覆盖了，现在存的是**带噪声的 seed-0**（末端 5.19 m）。即：**视频/图 = clean 运行，
> npz = 带噪运行**，两者不是同一次。要恢复 clean 数据需按 §4 重跑。

## 3. 为什么被取代（v1 的已知问题）

1. **策略速度跟踪差**：直走 0.2 m/s 达成率仅 30%、0.4 m/s 为 70%；原地转 0.4 rad/s 仅 54%、0.8 rad/s 仅 40%。
   → 导致闭环"速度跟踪误差"高达 **0.33 m/s**，稳态跟踪 RMSE 2.8 m（5 seeds）。
2. **闭环 A/B 结论不稳健**：MGVFDC vs MGVF 的差异随导航指令滤波参数翻转
   （τ=0.25/rate=3 时补偿恢复快 7×，τ=0.15/rate=6 时反过来）。
   唯一稳健结论是"**MGVFDC 的安全裕度更一致**"（1.03±0.03 vs 0.78±0.39）。
3. 原因是"扰动"主体是**执行器饱和**（策略最多跑 ~0.8 m/s 且低速欠跟踪），不是论文假设的匹配扰动。

## 4. 如何重新生成这些旧产物（复现命令）

需要旧权重 `iteration_3000.pkl`（`D:\workpalce\code\checkpoints\xg_velocity\checkpoints_20260912_151104\`）：

```powershell
cd D:\workpalce\gvf_mujoco
$py = "D:\anaconda\envs\MJX\python.exe"
& $py -m gvf_mj.run_sim --clean --compare                       # 单次（出图用）
& $py -m gvf_mj.run_sim --seeds 5 --compare                     # 多种子指标
& $py -m gvf_mj.render --npz outputs/run3d_mgvfdc.npz           # 图 + 视频
```
> 注意：当前 `nav.py` 的标定表已按新模型（iteration_4500）更新，
> 要严格复现 v1 需把 `CALIB_ACT` 改回 `[0, 0.061, 0.280, 0.491, 0.676, 0.811]` 且不传 `--no-odom-zero`。
