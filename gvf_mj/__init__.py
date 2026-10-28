"""gvf_mj —— 把 code/ 里训练好的 XG 策略放进【原生 MuJoCo】做 3D 具身仿真的封装包。

设计原则
--------
1. 完全不修改 `code/`：只读取它的 MJCF、checkpoint 与配置数值（下面用常量抄写并注明出处）。
2. 不依赖 MJX/JAX 跑仿真：用原生 `mujoco`（CPU），因此可以在无 GPU 的机器上跑，
   也可以在租来的无显示器 GPU 服务器上跑（渲染与仿真分离，见 render.py）。
3. 与训练严格对齐的 8 个数值（来自 code/src/cfg/xg_config.py 与 envs/backends/generic_env.py）：
     dt = 0.005            (x g_config.py:127，覆盖 xg.xml 里的 0.002)
     decimation = 4        (xg_config.py:79)  → 策略 50 Hz
     action_scale = 0.5    (xg_config.py:78)
     action_clip = 1.0     (xg_config.py:81)
     Kp = 20.0, Kd = 0.5   (xg_config.py:76-77)
     torque_limit = 28 N·m (robots/xg.py:55)
     模型侧 dof_damping[6:] = 0、actuator 增益 = 0（generic_env.py:61-68 的 use_custom_pd 分支）
     观测 48D（components/tasks/velocity/observations.py:24-32）
"""
