"""从 code/ 的 .pkl checkpoint 里取出 actor（MLP），用 numpy 做前向。

checkpoint 结构（已核实）：
  {'params': {'params': {'actor_mlp_0/kernel'(48,512), 'actor_mlp_0/bias'(512,),
                         'actor_mlp_1'(512,256), 'actor_mlp_2'(256,128),
                         'actor_out'(128,12), 'critic_*', 'std'(12,)},
              'opt_state', 'step': 60000, 'iteration': 3000}

网络定义见 code/src/networks/mlp.py：MLP(48→512→256→128→12)，ELU，输出无 squash。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

DEFAULT_CKPT = r"D:\workpalce\code\checkpoints\xg_velocity\checkpoints_20260912_151104\iteration_3000.pkl"


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class ActorMLP:
    """48 维观测 → 12 维动作均值（确定性推理）。"""

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT):
        self.ckpt_path = str(ckpt_path)
        with open(self.ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.iteration = ckpt.get("iteration")
        self.step = ckpt.get("step")
        p = ckpt["params"]["params"] if "params" in ckpt["params"] else ckpt["params"]

        self.layers = []
        for i in (0, 1, 2):
            self.layers.append((
                np.asarray(p[f"actor_mlp_{i}"]["kernel"], dtype=np.float64),
                np.asarray(p[f"actor_mlp_{i}"]["bias"], dtype=np.float64),
            ))
        self.out = (
            np.asarray(p["actor_out"]["kernel"], dtype=np.float64),
            np.asarray(p["actor_out"]["bias"], dtype=np.float64),
        )
        self.std = np.asarray(p["std"], dtype=np.float64) if "std" in p else None

        self.n_params = sum(w.size + b.size for w, b in self.layers) + self.out[0].size + self.out[1].size

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in self.layers:
            x = elu(x @ w + b)
        w, b = self.out
        return x @ w + b

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self(np.asarray(obs)[None, :])[0]

    def describe(self) -> str:
        dims = [self.layers[0][0].shape[0]] + [w.shape[0] for w, _ in self.layers] + \
               [self.layers[-1][0].shape[1], self.out[0].shape[1]]
        return (f"ActorMLP{'→'.join(map(str, dims))} | 参数量 {self.n_params/1e6:.2f}M | "
                f"iteration={self.iteration} step={self.step}")
