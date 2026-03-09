# task2-onlinempc

沙漠穿越问题（数学建模）求解工程，包含：

- **Level1/2**：离线最优（Oracle/标准解对齐）
- **Level3/4**：在线滚动优化（MPC, Receding Horizon）
- 单元测试 + Fuzzing + 与 `tools` 标准脚本回测对比

---

## 1. 项目目标

本项目用于求解 `Challenges.md` 中的“穿越沙漠”任务，并满足：

- Level3/4 使用在线决策（仅已知当天天气）
- 支持**可调视野**（`horizon`）
- 支持多天气情景优化（`n_scenarios`）
- 与 `tools` 中标准实现进行一致性验证

---

## 2. 目录结构

```text
task2-onlinempc/
├── src/
│   ├── env/
│   │   ├── config.py
│   │   └── utils.py
│   ├── oracle.py
│   └── mpc/
│       ├── __init__.py
│       ├── predictor.py
│       ├── core.py
│       ├── simulator.py
│       └── validator.py
├── tests/
│   ├── mpc/
│   │   ├── test_mpc_core.py
│   │   ├── test_predictor.py
│   │   └── test_online_loop.py
│   └── test_level34_degenerate_level1_level2.py
├── scripts/
│   ├── run_level3.py
│   └── run_level4.py
├── examples/
│   ├── level3_local_horizon.py
│   ├── level3_global_horizon.py
│   ├── level4_local_horizon.py
│   └── level4_global_horizon.py
├── tools/
│   ├── level1-solved.py
│   ├── level2-solved.py
│   └── validator.py
└── outputs/
    └── level34_results/
```

---

## 3. 环境与依赖

Windows 下进入项目并运行（PowerShell）：

```powershell
uv sync
uv run -V
```

---

## 4. 快速开始

### 4.1 运行 Level3（在线 MPC）

```powershell
uv run scripts\run_level3.py --horizon 4 --scenarios 3 --init-water 180 --init-food 180
```

结果输出到：

- `outputs/level34_results/level3_result.json`

### 4.2 运行 Level4（在线 MPC）

```powershell
uv run scripts\run_level4.py --horizon 8 --scenarios 5 --init-water 220 --init-food 220
```

---

## 5. 参数说明

- `horizon`：MPC每次优化时向未来看的天数（可调）
  - 小：更快，偏短视
  - 大：更稳健，计算更慢
- `n_scenarios`：每次优化采样的天气情景数
  - 小：速度快
  - 大：对天气不确定性更稳健
- `init-water` / `init-food`：第0天起点初始购买量（箱）

> 简化理解：`horizon` 决定“看多远”，`n_scenarios` 决定“看几种未来”。

---

## 6. Examples（局部/全局视野）

四个样例脚本：

- `examples/level3_local_horizon.py`
- `examples/level3_global_horizon.py`
- `examples/level4_local_horizon.py`
- `examples/level4_global_horizon.py`

运行示例：

```powershell
uv run examples\level3_local_horizon.py
uv run examples\level3_global_horizon.py
uv run examples\level4_local_horizon.py
uv run examples\level4_global_horizon.py
```

---

## 7. 测试

### 7.1 只跑 MPC 测试

```powershell
uv run -m pytest -q tests\mpc
```

### 7.2 跑退化一致性测试（Level3/4 退化到 Level1/2）

```powershell
uv run -m pytest -q tests\test_level34_degenerate_level1_level2.py
```

### 7.3 全量测试（可选）

```powershell
uv run -m pytest -q
```

---

## 8. 与 tools 标准解对比验证

`src/mpc/validator.py` 的 `compare_solver_with_tool_reference(...)` 会：

1. 加载 `tools/level1-solved.py` 或 `tools/level2-solved.py`
2. 跑标准模型拿目标值与初始采购
3. 转换为当前 `GameConfig`
4. 用当前求解器在同条件下回测
5. 输出状态、目标值、gap（`tool - mpc`）

用于检查当前在线/退化求解器是否与标准实现一致。

---

## 9. 常见问题

### Q1：为什么测试耗时长？

- `horizon` 大 + `n_scenarios` 大 + MILP 求解器时间限制高，会明显增加时间。
- Fuzzing（Hypothesis）会反复生成案例，也会拉长测试时间。

建议：
- 开发期先用较小 `horizon/scenarios`；
- CI 中保留关键回归测试；
- 慢测试可单独运行。

### Q2：`init-water` / `init-food` 是什么？

是第0天在起点的初始购买量（当前脚本由命令行给定），随后再进入在线滚动决策。

---

## 10. 说明

- 本仓库默认面向 Windows 路径与命令。
- 若需 Linux/macOS，请将路径与命令分隔符替换为对应形式。

---

## 11. Benchmark（horizon × n_scenarios）

### 11.1 跑批量实验（online 模式）

```powershell
uv run scripts\benchmark_horizon_scenarios.py --level 3 --episodes 20 --horizons 2 4 6 8 10 --scenarios 1 2 4 8 --workers 4 --solver-threads 1 --seed 42
```

并行建议：

- `--workers`：并行进程数（多 solver 并行）
- `--solver-threads`：单 solver 线程数（建议固定为 `1`）

原始结果输出到：

- `outputs/benchmarks/raw/*.csv`

### 11.2 统计汇总

```powershell
uv run scripts\summarize_benchmark.py
```

汇总输出到：

- `outputs/benchmarks/summary/benchmark_summary.csv`

### 11.3 绘图（matplotlib + seaborn）

```powershell
uv run scripts\plot_benchmark.py --summary outputs/benchmarks/summary/benchmark_summary.csv
```

图表输出到：

- `outputs/benchmarks/figures/`

包含：

- 终局资金热力图（Mean Final Money）
- 成功率热力图（Success Rate）
- 时间-收益 Pareto 散点图（Time vs Money）
