# 机器人系统辨识 — 参数辨识与逆动力学验证

本项目用于双足机器人（biped_s49）左臂的系统辨识与逆动力学验证。基于激励轨迹运行后捕获的传感器数据，通过最小二乘法辨识机器人的惯性参数、摩擦参数和转子惯量（armature），并与 URDF 标称模型的逆动力学预测进行误差对比。

## 项目结构

```
.
├── experiments/
│   ├── generate_excitation.py      # 激励轨迹生成（详见 README.md）
│   ├── identify_params.py           # 参数辨识（最小二乘 + 正则化）
│   ├── compare_identified_params.py  # 辨识参数验证（加载 NPZ，对比验证集效果）
│   └── compare_inverse_dynamics.py  # 逆动力学扭矩对比（RNEA vs 实测）
├── system_identification/
│   ├── utils.py                     # 工具函数（QR分解、Savitzky-Golay滤波、路径查找）
│   ├── inertia_model.py             # 惯性模型（基于 Pinocchio 的回归矩阵计算）
│   ├── excitation_generator.py      # 傅里叶轨迹生成（含 tanh 有界映射）
│   └── excitation_optimization.py   # 优化目标函数与摩擦回归子生成
├── robot_description/
│   └── biped_s49_description/
│       └── urdf/                    # 机器人 URDF 模型（左臂/右臂/全身）
├── log_data/                        # 训练用传感器数据（详见下文）
├── test_data/                       # 验证集传感器数据（格式同 log_data）
├── traj_data/
│   └── csv_data/                    # 生成的激励轨迹 CSV 文件
├── environment.yml                  # Conda 环境配置
└── setup.py                         # Python 包安装配置
```

## 数据存放位置与格式

### 数据目录

所有捕获的数据存放在 `log_data/` 目录下，包含以下 CSV 文件：

| 文件 | 说明 | 列格式 |
|------|------|--------|
| `sensors_joint_q.csv` | 关节位置（滤波后） | `time, j12, j13, ..., j25`（14 个关节） |
| `sensors_joint_v.csv` | 关节速度（滤波后） | `time, j12, j13, ..., j25`（14 个关节） |
| `sensors_joint_torque.csv` | 关节扭矩（滤波后） | `time, j12, j13, ..., j25`（14 个关节） |
| `csv_trajectory_a.csv` | 指令期望加速度 | `time, j0, j1, ..., j13`（14 个关节） |

### 关节映射

传感器数据包含全身 14 个关节（j12-j25），辨识脚本自动提取**前 7 列**（j12-j18）作为左臂数据。指令加速度文件中 j0-j6 对应左臂 7 个关节。

### 时间戳

所有 CSV 文件第一列为时间戳（秒），采样率约 500Hz。各文件时间戳对齐，无需额外同步。

## 脚本一：identify_params.py — 参数辨识

### 功能

1. 加载滤波后的传感器数据（q, v, torque）和指令加速度
2. 对速度做中心差分 + Savitzky-Golay 滤波计算加速度
3. 裁减首尾边界帧（消除差分毛刺）
4. 构建回归矩阵：`[惯性回归子 | 摩擦回归子 | armature 回归子]`
5. QR 降维分析条件数
6. **变量投影**（Variable Projection）：非线性优化摩擦平滑参数 `vbrk`，内层用正则化最小二乘求解 `φ`：`min ||Y(vbrk)·φ - τ||² + λ·||φ - φ₀||²`（含 per-joint RMS 归一化）
7. 输出：每关节质量/质心、摩擦参数、armature、三方误差对比表
8. 绘图：传感器数据概览、期望加速度对比、实测 vs 标称(RNEA) vs 辨识扭矩
9. **验证集验证**：在 `test_data/` 上用辨识参数构建回归器并预测力矩，输出验证集三方误差对比表与绘图

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot` | str | `biped_s49_left_arm` | 机器人 URDF 名称 |
| `--friction_model` | str | `symmetric` | 摩擦模型。`symmetric`（对称：库仑 tanh + 粘性线性）/ `asymmetric`（正负方向独立） |
| `--data_dir` | str | `./log_data` | 训练用传感器数据目录 |
| `--val_dir` | str | `./test_data` | 验证集数据目录，辨识完成后在该数据上验证泛化精度 |
| `--trim` | int | `1` | 裁减首尾帧数，消除差分边界毛刺 |
| `--output_urdf` | str | `None` | 辨识结果回写 URDF 路径（默认不回写） |
| `--save_params` | str | `experiments/identified_params.npz` | 保存辨识参数 `φ` 和 `vbrk` 到 NPZ 文件，供独立验证脚本加载使用 |
| `--lambda_reg` | float | `0.0001` | 正则化权重，拉向 URDF 初始猜测 |
| `--vbrk_init` | float | `0.0001` | `vbrk` 初始猜测（摩擦平滑参数，控制 tanh 过渡平滑度） |
| `--vbrk_lb` | float | `1e-6` | `vbrk` 下界 |
| `--vbrk_ub` | float | `0.1` | `vbrk` 上界 |
| `--val_acc_method` | str | `central` | 验证集加速度计算方式。`central`：中心差分 + Savitzky-Golay 多项式滤波（非因果，利用前后样本）；`causal`：因果后向差分 + 因果 Savitzky-Golay（仅用过去样本，无未来信息）；`zero`：加速度全为零，仅评估重力和摩擦项贡献 |

### 辨识参数布局

参数向量 `φ` 的结构为：

```
φ = [ 惯性参数(10×N) | 摩擦参数 | armature(N) ]
```

| 部分 | 参数数/关节 | 说明 |
|------|------------|------|
| 惯性 | 10 | 质量、质心杠杆(3)、惯性张量(6) |
| 摩擦（对称） | 2 | 库仑系数 Fc、粘性系数 Fv |
| 摩擦（非对称） | 4 | Fc⁺, Fv⁺, Fc⁻, Fv⁻ |
| Armature | 1 | 转子惯量（折算到关节侧） |

### 初始猜测

| 参数 | 初始猜测来源 |
|------|------------|
| 惯性 | URDF 标称值（`inertia_model.dyn_param`） |
| 库仑摩擦 | URDF `model.friction` |
| 粘性摩擦 | URDF `model.damping` |
| Armature | 0 |

### 摩擦回归子

摩擦基函数与仿真模型一致，使用 `tanh(v/eps)` 作为库仑摩擦的平滑符号函数：

- 初始 `vbrk=0.00005`（`vcoul=vbrk*2`），通过**变量投影**非线性优化得到最优 `vbrk`
- 库仑特征：`tanh(v / (2·vbrk))`
- 粘性特征：`v`（线性）

优化器使用 `scipy.optimize.least_squares`（`trf` 方法），在 `[vbrk_lb, vbrk_ub]` 范围内搜索使残差最小的 `vbrk`，内层求解正则化最小二乘得到 `φ`。

### 运行示例

```bash
# 默认运行（左臂，对称摩擦，正则化 0.0001）
conda run -n sysid python experiments/identify_params.py

# 裁减更多边界帧
python experiments/identify_params.py --trim 10

# 非对称摩擦模型
python experiments/identify_params.py --friction_model asymmetric

# 加大正则化（更接近 URDF 标称值）
python experiments/identify_params.py --lambda_reg 0.01

# 回写辨识结果到指定 URDF
python experiments/identify_params.py --output_urdf robot_description/biped_s49_description/urdf/biped_s49_left_arm_identified.urdf
```

### 输出说明

#### 终端输出

1. **加载信息**：样本数、关节数
2. **回归矩阵**：形状、QR 降维后条件数
3. **Per-joint 扭矩 RMS**：用于归一化的参考值
4. **辨识结果**：
   - 每关节质量与质心
   - 每关节库仑/粘性摩擦系数
   - 每关节 armature
5. **三方误差对比表**：

```
================================================================================
3-way RMS error comparison: measured vs nominal (RNEA) vs identified
----------------------------------------------------------------------------------------
 Joint |   RMS meas |   Nominal err    ratio |    Ident err   ratio
  J1   |     6.107 |     0.395171  0.0647 |     0.231711  0.0379
  ...
================================================================================
```

- **RMS meas**：实测扭矩的 RMS 幅值
- **Nominal err / ratio**：URDF 标称模型（RNEA + URDF 摩擦 + armature=0.05）与实测的 RMS 误差及比例
- **Ident err / ratio**：辨识后模型与实测的 RMS 误差及比例

辨识完成后，在 `test_data/` 上输出 **验证集三方误差对比表**（格式同上），用于评估泛化能力。

#### 图表输出

1. **传感器数据概览**（3 行子图）：位置 q、速度 v、力矩 tau（裁减后数据）
2. **期望加速度大图**（7 行子图，每关节一行）：蓝色=期望加速度（来自 `csv_trajectory_a.csv`），红色=中心差分加速度（来自速度滤波后），裁减后数据
3. **训练集扭矩对比图**（7 行子图）：蓝色=实测扭矩，绿色=标称 RNEA 预测，红色虚线=辨识后预测
4. **验证集扭矩对比图**（7 行子图）：在 `test_data/` 上的三方力矩对比，用于验证辨识参数泛化能力

## 脚本二：compare_identified_params.py — 辨识参数独立验证

### 功能

1. 加载辨识保存的 NPZ 参数文件（`φ` 和 `vbrk`）
2. 加载验证集数据（q, v, torque），计算加速度
3. 用加载的 `φ` 和 `vbrk` 构建回归矩阵，预测力矩
4. 对比三方误差：实测 vs 标称 RNEA vs 辨识参数
5. 绘图：力矩对比 + 加速度对比

无需 URDF 回写或 Pinocchio 模型参与预测，纯矩阵运算 `τ = Y(q,v,a)·φ` 完成逆动力学预测。

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--params` | str | `experiments/identified_params.npz` | NPZ 参数文件路径 |
| `--data_dir` | str | `./test_data` | 验证集数据目录 |
| `--trim` | int | `1` | 裁减首尾帧数 |
| `--val_acc_method` | str | `central` | 加速度计算方式。`central`：中心差分 + Savgol；`causal`：因果后向差分 + 因果 Savgol；`zero`：加速度全为零（仅评估重力和摩擦） |

### 运行示例

```bash
# 默认：加载 experiments/identified_params.npz，在 test_data 上验证
python experiments/compare_identified_params.py

# 使用因果加速度验证
python experiments/compare_identified_params.py --val_acc_method causal

# 无加速度输入验证（仅重力和摩擦贡献）
python experiments/compare_identified_params.py --val_acc_method zero

# 指定其他参数文件或数据目录
python experiments/compare_identified_params.py --params my_params.npz --data_dir ./other_data
```

## 脚本三：compare_inverse_dynamics.py — 逆动力学验证

### 功能

1. 加载滤波后的传感器数据（q, v, torque）
2. 中心差分 + Savitzky-Golay 滤波计算加速度
3. 裁减首尾边界帧
4. 加载 pinocchio URDF 模型，逐帧计算逆动力学扭矩
5. 逆动力学采用分步计算（与 RNEA 等精度）：
   - `pin.crba` → M(q)
   - `pin.computeCoriolisMatrix` → C(q, q̇)
   - `pin.computeGeneralizedGravity` → g(q)
   - τ = M·q̈ + C·q̇ + g
6. 摩擦补偿：`coulomb·tanh(v/eps) + viscous·v + armature·q̈`（参数来自 URDF）
7. 绘图：每关节实测 vs RNEA 预测扭矩对比，标注 RMS 误差

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot` | str | `biped_s49_left_arm` | 机器人 URDF 名称 |
| `--data_dir` | str | `./log_data` | 传感器数据目录 |
| `--trim` | int | `10` | 裁减首尾帧数 |
| `--window_length` | int | `21` | Savitzky-Golay 滤波窗长（奇数） |
| `--polyorder` | int | `3` | Savitzky-Golay 多项式阶数 |

### 运行示例

```bash
# 默认运行
python experiments/compare_inverse_dynamics.py

# 更大滤波窗口（更平滑）
python experiments/compare_inverse_dynamics.py --window_length 51 --polyorder 3

# 裁减更多边界帧
python experiments/compare_inverse_dynamics.py --trim 20
```

## 推荐使用流程

### 1. 生成激励轨迹

```bash
python experiments/generate_excitation.py --robot biped_s49_left_arm
```

生成的轨迹 CSV 保存到 `traj_data/csv_data/`，包含 `t, q1...qN, v1...vN, a1...aN`。

### 2. 在实机/仿真上运行激励轨迹

将生成的轨迹下发到机器人控制器，以位置控制模式运行，同时记录传感器数据。运行结束后将捕获的数据放入 `log_data/` 目录，确保文件名和格式与上文一致。

### 3. 验证标称模型精度

```bash
python experiments/compare_inverse_dynamics.py --robot biped_s49_left_arm --trim 10
```

查看标称 URDF 模型的逆动力学预测与实测扭矩的误差，判断是否需要辨识。

### 4. 执行参数辨识

```bash
python experiments/identify_params.py --robot biped_s49_left_arm --friction_model symmetric --trim 10 --lambda_reg 0.0001
```

辨识完成后自动保存参数到 `experiments/identified_params.npz`。查看训练集三方误差对比表，确认辨识后的模型误差是否低于标称模型。

### 5. 验证辨识泛化能力

将另一组采集数据放入 `test_data/` 目录，重新运行：

```bash
python experiments/identify_params.py --robot biped_s49_left_arm --lambda_reg 0.0001
```

脚本会在训练集辨识完成后，自动在 `test_data/` 上构建回归器并预测，输出验证集三方误差对比表与绘图，用于评估辨识参数的泛化能力。

### 6. 独立验证辨识参数（可选）

加载保存的 NPZ 参数文件，在任意验证集上独立验证，无需重新辨识：

```bash
# 默认加载 experiments/identified_params.npz
python experiments/compare_identified_params.py

# 无加速度输入验证（仅重力和摩擦）
python experiments/compare_identified_params.py --val_acc_method zero
```

### 7. 回写 URDF（可选）

```bash
python experiments/identify_params.py --output_urdf robot_description/biped_s49_description/urdf/biped_s49_left_arm_identified.urdf
```

将辨识后的摩擦参数回写到 URDF 文件，用于后续控制。

## 加速度计算说明

辨识中加速度采用**中心差分 + Savitzky-Golay 滤波**方式计算：

```
a[i] = (v[i+1] - v[i-1]) / (t[i+1] - t[i-1])
```

滤波参数（`identify_params.py` 中硬编码）：
- `window_length = 801`
- `polyorder = 5`

如需调整，修改 `load_data()` 中的 `savgol_filter_acceleration` 调用参数。

### 验证集加速度计算方式（可选）

验证集加速度计算方式通过 `--val_acc_method` 选择：

| 选项 | 方法 | 说明 |
|------|------|------|
| `central`（默认） | 中心差分 + Savitzky-Golay 多项式滤波 | 与训练集一致，`a[i] = (v[i+1] - v[i-1]) / (t[i+1] - t[i-1])`，再用 `savgol_filter`（窗长 801，阶数 5）平滑。非因果，利用前后样本，精度高但不适用于在线场景 |
| `causal` | 因果后向差分 + 因果 Savitzky-Golay | `a[k] = (v[k] - v[k-1]) / (t[k] - t[k-1])`，再用 `savgol_coeffs`（pos=window-1）做因果平滑。仅用过去和当前样本，无未来信息，适合在线/实时部署评估 |
| `zero` | 加速度全为零 | `a = 0`，回归矩阵中惯性和 armature 项无贡献，仅评估重力和摩擦项的辨识精度。适用于静态/准静态场景验证 |

验证集加速度图的图例和标题会根据所选方法显示对应名称（"Central Diff + Savgol"、"Causal Savgol" 或 "Zero Acceleration"）。

```bash
# 默认：中心差分 + 多项式滤波
python experiments/identify_params.py --val_acc_method central

# 因果多项式滤波（评估在线部署效果）
python experiments/identify_params.py --val_acc_method causal
```

`compare_inverse_dynamics.py` 中滤波参数通过命令行 `--window_length` 和 `--polyorder` 调整。

## 正则化与归一化

### 正则化

求解目标：`min_{vbrk, φ} ||Y(vbrk)·φ - τ||² + λ·||φ - φ₀||²`

采用**变量投影**（Variable Projection）两层求解：

1. **外层**（非线性）：优化 `vbrk`（摩擦 tanh 平滑参数），使用 `scipy.optimize.least_squares`（`trf`）
2. **内层**（线性）：给定 `vbrk`，求解正则化最小二乘得到 `φ`

- `φ₀`：初始猜测（惯性/摩擦来自 URDF，armature=0）
- `λ`：正则化权重（`--lambda_reg`），越大越接近标称值，越小越拟合数据
- 实现：增广系统 `[Y; √λ·I]·φ = [τ; √λ·φ₀]`

### Per-joint RMS 归一化

对每个关节的回归器行和扭矩按该关节的扭矩 RMS 归一化：

```
Y_norm[i,j,:] = Y[i,j,:] / rms(τ_j)
τ_norm[i,j]   = τ[i,j] / rms(τ_j)
```

使大小扭矩关节的误差贡献均衡，避免大扭矩关节淹没小扭矩关节的拟合。
