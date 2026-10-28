# Thermal Expansion Error Prediction

用于预测机床热膨胀导致的三轴位置误差的研究与工程化复现实验。仓库同时保留：

- **24 件工件设计几何版（当前研究配置）**：GRU-Attention、原有 NC 融合与物理风险保持不变，新增物理输入只启用设计点/CAD 几何 11 项；
- **Hybrid v10（保留的原有主线）**：结合机床姿态、进给/主轴状态、温度统计特征与时序特征的多输出神经网络；
- **2023 LSTM 基线**：基于 16 路温度数据预测一维 `delta L`；
- MATLAB 与 Minitab 的早期对照实现、原始报告和拟合结果。

## 结果快照

不同实验的数据集与切分方式不相同，以下指标不能直接横向比较。

| 实验 | 评估口径 | 结果 |
| --- | --- | --- |
| 24 件固定数据集，新增设计几何 11 项 | 同一 1,978 个测试点，三轴预测残差范数 `< 0.04 mm` | **91.81%**（1,816/1,978）；相对全 30 项版本 +0.81 个百分点 |
| 24 件固定数据集，新增物理特征 30 项 | 同一 1,978 个测试点，同一阈值 | **91.00%**（1,800/1,978） |
| Hybrid v10，扩展数据集最优已记录调参（trial 612） | 1,288 个测试点，三维实际位置欧氏误差 `< 0.04` | **80.67%**（1,039/1,288）；平均/中位/最大距离为 0.031020 / 0.018602 / 0.374633 |
| Hybrid v10，最新部署快照（2026-07-23） | `模型10.json` 的 76 个测试点，同一 0.04 阈值 | **88.16%**（67/76）；平均距离 0.023261 |
| GRU-Attention 实验 `exp1_gru_highfreq_300` | 17 个 JSON，1,279 个测试点，三维位置距离 `< 0.04` | **81.16%**（1,038/1,279）；X/Y/Z 单轴分别为 92.10% / 90.15% / 95.47% |
| NC 融合 + 物理风险预处理 v1 | 同一 17 个 JSON、同一 1,279 个测试点，三维位置距离 `< 0.04` | **81.70%**（1,045/1,279）；物理高风险 45/1,279 |
| 2023 LSTM #95 | 温度→一维热伸长预测 | 训练 RMSE 0.126，测试 RMSE 0.127 |

Hybrid v10 模型说明见 [models/hybrid_v10/MODEL_CARD.md](models/hybrid_v10/MODEL_CARD.md)。最新部署快照的逐点预测、失败样本诊断和三轴图像位于 [results/hybrid_v10/model10_2026-07-23](results/hybrid_v10/model10_2026-07-23)。

24 件研究配置的架构、完整训练参数、已训练模型、逐点结果和复现命令见 [设计几何 11 项版本](experiments/fixed24_20261027/geometry11/README.md)。固定融合数据、NC 程序、设计点、STL 和 SHA256 清单位于同一[实验项目](experiments/fixed24_20261027/README.md)，由两个版本共用。91.81% 是在反复查看的固定测试集上选出的探索性结果，尚未通过新工件盲测。

81.16% 实验的架构、训练/评估命令、模型文件与逐点结果见 [results/exp1_gru_highfreq_300/README.md](results/exp1_gru_highfreq_300/README.md)。`lstm_keras/main.py` 和 [lstm_keras/exp1_gru_highfreq_300.py](lstm_keras/exp1_gru_highfreq_300.py) 对应已从头重训并验证的 Exp1 源码；后续架构实验不属于这组结果。

81.70% 版本的完整训练代码、架构图、训练参数、保存模型与逐点结果见 [lstm_nc_risk/README.md](lstm_nc_risk/README.md)。原始 NC 程序尚未入库，完整重训需按说明提供程序映射；同一测试集曾被多次查看，本次对照属于探索性评估。

## 项目结构

```text
data/
  hybrid_json/                         # 原 Hybrid v10 训练/预测所需的 17 个 JSON 数据文件（共 9,908 点）
  thermal expansion error original.csv # 2023 LSTM 原始温度数据（4,683 行 × 17 列）
lstm_keras/
  main.py                              # Hybrid v10 训练和评估入口
  auto_tune.py                         # 自动超参数搜索
  ensemble_train.py                    # 多随机种子集成训练
  tool_position_prediction.py          # 单工序预测命令行封装
  best model/                          # 2023 LSTM #95 模型与图像
models/hybrid_v10/
  best_full_dataset/                   # trial 612 模型与匹配的预处理工件
  deployment_snapshot/                 # 2026-07-23 部署快照模型与工件
results/hybrid_v10/                    # 已归档的当前结果与诊断
experiments/fixed24_20261027/         # 24 件固定数据、原 91.00% 基线及几何 11 项版本
lstm_matlab/                           # MATLAB LSTM 基线
lr_minitab/                            # Minitab 线性回归对照
Report.pdf                             # 2023 项目报告
```

## 环境配置

已验证的开发环境为 Windows + Python 3.11。推荐使用干净的虚拟环境，不提交本地 `env/` 或 `.venv/`。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 Windows 上导入 TensorFlow 报 DLL 初始化错误，请安装/修复 Microsoft Visual C++ Redistributable，并确认 CPU 支持该 TensorFlow 发行版要求的指令集；这是运行环境问题，不影响仓库内文件。

## 数据格式与训练逻辑

### Hybrid v10 JSON 数据

每个 `data/hybrid_json/*.json` 文件必须包含 `point_to_frame_ai_block_avg` 数组。每个点至少使用：

- `point_error_xyz`：三轴监督目标；
- `mapped_frame_pose`：X/Y/Z/A/C 姿态；
- `mapped_machine_state`：进给速度、主轴转速；
- `mapped_motion`：数值化运动特征；
- `ai_block_avg`：温度通道 ch3–ch6 和振动统计；
- `mapped_file_index`、`mapped_frame_id`、`point_index`：时序分组键。

程序按源文件和帧排序，生成温度、姿态差分、加工进度与热积累特征；再使用 `TIME_STEPS=10` 的帧级窗口构造时序输入。默认采用 `phase_blocks`：每个源文件在时间顺序上按块分为训练、验证、测试集（验证/测试比例均为 15%）。训练目标为三轴回归，损失函数额外惩罚三维位置距离超过 0.04 及单轴精度误差；同时训练误差分级辅助头。Early Stopping 按验证损失恢复最佳权重。

### 训练

以下命令使用仓库内 17 个 JSON 文件训练。新模型、预处理工件、预测 CSV 和图像将写入指定目录，不会覆盖归档的最佳模型。

```powershell
python .\lstm_keras\main.py --run-mode train --data-dir .\data\hybrid_json --output-dir .\results\train_20260804
```

常用环境变量均以 `TEE_` 开头，例如：

```powershell
$env:TEE_RANDOM_SEED = "2031"
$env:TEE_TIME_STEPS = "10"
$env:TEE_EPOCHS = "400"
$env:TEE_PATIENCE = "60"
$env:TEE_SPLIT_MODE = "phase_blocks" # 还可使用 random_phase_chunks、random_frames、leave_one_json_out
python .\lstm_keras\main.py --run-mode train --output-dir .\results\experiment_a
```

自动调参和集成训练分别可运行：

```powershell
python .\lstm_keras\auto_tune.py --help
python .\lstm_keras\ensemble_train.py --help
```

它们会产生较多中间模型和日志；请将输出目录放在 `results/`，不要直接提交自动搜索全过程。

### 评估与预测

默认命令加载归档的 trial 612 模型和匹配的 `.pkl` 预处理工件，使用所有 JSON 数据并按默认切分评估：

```powershell
python .\lstm_keras\main.py --run-mode predict_only --data-dir .\data\hybrid_json --output-dir .\results\evaluation
```

只评估一个加工过程时，指定其文件名（文件名含中文时请保留引号）：

```powershell
python .\lstm_keras\main.py --run-mode predict_only --data-dir .\data\hybrid_json --predict-source-name "模型10.json" --output-dir .\results\model10
```

要复现归档的 2026-07-23 部署快照，显式传入该模型及其匹配工件：

```powershell
python .\lstm_keras\main.py --run-mode predict_only `
  --data-dir .\data\hybrid_json `
  --model-path .\models\hybrid_v10\deployment_snapshot\hybrid_spatiotemporal_model_v10.keras `
  --artifact-path .\models\hybrid_v10\deployment_snapshot\hybrid_spatiotemporal_artifacts_v10.pkl `
  --predict-source-name "模型10.json" `
  --output-dir .\results\model10_recheck
```

评估会打印每轴 RMSE/MAE、三维位置距离统计和 `< 0.04` 的通过率，并生成：

- `tool_error_test_predictions_v10_hybrid_spatiotemporal.csv`：逐点真值、预测值和距离；
- `tool_error_test_position_failures_v10_hybrid_spatiotemporal.csv`：阈值失败样本；
- 按数据源、近阈值、特征区间、加工进度的诊断 CSV；
- X/Y/Z 预测对比图与位置距离柱图。

## 2023 LSTM 基线

`data/thermal expansion error original.csv` 为 16 个温度通道和 `delta L`。`lstm_keras/thermal expansion error.csv` 是旧流程生成的带 `Forward delta L` 列版本。早期 Keras Notebook、MATLAB 脚本与 Minitab 拟合结果用于研究对照；其历史中间模型和自动生成文件未纳入版本控制。

## 发布说明

本仓库已包含训练数据、当前模型、模型匹配的预处理工件，以及可核对的结果快照。`.gitignore` 排除了虚拟环境、缓存和大量可再生的调参中间产物；新增实验请输出到 `results/`。
