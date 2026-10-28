# 24 件固定数据集：设计几何 11 项与 91.00% 对照

本目录保存完整、固定的 24 件实验输入与两版可复现模型：[当前设计几何 11 项版本](geometry11/README.md)测试准确率 **91.81%**；[原全 30 项基线](baseline/accuracy_summary.json)为 **91.00%**。`src/` 是训练时的网络、特征、风险预处理及评估代码快照；原仓库架构未修改。`data/` 含融合点、设计点、NC 程序和 STL；`source_manifest.json` 记录逐文件 SHA256 和来源。

## 数据与结果

- 共 24 件工件，排除 1739、1892，包含 0029、1820。1673 使用同一测量表导出的实测点与设计点均满足 Z≥0 的 742 点版本。
- 按 NC 加工阶段分块，每件均有训练、验证、测试点：分别为 10,372、1,976、1,978 点。逐点身份保存在 `baseline/*_point_identities.csv`。
- 从随机初始化训练：种子 2031，10 帧时序窗，batch 128，最多 300 轮，早停耐心 60。模型选择依赖验证集。
- 三轴预测误差残差的欧氏范数 <0.04 mm 为达标。保存模型 `baseline/physical_geometry_nc_model.keras` 在测试集 **1800/1978 = 91.00%**。
- 已从本目录的代码和数据重新加载模型，测试集逐点预测最大差为 **0 mm**。原始预测、模型、预处理器、训练记录和审计文件均在 `baseline/`。

下图展示保留作对照的原全 30 项特征版本；当前 11 项版本的架构见 [geometry11/README.md](geometry11/README.md)。

```mermaid
flowchart LR
  A[温度、振动、运动时序] --> B[10 帧 GRU + Attention]
  C[原 NC 特征] --> D[静态融合]
  E[设计点/CAD 几何 11 项] --> D
  F[局部 NC/运动 19 项] --> D
  B --> G[融合层与三轴误差预测]
  D --> G
  R[物理风险分数与逐点权重] --> G
```

## 复现

在本目录安装 `requirements.txt` 的依赖，然后执行：

```powershell
python train_baseline.py --mode verify
python train_baseline.py --mode train
python geometry11/train_geometry11.py --mode verify
python geometry11/train_geometry11.py --mode train
```

两个 `verify` 均根据相对路径生成运行清单，核对输入哈希，重载各自保存的模型并逐点比较预测结果；`train` 从随机初始化重训，写入被忽略的输出目录，不覆盖归档模型。基线参数见 `training_parameters.json`，几何版参数见 `geometry11/training_parameters.json`。运行时清单保存在被忽略的 `runtime/`。

本次只上传被选作正式版本的几何 11 项模型及其结果；其他消融实验仍在本地，不属于提交内容。几何版相对全特征版提高 16/1978 点，但两版共用已多次查看的固定测试集；是否稳定更优需要新工件盲测。
