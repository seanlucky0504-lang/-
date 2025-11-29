# 数据挖掘任务完成说明

本文档概述本项目如何覆盖数据预处理、探索性分析、算法建模、评估与 AI 辅助五个环节，方便复现或查阅。

## 1. 数据预处理
- **数据来源**：使用 UCI Red Wine Quality 数据集，离线快照位于 `data/winequality-red-sample.csv`，原始链接：[UCI 存储库](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)，亦列于 Awesome Public Datasets（https://github.com/awesomedata/awesome-public-datasets）。
- **字段清洗**：在加载时统一空格和分隔符，确保列名可用于后续计算。详见 `load_data` 中的列名清洗逻辑。
- **去重与标签生成**：`preprocess_data` 会移除重复记录并基于品质分数生成二分类标签 `quality_label`（质量≥7 记为 1，反之为 0），为分类任务做准备。
- **统计汇总产物**：数值型特征的均值、方差、最小值和最大值会写入 `reports/summary_stats.csv`，为后续分析和大模型提示提供基础统计。

## 2. 探索性分析与统计可视化
- **分布与相关性**：脚本在 `reports/figures/` 下生成品质分布条形图、相关系数热力图以及逻辑回归混淆矩阵等 SVG 文件，便于快速理解标签平衡、特征相关性与模型表现。
- **交互式仪表板**：运行脚本后会输出 `reports/interactive_dashboard.html`，内含可调特征选择的散点图、直方图与热力图，支持浏览器内交互式探索。
- **可复用 EDA 提示**：`--ask` 选项会将关键统计拼装成上下文，可在有/无 DeepSeek API Key 的情况下返回在线或离线的探索建议。

## 3. 数据挖掘任务与算法
- **任务定义**：基于理化指标预测葡萄酒品质是否为高分（质量≥7），属于二分类问题。
- **算法实现**：脚本内置纯标准库实现的逻辑回归训练与推理（无第三方依赖），支持 80/20 随机划分训练与测试集。可视化混淆矩阵用于直观展示分类结果。
- **平台与工具**：全部逻辑在 `scripts/wine_quality_analysis.py` 中完成，可直接使用系统 Python 运行；如需 AI 辅助，可通过 DeepSeek API 生成探索性分析建议。

## 4. 评估方案与结果
- **评估标准**：使用 F1-score 与 ROC-AUC 在划分出的 20% 测试集上评价分类器，兼顾类别不平衡与阈值无关的排序能力。
- **示例结果**：当前产物中，`reports/model_metrics.json` 给出了训练/测试集的 F1 与 ROC-AUC，可根据再次运行结果自动更新。

## 5. AI 工具辅助
- **DeepSeek 链式调用**：提供 `--ask "你的问题"` 接口，自动汇总数据概览并发送至 DeepSeek（需配置 `DEEPSEEK_API_KEY`），返回中文分析建议。
- **降级与兼容性**：未提供密钥或网络不可达时，会输出基于本地统计的离线建议。支持环境变量、`.env` 文件与 CLI 参数三种方式传递密钥，适配 Linux/Windows/PowerShell。

> 快速开始：`python scripts/wine_quality_analysis.py --ask "分析酒精度与质量的关系"`，可同时生成数据预处理、可视化、建模与评估的所有产物。
