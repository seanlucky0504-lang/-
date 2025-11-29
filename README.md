# 红酒质量数据挖掘（中文说明）

基于 [awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets) 列表中的 UCI 红酒质量数据集，提供**离线可运行**的一站式数据挖掘流程。`data/winequality-red-sample.csv` 保存了数据快照，保证在无网络环境下也能完成预处理、探索、建模与评估。

## 流程与评估标准
1. **数据获取与预处理**：清洗列名、去重，按质量分数（≥7）生成二分类标签。
2. **探索性分析（EDA）**：生成描述性统计、质量分布柱状图、特征相关性热力图；支持调用 DeepSeek API 做“任意问题—上下文”链式分析；额外输出可交互的浏览器可视化界面。
3. **建模与可视化**：使用纯标准库实现逻辑回归，包含标准化、训练/测试划分，并输出混淆矩阵 SVG 与交互式散点/直方图视图。
4. **评估方案**：在 20% 留出集上以 **F1-score 与 ROC-AUC** 作为统一评估标准，并将结果写入 `reports/model_metrics.json`。

## 运行方式
```bash
# 基础流程（预处理 + EDA + 训练评估）
python scripts/wine_quality_analysis.py

# 附带 DeepSeek 链式探索，需提前设置 DEEPSEEK_API_KEY
DEEPSEEK_API_KEY=your_key \
python scripts/wine_quality_analysis.py --ask "给出提升 F1 的特征工程建议"
```

### 如何配置和调用 DeepSeek API Key？
1. **获取密钥**：在 DeepSeek 控制台创建密钥，复制得到的字符串。
2. **临时设置（推荐在命令行使用 `--ask`）**：在运行命令前导出环境变量，例如：
   ```bash
   export DEEPSEEK_API_KEY="你的密钥"
   # 如需自定义网关，可选设定 DEEPSEEK_API_URL，默认使用官方地址
   # export DEEPSEEK_API_URL="https://api.deepseek.com/chat/completions"

   python scripts/wine_quality_analysis.py --ask "分析酒精度与质量的关系"
   ```
3. **一次性调用（不修改 shell 环境）**：直接在命令前传入变量即可：
   ```bash
   DEEPSEEK_API_KEY="你的密钥" \
   python scripts/wine_quality_analysis.py --ask "推荐提升 F1 的特征选择方案"
   ```
4. **未配置密钥时的行为**：脚本会自动降级到离线模式，返回基于本地统计的中文建议，并提示需要提供 `DEEPSEEK_API_KEY` 以获得在线分析。

> 说明：`scripts/wine_quality_analysis.py` 会读取 `DEEPSEEK_API_KEY`（必填）和 `DEEPSEEK_API_URL`（可选，缺省为官方地址），并在 `reports/deepseek_chain_output.txt` 中保存返回内容。

输出位置：
- `reports/summary_stats.csv`：各特征统计量
- `reports/model_metrics.json`：评估指标（F1、ROC-AUC）与评估标准说明
- `reports/figures/*.svg`：质量分布、相关性热力图、混淆矩阵
- `reports/interactive_dashboard.html`：交互式可视化（直方图、散点图、相关系数热力图，可在浏览器直接打开，无需服务器）
- `reports/deepseek_chain_output.txt`：DeepSeek 返回的中文探索建议

## 更多探索方案（可直接作为 `--ask` 提问）
- “依据方差最大的前三个理化指标，设计新的交互项是否有助于提升模型表现？”
- “在类别不平衡背景下，建议的采样或阈值移动策略是什么？”
- “请列出与酒精度、挥发性酸度最强相关的前 3 个特征，并给出可视化思路。”
- “若改为多分类预测原始质量分数，应如何调整评估指标与模型选择？”

> 注：代码中提供了降级策略，如果未设置 `DEEPSEEK_API_KEY`，`--ask` 将返回基于本地统计的离线建议。
