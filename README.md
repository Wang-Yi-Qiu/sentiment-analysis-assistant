# 基于 BERT 的中文情感分析系统项目报告
> **Project Report: BERT-based Chinese Sentiment Analysis System**
> *此文档旨在辅助生成项目汇报 PPT，详细记录了从 0 到 1 的构建全过程。*

## 1. 项目背景与目标 (Project Background & Goals)
### 1.1 背景
随着互联网评论数据的爆炸式增长，如何自动识别中文文本背后的情感倾向（积极/消极/中性）成为关键需求。传统机器学习方法在语义理解上存在局限，因此本项目采用深度学习模型 BERT 进行构建。

### 1.2 核心目标
1.  **高精度模型**：基于预训练 BERT 模型进行微调 (Fine-tuning)，实现对中文评论的精准分类。
2.  **多领域覆盖**：融合通用语料 (clapAI) 与垂直领域语料 (中医/电商)，提升泛化能力。
3.  **全流程落地**：包含数据清洗、模型训练、可视化监控、Web 交互演示及云端部署支持。

---

## 2. 技术架构 (Technical Architecture)

| 组件 (Component) | 技术选型 (Technology) | 说明 (Description) |
| :--- | :--- | :--- |
| **基础模型 (Base Model)** | **Google BERT (bert-base-chinese)** | 12层 Transformer 编码器，具有强大的中文语义理解能力。 |
| **深度学习框架 (DL Framework)** | **PyTorch + Hugging Face Transformers** | 提供灵活的模型构建与训练接口。 |
| **硬件加速 (Accelerator)** | **MPS (Apple Silicon) / CUDA (Cloud)** | 代码自动适配 Mac 本地加速与云端 NVIDIA GPU 加速。 |
| **交互界面 (Web UI)** | **Gradio** | 快速构建可视化的模型演示网页。 |
| **数据分析 (Analytics)** | **Matplotlib + Seaborn** | 用于绘制数据分布图与训练损失/准确率曲线。 |

---

## 3. 详细实施步骤 (Implementation Steps)

### 步骤一：环境搭建与硬件适配 (Environment Setup)
*   **挑战**：在 Mac Mini (M系列芯片) 上实现高效训练。
*   **解决方案**：利用 PyTorch 的 `mps` 后端，代码中实现了自动设备检测逻辑：优先使用 MPS (Mac)，其次 CUDA (NVIDIA)，最后 CPU。
*   **成果**：在 Mac 本地环境下成功开启硬件加速，大幅缩短训练时间。

### 步骤二：数据工程 (Data Engineering)
*   **多源异构数据融合**：
    *   **通用数据**：`clapAI/MultiLingualSentiment` (筛选中文部分)。
    *   **垂类数据**：`OpenModels/Chinese-Herbal-Medicine-Sentiment` (医疗/电商领域)。
*   **数据清洗管道 (`src/dataset.py`)**：
    *   剔除无效评论（如“默认好评”、“无填写内容”）。
    *   过滤过短文本（长度 < 2）。
    *   **标签统一**：将不同数据集的标签统一映射为标准格式：`0 (Negative)`, `1 (Neutral)`, `2 (Positive)`。
*   **优化**：实现了 **多进程 (Multiprocessing)** 数据处理，利用多核 CPU 加速 Tokenization（分词）过程。

### 步骤三：模型训练与微调 (Model Training)
*   **策略**：全参数微调 (Full Fine-tuning)。
*   **配置**：Batch Size 32, Learning Rate 2e-5, Epochs 3。
*   **智能特性**：
    *   **实时监视 (`src/monitor.py`)**：专门编写监控脚本，读取 Checkpoint 日志，实时输出 Loss 和 Accuracy 变化。
    *   **断点续训**：支持从最新的 Checkpoint 恢复训练，防止意外中断导致前功尽弃。
    *   **云端适配 (`train_cloud.py`)**：生成了独立的单文件训练脚本，支持一键上传至 AutoDL/Colab 等云服务器，自动下载数据并利用 CUDA 极速训练。

### 步骤四：结果可视化与评估 (Visualization & Eval)
*   **指标**：Accuracy (准确率), F1-Score (F1分数), Precision, Recall。
*   **可视化 (`src/visualization.py`)**：
    *   **数据分布图**：通过饼图展示正负样本比例，确保数据平衡。
    *   **训练曲线**：自动绘制 Loss 下降曲线和 验证集 Accuracy 上升曲线，直观判断模型收敛情况。

### 步骤五：应用交付 (Deployment)
*   **Web 演示 (`demo/web_demo.py`)**：
    *   开发了基于 Gradio 的 Web 界面。
    *   支持用户输入任意中文文本，实时返回情感倾向及置信度分数。
    *   包含预设样例，方便快速测试。
*   **交互式教程 (`notebooks/`)**：提供了详细注释的 Jupyter Notebook，用于教学和演示完整流程。

---

## 4. 项目亮点 (Project Highlights)
1.  **跨平台兼容**：一套代码同时完美支持 Mac (MPS) 和 Linux/Windows (CUDA)。
2.  **工程化规范**：目录结构清晰 (`src`, `data`, `results`, `checkpoints`)，模块化设计高。
3.  **用户体验**：
    *   训练过程不仅有进度条，还有专门的 Monitor 脚本。
    *   Web 界面美观易用，支持详细的分数展示。
    *   云端脚本 `train_cloud.py` 极大降低了部署门槛。

---

## 5. 成果展示 (Results)
*(此部分可用于 PPT 插入截图)*
- **训练效果**：在验证集上 Accuracy 稳步提升（具体数值参考 Monitor 输出）。
- **演示界面**：Web UI 成功运行，能够准确识别“物流太慢”（消极）和“强烈推荐”（积极）等语义。

---

## 6. 如何运行 (Quick Start)
1.  **本地训练**: `python -m src.train`
2.  **开启监控**: `python src/monitor.py`
3.  **启动演示**: `python demo/web_demo.py`
