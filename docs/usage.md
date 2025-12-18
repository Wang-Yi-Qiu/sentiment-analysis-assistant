# 中文情感分析模型使用指南 (Chinese Sentiment Analysis Usage Guide)

本项目构建了一个高精度中文情感分析模型，结合了通用语料（clapAI）和垂直领域语料（中医药、电商）。

## 1. 环境准备 (Environment Setup)
已在您的 `learning_AI` 环境中配置完毕。
若需手动安装依赖，请执行：
```bash
/opt/homebrew/anaconda3/envs/learning_AI/bin/pip install -r requirements.txt
```

## 2. 训练模型 (Training)
Mac Mini 上已开启 MPS (Metal Performance Shaders) 加速。
运行以下命令开始训练（默认 3 个 Epoch，约需数小时）：

```bash
/opt/homebrew/anaconda3/envs/learning_AI/bin/python -m src.train
```

模型 Checkpoints 将保存在 `checkpoints/` 目录下。

## 3. 可视化交互界面 (Web UI) **[NEW]**
我们提供了一个简单易用的 Web 界面，可以直接在浏览器中测试模型：

```bash
/opt/homebrew/anaconda3/envs/learning_AI/bin/python src/app.py
```
运行后，复制终端显示的 URL (通常是 http://127.0.0.1:7860) 在浏览器打开即可。

## 4. 交互式教程 (Jupyter Notebook) **[NEW]**
如果您想一步步了解代码是如何运行的，并查看**数据分布图**和**训练曲线**，请运行 Jupyter Notebook：

```bash
/opt/homebrew/anaconda3/envs/learning_AI/bin/jupyter notebook notebooks/Chinese_Sentiment_Tutorial.ipynb
```

本教程包含详细的中文注释，适合小白入门。

## 5. 模型预测 (CLI Inference)
命令行预测方式依然保留：
```bash
/opt/homebrew/anaconda3/envs/learning_AI/bin/python src/predict.py
```

## 6. 关键文件说明
- `src/app.py`: Web 交互界面启动脚本。
- `src/visualization.py`: 用于绘制数据分布和训练曲线的工具。
- `notebooks/`: 包含交互式教程。
- `src/config.py`: 配置文件。
- `src/train.py`: 训练主脚本。
