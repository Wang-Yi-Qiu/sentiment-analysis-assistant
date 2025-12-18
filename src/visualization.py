import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

# 设置中文字体 (尝试自动寻找可用字体)
def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False

def plot_data_distribution(dataset_dict):
    """
    绘制数据集中 Positive/Neutral/Negative 的分布饼图
    """
    set_chinese_font()
    
    # 统计数量
    train_labels = dataset_dict['train']['label'] if 'label' in dataset_dict['train'].features else [x['label'] for x in dataset_dict['train']]
    
    # 映射回字符串以便显示
    id2label = {0: 'Negative (消极)', 1: 'Neutral (中性)', 2: 'Positive (积极)'}
    labels_str = [id2label.get(x, str(x)) for x in train_labels]
    
    df = pd.DataFrame({'Label': labels_str})
    counts = df['Label'].value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('训练集情感分布')
    plt.show()

def plot_training_history(log_history, save_path=None):
    """
    根据 Trainer 的 log_history 绘制 Loss 和 Accuracy 曲线
    """
    set_chinese_font()
    
    if not log_history:
        print("没有可用的训练日志。")
        return
    
    df = pd.DataFrame(log_history)
    
    # 过滤掉没有 loss 或 eval_accuracy 的行
    train_loss = df[df['loss'].notna()]
    eval_acc = df[df['eval_accuracy'].notna()]
    
    plt.figure(figsize=(14, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss', color='salmon')
    if 'eval_loss' in df.columns:
        eval_loss = df[df['eval_loss'].notna()]
        plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', color='skyblue')
    plt.title('训练损失 (Loss) 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy Curve
    if not eval_acc.empty:
        plt.subplot(1, 2, 2)
        plt.plot(eval_acc['epoch'], eval_acc['eval_accuracy'], label='Validation Accuracy', color='lightgreen', marker='o')
        plt.title('验证集准确率 (Accuracy)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        print(f"Saving plot to {save_path}...")
        plt.savefig(save_path)
    # plt.show() # 在服务器或脚本运行时通常不需要阻塞显示，或者用户可能看不到弹窗

def load_and_plot_logs(log_dir):
    """
    从 checkpoint 目录加载 trainer_state.json 并绘图
    """
    json_path = os.path.join(log_dir, 'trainer_state.json')
    if not os.path.exists(json_path):
        print(f"未找到日志文件: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    plot_training_history(data['log_history'])
