
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
class Config:
    # 基础模型
    BASE_MODEL = "google-bert/bert-base-chinese"
    
    # 目录配置 (根据用户要求指定)
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    
    # 标签配置
    NUM_LABELS = 3
    LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
    ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # 训练参数
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.1
    SAVE_STEPS = 500
    LOGGING_STEPS = 100

# ==========================================
# 2. 工具函数 (Utils)
# ==========================================
def ensure_directories():
    """ 确保所有必要的目录存在 """
    for path in [Config.DATA_DIR, Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.DOCS_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f">>> Created directory: {path}")

def plot_training_history(log_history, save_path):
    """ 绘制训练曲线并保存 """
    try:
        # 设置字体 (尝试通用中文字体，云端可能缺失，回退到英文)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        df = pd.DataFrame(log_history)
        train_loss = df[df['loss'].notna()]
        eval_acc = df[df['eval_accuracy'].notna()]
        
        if train_loss.empty:
            return

        plt.figure(figsize=(12, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(train_loss['epoch'], train_loss['loss'], label='Train Loss', color='#FF6B6B')
        if 'eval_loss' in df.columns:
            eval_loss = df[df['eval_loss'].notna()]
            plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Val Loss', color='#4ECDC4')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        if not eval_acc.empty:
            plt.subplot(1, 2, 2)
            plt.plot(eval_acc['epoch'], eval_acc['eval_accuracy'], label='Val Accuracy', color='#6BCB77', marker='o')
            plt.title('Accuracy Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f">>> Plot saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Plotting failed ({e})")

# ==========================================
# 3. 数据处理 (Data Processor)
# ==========================================
class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def clean_data(self, example):
        text = example['text']
        if text is None: return False
        if "此用户未填写评价内容" in text: return False
        if len(text.strip()) < 2: return False
        return True

    def unify_labels(self, example):
        label = example['label']
        if isinstance(label, str):
            label = label.lower()
            if label in ['negative', 'pos', '0']: return {'label': 0}
            elif label in ['neutral', 'neu', '1']: return {'label': 1}
            elif label in ['positive', 'neg', '2']: return {'label': 2}
        return {'label': int(label)}

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=Config.MAX_LENGTH)

    def get_dataset(self):
        print(">>> Loading Datasets...")
        # 指定 cache_dir 为 data 目录
        ds_clap = load_dataset("clapAI/MultiLingualSentiment", split="train", trust_remote_code=True, cache_dir=Config.DATA_DIR)
        ds_med = load_dataset("OpenModels/Chinese-Herbal-Medicine-Sentiment", split="train", trust_remote_code=True, cache_dir=Config.DATA_DIR)
        
        # 列对齐
        if 'review_text' in ds_med.column_names: ds_med = ds_med.rename_column('review_text', 'text')
        if 'sentiment_label' in ds_med.column_names: ds_med = ds_med.rename_column('sentiment_label', 'label')
        if 'language' in ds_clap.column_names: ds_clap = ds_clap.filter(lambda x: x['language'] == 'zh')
            
        common_cols = ['text', 'label']
        combined = concatenate_datasets([ds_clap.select_columns(common_cols), ds_med.select_columns(common_cols)])
        
        # 清洗与处理
        combined = combined.filter(self.clean_data).map(self.unify_labels)
        tokenized = combined.map(self.tokenize_function, batched=True, remove_columns=['text', 'label'])
        
        return tokenized.train_test_split(test_size=0.1)

# ==========================================
# 4. Metrics
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

# ==========================================
# 5. 主流程
# ==========================================
def main():
    print("=== Cloud Training Script ===")
    ensure_directories()
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Enabled: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU")

    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
    processor = DataProcessor(tokenizer)
    dataset = processor.get_dataset()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL, 
        num_labels=Config.NUM_LABELS,
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID
    )
    
    training_args = TrainingArguments(
        output_dir=Config.CHECKPOINT_DIR,    # Checkpoints 存放在这里
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        logging_dir=os.path.join(Config.RESULTS_DIR, 'logs'), # Logs 存放在 Results
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=Config.SAVE_STEPS,
        save_steps=Config.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print(">>> Starting Training...")
    trainer.train()
    
    # 保存最终模型到 checkpoints/final_model
    final_path = os.path.join(Config.CHECKPOINT_DIR, "final_model")
    print(f">>> Saving Final Model to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # 绘制曲线到 results/
    print(">>> Generating Plots...")
    plot_path = os.path.join(Config.RESULTS_DIR, "training_curves_cloud.png")
    plot_training_history(trainer.state.log_history, plot_path)
    
    print(">>> All Done!")

if __name__ == "__main__":
    main()
