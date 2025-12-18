import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from .config import Config
from .dataset import DataProcessor
from .metrics import compute_metrics
from .visualization import plot_training_history

def main():
    # 0. 设备检测 (针对 Mac Mini 优化)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Mac Silicon Acceleration)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    # 1. 初始化 Tokenizer
    print(f"Loading tokenizer from {Config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)

    # 2. 准备数据
    print("Preparing datasets...")
    processor = DataProcessor(tokenizer)
    # 使用 Config.DATA_DIR 确保数据下载到正确位置
    # 使用多进程加速数据处理
    num_proc = max(1, os.cpu_count() - 1)
    # 注意: get_processed_dataset 内部需要实现真实的加载逻辑，这里假设 dataset.py 已经完善
    # 如果 dataset.py 中有模拟逻辑，实际运行时需要联网下载数据
    dataset = processor.get_processed_dataset(cache_dir=Config.DATA_DIR, num_proc=num_proc)
    
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(eval_dataset)} samples.")

    # 3. 加载模型
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL, 
        num_labels=Config.NUM_LABELS,
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID
    )
    model.to(device)

    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=Config.RESULTS_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        logging_dir=os.path.join(Config.RESULTS_DIR, 'logs'),
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_steps=Config.SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Mac MPS 特定优化:
        # huggingface trainer 默认支持 mps，如果不手动指定 no_cuda，它通常会自动检测
        # 但为了保险，我们可以尽量让 trainer 自己处理，或者显式use_mps_device (老版本不仅用)
        # 最新版 transformers 会自动通过 accelerate 处理 device
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. 开始训练
    print("Starting training...")
    trainer.train()

    # 7. 保存最终模型
    print(f"Saving model to {Config.CHECKPOINT_DIR}...")
    trainer.save_model(Config.CHECKPOINT_DIR)
    tokenizer.save_pretrained(Config.CHECKPOINT_DIR)
    
    # 8. 绘制训练曲线
    print("Generating training plots...")
    plot_save_path = os.path.join(Config.RESULTS_DIR, 'training_curves.png')
    plot_training_history(trainer.state.log_history, save_path=plot_save_path)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
