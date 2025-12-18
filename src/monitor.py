import os
import time
import json
import glob
import pandas as pd
from datetime import datetime

def get_latest_checkpoint(checkpoint_dir):
    # 查找所有 checkpoint-XXX 文件夹
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # 按修改时间排序，最新的在最后
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]

def read_metrics(checkpoint_path):
    state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_file):
        return None
    
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        return data.get("log_history", [])
    except:
        return None

def monitor(checkpoint_dir="checkpoints"):
    print(f"👀 开始监视训练目录: {checkpoint_dir}")
    print("按 Ctrl+C 退出监视")
    print("-" * 50)
    
    last_step = -1
    
    while True:
        latest_ckpt = get_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            folder_name = os.path.basename(latest_ckpt)
            logs = read_metrics(latest_ckpt)
            
            if logs:
                # 找到最新的 eval 记录
                latest_log = logs[-1]
                current_step = latest_log.get('step', 0)
                
                # 如果有更新
                if current_step != last_step:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # 尝试寻找验证集指标 (eval_accuracy 等)
                    # log_history 混杂了 training loss 和 eval metrics
                    # 我们倒序找最近的一个包含 eval_accuracy 的记录
                    eval_record = None
                    train_record = None
                    
                    for log in reversed(logs):
                        if 'eval_accuracy' in log and eval_record is None:
                            eval_record = log
                        if 'loss' in log and train_record is None:
                            train_record = log
                        if eval_record and train_record:
                            break
                    
                    print(f"[{timestamp}] 最新检查点: {folder_name}")
                    if train_record:
                        print(f"   📉 Training Loss: {train_record.get('loss', 'N/A'):.4f} (Epoch {train_record.get('epoch', 'N/A'):.2f})")
                    if eval_record:
                        print(f"   ✅ Eval Accuracy: {eval_record.get('eval_accuracy', 'N/A'):.4f}")
                        print(f"   ✅ Eval F1 Score: {eval_record.get('eval_f1', 'N/A'):.4f}")
                    print("-" * 50)
                    
                    last_step = current_step
        
        time.sleep(10) # 每10秒检查一次

if __name__ == "__main__":
    # 尝试从 config 读取路径，如果失败则使用默认
    try:
        from config import Config
        ckpt_dir = Config.CHECKPOINT_DIR
    except:
        ckpt_dir = "checkpoints"
        
    monitor(ckpt_dir)
