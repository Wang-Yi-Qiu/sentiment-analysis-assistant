import os
import sys
from transformers import AutoTokenizer
from config import Config
from dataset import DataProcessor

def main():
    print("⏳ 开始下载并处理数据...")
    
    # 1. 确保 data 目录存在
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)
        
    # 2. 初始化流程
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
    processor = DataProcessor(tokenizer)
    
    # 3. 获取处理后的数据 (get_processed_dataset 内部已经有加载逻辑)
    # 注意：我们这里为了保存原始数据，可能需要调用 load_clap_data 和 load_medical_data
    # 但 DataProcessor.get_processed_dataset 返回的是 encode 后的数据。
    # 用户可能想要的是 Raw Data 或者 Processed Data。
    # 这里我们保存 Processed Data (Ready for Training) 到磁盘
    
    dataset = processor.get_processed_dataset()
    
    save_path = os.path.join(Config.DATA_DIR, "processed_dataset")
    print(f"💾 正在保存处理后的数据集到: {save_path}")
    dataset.save_to_disk(save_path)
    
    print("✅ 数据保存完成！")
    print(f"   Train set size: {len(dataset['train'])}")
    print(f"   Test set size: {len(dataset['test'])}")
    print("   下次加载可直接使用: from datasets import load_from_disk")

if __name__ == "__main__":
    main()
