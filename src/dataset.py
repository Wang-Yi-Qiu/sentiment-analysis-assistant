import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from .config import Config

class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_clap_data(self):
        """
        加载 clapAI/MultiLingualSentiment 数据集的中文部分
        """
        print("Loading clapAI/MultiLingualSentiment (zh)...")
        try:
            # 假设数据集结构支持 language='zh' 筛选，或者我们加载后筛选
            # 注意：实际使用时可能需要根据具体 Hugging Face dataset 的 config name 调整
            ds = load_dataset("clapAI/MultiLingualSentiment", "zh", split="train", trust_remote_code=True)
        except Exception:
            # Fallback if specific config not found, load all and filter (demo logic)
            print("Warning: Could not load 'zh' specific config, attempting to load generic...")
            ds = load_dataset("clapAI/MultiLingualSentiment", split="train", trust_remote_code=True)
            ds = ds.filter(lambda x: x['language'] == 'zh')
        
        # 映射标签 (假设原标签格式需要调整，这里做通用处理)
        # 假设原数据集 label已经是 0,1,2 或者需要 map
        # 这里为了演示，我们假设它已经是标准格式，或者我们需要查看数据结构
        # 为保证稳健性，我们在 map_function 中处理
        return ds

    def load_medical_data(self):
        """
        加载 OpenModels/Chinese-Herbal-Medicine-Sentiment 垂直领域数据
        """
        print("Loading OpenModels/Chinese-Herbal-Medicine-Sentiment...")
        ds = load_dataset("OpenModels/Chinese-Herbal-Medicine-Sentiment", split="train", trust_remote_code=True)
        return ds

    def clean_data(self, examples):
        """
        数据清洗逻辑
        """
        text = examples['text']
        
        # 1. 剔除“默认好评”噪音
        if "此用户未填写评价内容" in text:
            return False
            
        # 简单长度过滤，太短的可能无意义
        if len(text.strip()) < 2:
            return False
            
        return True

    def unify_labels(self, example):
        """
        统一标签为: 0 (Negative), 1 (Neutral), 2 (Positive)
        """
        label = example['label']
        
        # 根据数据集实际情况调整映射逻辑
        # 这里假设传入的数据集 label 可能是 string 或 int
        # 这是一个示例映射，实际运行时需根据 print(ds.features) 确认
        if isinstance(label, str):
            label = label.lower()
            if label in ['negative', 'pos', '0']: # 示例
                return {'labels': 0}
            elif label in ['neutral', 'neu', '1']:
                return {'labels': 1}
            elif label in ['positive', 'neg', '2']:
                return {'labels': 2}
        
        # 如果已经是 int，确保在 0-2 之间
        return {'labels': int(label)}

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH
        )

    def get_processed_dataset(self, cache_dir=None, num_proc=1):
        # 默认使用 Config.DATA_DIR 作为缓存目录
        if cache_dir is None:
            cache_dir = Config.DATA_DIR

        # 0. 尝试从本地加载已处理的数据
        processed_path = os.path.join(cache_dir, "processed_dataset")
        if os.path.exists(processed_path):
            print(f"Loading processed dataset from {processed_path}...")
            return load_from_disk(processed_path)

        # 1. 加载数据
        ds_clap = self.load_clap_data()
        ds_med = self.load_medical_data()
        
        # 2. 统一列名 (确保都有 'text' 和 'label')
        # OpenModels keys: ['username', 'user_id', 'review_text', 'review_time', 'rating', 'product_id', 'sentiment_label', 'source_file']
        if 'review_text' in ds_med.column_names:
            ds_med = ds_med.rename_column('review_text', 'text')
        if 'sentiment_label' in ds_med.column_names:
            ds_med = ds_med.rename_column('sentiment_label', 'label')
        
        # 3. 数据清洗
        print("Cleaning datasets...")
        ds_med = ds_med.filter(self.clean_data)
        ds_clap = ds_clap.filter(self.clean_data)
        
        # 4. 合并
        # 确保 features 一致
        common_cols = ['text', 'label']
        ds_clap = ds_clap.select_columns(common_cols)
        ds_med = ds_med.select_columns(common_cols)
        
        combined_ds = concatenate_datasets([ds_clap, ds_med])
        
        # 5.标签处理 & Tokenization
        # transform label -> labels
        combined_ds = combined_ds.map(self.unify_labels, remove_columns=['label'])
        
        # tokenize and remove text
        tokenized_ds = combined_ds.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
        
        # 划分训练集和验证集
        split_ds = tokenized_ds.train_test_split(test_size=0.1)
        
        return split_ds
