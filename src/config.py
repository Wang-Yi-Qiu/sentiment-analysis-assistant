import os

class Config:
    # 路径配置
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    
    # 模型配置
    BASE_MODEL = "google-bert/bert-base-chinese"
    NUM_LABELS = 3
    MAX_LENGTH = 128
    
    # 训练配置
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 100
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    
    # 标签映射
    LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
    ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
