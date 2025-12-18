import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import Config

class SentimentPredictor:
    def __init__(self, model_path=None):
        # 1. 如果未指定路径，尝试自动寻找最新的模型
        if model_path is None:
            # 优先检查 Config.CHECKPOINT_DIR (如果训练完成，final_model 会在这里)
            if os.path.exists(os.path.join(Config.CHECKPOINT_DIR, "config.json")):
                model_path = Config.CHECKPOINT_DIR
            else:
                # 如果没有 final_model，尝试寻找最新的 checkpoint (在 results 目录)
                import glob
                ckpt_list = glob.glob(os.path.join(Config.RESULTS_DIR, "checkpoint-*"))
                if ckpt_list:
                    # 按修改时间排序，取最新的
                    ckpt_list.sort(key=os.path.getmtime)
                    model_path = ckpt_list[-1]
                    print(f"Using latest checkpoint found: {model_path}")
                else:
                    # 只有在真的找不到时才回退
                    model_path = Config.CHECKPOINT_DIR

        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except OSError:
            print(f"Warning: Model not found at {model_path}. Loading base model for demo purpose.")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.BASE_MODEL, num_labels=Config.NUM_LABELS)
            
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=Config.MAX_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            score = probabilities[0][prediction].item()

        label = Config.ID2LABEL.get(prediction, "unknown")
        return {
            "text": text,
            "sentiment": label,
            "confidence": f"{score:.4f}"
        }

if __name__ == "__main__":
    # Demo
    predictor = SentimentPredictor()
    test_texts = [
        "这家店的快递太慢了，而且东西味道很奇怪。",
        "非常不错，包装很精美，下次还会来买。",
        "感觉一般般吧，没有想象中那么好，但也还可以。"
    ]
    
    print("\nPredicting...")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
        print("-" * 30)
