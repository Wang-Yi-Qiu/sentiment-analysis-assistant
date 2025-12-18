
import gradio as gr
import sys
import os

# 将项目根目录加入路径，以便能以包的形式导入 src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.predict import SentimentPredictor

# 初始化预测器
try:
    predictor = SentimentPredictor()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败 (可能需要先运行训练): {e}")
    # Fallback mock for demo UI preview
    class MockPredictor:
        def predict(self, text):
            return {'sentiment': 'neutral', 'confidence': 0.0}
    predictor = MockPredictor()

def analyze_sentiment(text):
    if not text.strip():
        return "请输入只有效的文本。", "N/A"
        
    result = predictor.predict(text)
    
    # 转换为友好显示
    label_map = {
        'positive': '😊 积极 (Positive)', 
        'neutral': '😐 中性 (Neutral)', 
        'negative': '😡 消极 (Negative)'
    }
    
    friendly_label = label_map.get(result['sentiment'], result['sentiment'])
    confidence_score = float(result['confidence'])
    
    # 返回: 
    # 1. 标签概率字典 (用于 Label 组件)
    # 2. 文本详细结果
    return {
        '积极': confidence_score if result['sentiment'] == 'positive' else 0.0,
        '中性': confidence_score if result['sentiment'] == 'neutral' else 0.0,
        '消极': confidence_score if result['sentiment'] == 'negative' else 0.0
    }, f"预测结果: {friendly_label}\n置信度: {confidence_score:.4f}"

# 构建 Gradio 界面
with gr.Blocks(title="中文情感分析演示") as demo:
    gr.Markdown("# 🎭 中文情感分析 AI")
    gr.Markdown("输入一段中文文本，模型将判断其情感倾向 (积极/消极/中性)。")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="输入文本", 
                placeholder="例如：这家餐厅真的太好吃了，强烈推荐！",
                lines=5
            )
            analyze_btn = gr.Button("开始分析", variant="primary")
            
        with gr.Column():
            res_label = gr.Label(label="情感概率", num_top_classes=3)
            res_text = gr.Textbox(label="详细结果")
            
    # 示例
    gr.Examples(
        examples=[
            ["这就去把差评改了！"],
            ["物流太慢了，而且东西也是坏的，非常失望。"],
            ["如果不看价格的话，确实是不错的产品。"],
            ["今天天气真不错。"]
        ],
        inputs=input_text
    )
    
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=input_text,
        outputs=[res_label, res_text]
    )

if __name__ == "__main__":
    # Gradio 6.0+ 建议将 theme 放在 launch 中，或者 Blocks 中（警告说 moved to launch? 通常是 Block 构造参数）
    # 但实际 Gradio 版本不同可能有差异。
    # 根据用户报错 "The parameters have been moved ... to the launch() method ...: theme"
    # 我们听从报错建议。
    demo.launch(theme=gr.themes.Soft())
