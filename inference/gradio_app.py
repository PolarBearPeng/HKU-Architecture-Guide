"""
Gradio应用界面
"""

import gradio as gr
from PIL import Image
import sys
sys.path.append(".")
from config.config import Config
from inference.inference import HKUBuildingRecognizer

def create_gradio_app():
    """创建Gradio应用界面"""
    config = Config()
    
    # 创建识别器
    print("正在加载模型，请稍候...")
    recognizer = HKUBuildingRecognizer()
    print("模型加载完成，准备就绪")
    
    # 定义接口函数
    def process_image(image, question=None):
        if question is None or question.strip() == "":
            question = "这是香港大学的哪个建筑？请详细介绍它的特点、历史和功能。"
        
        try:
            answer = recognizer.identify_building(image, question)
            return answer
        except Exception as e:
            return f"处理图像时出错: {str(e)}"
    
    # 创建Gradio界面
    with gr.Blocks(title="香港大学建筑识别") as app:
        gr.Markdown("# 香港大学建筑识别系统")
        gr.Markdown("上传香港大学建筑图片，系统将识别并提供相关信息。")
        
        with gr.Row():
            with gr.Column():
                # 输入部分
                image_input = gr.Image(label="上传建筑图片", type="pil")
                question_input = gr.Textbox(
                    label="问题 (可选)", 
                    placeholder="这是香港大学的哪个建筑？请详细介绍它的特点、历史和功能。"
                )
                identify_button = gr.Button("识别建筑")
            
            with gr.Column():
                # 输出部分
                building_info = gr.Textbox(label="建筑信息", lines=15)
        
        # 示例
        gr.Examples(
            examples=[
                ["examples/Main_Building_1.jpg", "这是什么建筑？"],
                ["examples/Main_Library_G_1.jpg", "请介绍这个图书馆。"],
                ["examples/RAYSON_HUANG_1.jpg", "这个建筑有什么历史？"]
            ],
            inputs=[image_input, question_input],
        )
        
        # 设置点击事件
        identify_button.click(
            fn=process_image,
            inputs=[image_input, question_input],
            outputs=building_info
        )
        
    return app

def launch_app():
    """启动Gradio应用"""
    app = create_gradio_app()
    app.launch(share=True)

if __name__ == "__main__":
    launch_app()