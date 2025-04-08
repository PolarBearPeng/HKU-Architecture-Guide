"""
主入口文件
"""

import argparse
import os
from scripts.prepare_data import prepare_data
from model.finetune import finetune
from inference.gradio_app import launch_app

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="香港大学建筑识别系统")
    parser.add_argument("--mode", type=str, choices=["prepare", "train", "inference"], 
                      required=True, help="运行模式：prepare(数据准备)、train(训练)或inference(推理)")
    parser.add_argument("--data_dir", type=str, default="./hku_buildings_raw", help="原始数据目录")
    parser.add_argument("--processed_dir", type=str, default="./hku_buildings_processed", help="处理后的数据目录")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="基础模型路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        print("准备数据...")
        prepare_data(args.data_dir, args.processed_dir)
    elif args.mode == "train":
        print("开始训练...")
        finetune(args.model_path, args.output_dir, args.processed_dir)
    elif args.mode == "inference":
        print("启动推理界面...")
        launch_app()
    else:
        print(f"不支持的模式: {args.mode}")

if __name__ == "__main__":
    main()