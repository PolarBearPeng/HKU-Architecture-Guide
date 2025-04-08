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
    parser = argparse.ArgumentParser(description="HKU Architecture Guide")
    parser.add_argument("--mode", type=str, choices=["prepare", "train", "inference"], 
                      required=True, help="Mode：prepare data、training or inference")
    parser.add_argument("--data_dir", type=str, default="./hku_buildings_raw", help="raw data directory")
    parser.add_argument("--processed_dir", type=str, default="./hku_buildings_processed", help="processed data directory")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="base model path")
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        print("Preparing dataset...")
        prepare_data(args.data_dir, args.processed_dir)
    elif args.mode == "train":
        print("Begin training...")
        finetune(args.model_path, args.output_dir, args.processed_dir)
    elif args.mode == "inference":
        print("Lanuch inference interface...")
        launch_app()
    else:
        print(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()