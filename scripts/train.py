"""
训练启动脚本
"""

import argparse
import sys
sys.path.append(".")
from model.finetune import finetune
from config.config import Config

def main():
    """训练模型的主函数"""
    config = Config()
    
    parser = argparse.ArgumentParser(description="微调JanusPro模型识别香港大学建筑")
    parser.add_argument("--model_path", type=str, default=config.MODEL_PATH, help="基础模型路径")
    parser.add_argument("--data_dir", type=str, default=config.TRAIN_DATA_DIR, help="训练数据目录")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="学习率")
    
    args = parser.parse_args()
    
    # 更新配置
    config.MODEL_PATH = args.model_path
    config.TRAIN_DATA_DIR = args.data_dir
    config.OUTPUT_DIR = args.output_dir
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    # 开始微调
    print("开始微调模型...")
    finetune(args.model_path, args.output_dir, args.data_dir)
    print(f"微调完成，模型已保存至 {args.output_dir}")

if __name__ == "__main__":
    main()