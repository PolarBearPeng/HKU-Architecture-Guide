# config/config.py 修改
"""
配置参数
"""
from config.building_info import BUILDING_INFO
import os

# 在Config类中添加WEIGHT_DECAY属性
class Config:
    def __init__(self):
        # 路径配置
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 模型配置
        self.MODEL_PATH = "deepseek-ai/Janus-Pro-7B"  # 基础模型路径
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "output")  # 微调输出目录
        self.ADAPTER_PATH = self.OUTPUT_DIR  # LoRA适配器路径，默认与OUTPUT_DIR相同
        
        # 训练配置
        self.BATCH_SIZE = 1
        self.LEARNING_RATE = 2e-5
        self.EPOCHS = 3
        self.WEIGHT_DECAY = 0.01
        self.SAVE_STEPS = 100
        self.USE_FP16 = True
        self.SAVE_TOTAL_LIMIT = 2
        self.GRADIENT_ACCUMULATION_STEPS = 4
        
        # LoRA配置
        self.LORA_R = 16
        self.LORA_ALPHA = 32
        self.LORA_DROPOUT = 0.05
        self.TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # 数据集配置
        self.TRAIN_DATA_DIR = os.path.join(self.BASE_DIR, "hku_buildings_dataset")
        self.IMAGE_SIZE = (384, 384)  # 模型期望的输入尺寸
        self.VALIDATION_SPLIT = 0.1
        
        # 推理配置
        self.MAX_NEW_TOKENS = 512           # 生成的最大标记数
        self.TOP_P = 0.95                  # 生成时的top-p采样参数
        self.TEMPERATURE = 0.1             # 生成时的温度参数
        self.DO_SAMPLE = (self.TEMPERATURE > 0)  # 是否使用采样
        
        # 资源配置
        self.USE_CUDA = True               # 是否使用CUDA
        self.LOW_MEMORY_MODE = False       # 低内存模式（牺牲速度节省内存）
        
        # 界面配置
        self.EXAMPLES_DIR = os.path.join(self.BASE_DIR, "examples")  # 示例图片目录
        
        # 建筑描述模板
        self.BUILDING_DESCRIPTION_TEMPLATE = """
        这是香港大学的{building_name}。

        {building_description}

        建筑特点：{building_features}
        
        历史背景：{building_history}
        
        使用功能：{building_functions}
        """
        
        # 建筑信息字典，从单独文件导入
        self.BUILDING_INFO = BUILDING_INFO