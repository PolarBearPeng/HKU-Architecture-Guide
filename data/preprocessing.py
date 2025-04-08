"""
图像预处理工具
"""

from PIL import Image
import torchvision.transforms as transforms
from config.config import Config

def preprocess_image(image, target_size=None, max_size=1024):
    """
    预处理图像用于模型输入
    
    Args:
        image: PIL图像
        target_size: 目标尺寸(高, 宽)，如果为None则使用配置值
        max_size: 图像的最大尺寸，用于调整大图像
    
    Returns:
        处理后的PIL图像
    """
    config = Config()
    
    # 确保输入是PIL Image对象
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except Exception as e:
            print(f"无法将输入转换为PIL Image: {e}")
            raise TypeError("输入必须是PIL Image对象或可转换为PIL Image的数组")
    
    # 调整大小以提高处理速度
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    if target_size is None:
        target_size = config.IMAGE_SIZE
    
    # 创建预处理流程
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return preprocess(image)


def augment_training_image(image):
    """
    为训练增加数据增强
    
    Args:
        image: PIL图像
    
    Returns:
        增强后的PIL图像
    """
    config = Config()
    
    # 创建数据增强流程
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Resize(config.IMAGE_SIZE)
    ])
    
    return augment(image)