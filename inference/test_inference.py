# test_inference.py
import sys
sys.path.append(".")
from inference import HKUBuildingRecognizer
from PIL import Image

def test_model():
    """测试模型推理"""
    # 创建识别器
    print("加载模型...")
    recognizer = HKUBuildingRecognizer()
    
    # 测试图像
    test_image_path = "examples/IMG_5635.jpeg"
    print(f"使用测试图像: {test_image_path}")
    
    # 执行推理
    answer = recognizer.identify_building(test_image_path)
    
    print("\n======== 推理结果 ========")
    print(answer)
    print("==========================")

if __name__ == "__main__":
    test_model()