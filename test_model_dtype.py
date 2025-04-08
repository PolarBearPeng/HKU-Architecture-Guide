# test_model_dtype.py
import torch
import sys
sys.path.append(".")
from model.model_utils import load_model_and_processor, prepare_model_for_inference

def check_model_dtypes(model, name="模型"):
    """检查模型中所有参数的数据类型"""
    print(f"\n检查{name}的数据类型:")
    dtype_count = {}
    
    for name, param in model.named_parameters():
        if param.dtype not in dtype_count:
            dtype_count[param.dtype] = 0
        dtype_count[param.dtype] += 1
    
    for dtype, count in dtype_count.items():
        print(f"  {dtype}: {count}个参数")
    
    # 检查一些特定的关键参数
    print("\n检查关键参数:")
    for name, param in model.named_parameters():
        if 'bias' in name or 'weight' in name:
            if param.numel() > 0:  # 只检查非空参数
                print(f"  {name}: {param.dtype}")
                break

def test_model_preparation():
    """测试模型准备过程并检查数据类型"""
    print("加载基础模型...")
    from config.config import Config
    config = Config()
    
    base_model, processor = load_model_and_processor(config.MODEL_PATH)
    check_model_dtypes(base_model, "基础模型")
    
    print("\n应用模型准备...")
    model, _ = prepare_model_for_inference(base_model, processor, config.ADAPTER_PATH)
    check_model_dtypes(model, "准备后的模型")
    
    print("\n准备一个简单的前向传递...")
    # 创建一个简单的测试输入
    img = torch.rand(1, 3, 384, 384)
    if torch.cuda.is_available():
        img = img.to('cuda')
        img = img.to(torch.float16)
    
    # 获取底层模型
    base_model = model
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model'):
            base_model = model.base_model.model
        else:
            base_model = model.base_model
    
    print(f"测试输入的数据类型: {img.dtype}")
    
    try:
        # 检查视觉模型的一些关键模块的数据类型
        if hasattr(base_model, 'vision_model'):
            check_model_dtypes(base_model.vision_model, "视觉模型")
            
            # 尝试运行视觉模型的一部分
            with torch.no_grad():
                print("\n尝试视觉模型前向传递...")
                try:
                    img_features = base_model.vision_model(img)
                    print(f"成功! 输出形状: {img_features.shape}, 类型: {img_features.dtype}")
                except Exception as e:
                    print(f"视觉模型前向传递失败: {e}")
    except Exception as e:
        print(f"检查模型时出错: {e}")

if __name__ == "__main__":
    test_model_preparation()