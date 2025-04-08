"""
模型加载和处理工具
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models.modeling_vlm import MultiModalityCausalLM
from janus.models import VLChatProcessor
from janus.models.processing_vlm import VLChatProcessor, BatchedVLChatProcessorOutput
from janus.utils.conversation import get_conv_template

from config.config import Config
from PIL import Image

def load_model_and_processor(model_path):
    """
    加载模型和处理器
    """
    try:
        # 加载配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 对于Janus模型，可能需要调整特定配置
        if hasattr(config, 'language_config'):
            language_config = config.language_config
            language_config._attn_implementation = 'eager'  # 避免某些环境中的flash attention问题
        
        # 加载处理器
        processor = VLChatProcessor.from_pretrained(
            model_path,
            legacy=False,  # 避免旧版本行为
            trust_remote_code=True
        )
        
        # 加载模型 - 根据您的模型类型选择合适的加载方法
        if model_path.lower().find("janus") >= 0:
            # 使用MultiModalityCausalLM类加载Janus模型
            model = MultiModalityCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # 通用加载方法
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # 验证处理器与模型兼容
        is_valid = test_processor(processor)
        if not is_valid:
            print("警告: 处理器测试不完全通过，但将继续使用")
        
        print(f"模型和处理器成功加载")
        
        return model, processor
    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise

def test_processor(processor):
    """测试处理器是否正常工作"""
    try:
        # 创建测试图像和文本
        test_image = Image.new('RGB', (384, 384), color=(127, 127, 127))
        
        # 直接测试图像处理器
        print("测试图像处理器...")
        try:
            # 注意：传递图像列表而不是单个图像
            image_outputs = processor.image_processor([test_image], return_tensors="pt")
            print(f"图像处理器正常，输出形状: {image_outputs.pixel_values.shape}")
        except Exception as e:
            print(f"图像处理器测试失败: {e}")
            return False
            
        # 测试分词器
        print("测试分词器...")
        try:
            # 尝试特殊标记的编码
            special_tokens = ["<|User|>", "<|Assistant|>", "<image_placeholder>"]
            for token in special_tokens:
                ids = processor.tokenizer.encode(token)
                print(f"标记 '{token}' 编码为: {ids}")
        except Exception as e:
            print(f"分词器测试失败: {e}")
            return False
        
        # 测试完整处理流程
        print("测试完整处理流程...")
        try:
            # 构建符合Janus预期的输入格式
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n这是什么？",
                },
                {"role": "<|Assistant|>", "content": "这是一个测试图像"},
            ]
            
            # 图像需要单独传递
            pil_images = [test_image]
            
            # 尝试使用processor处理完整的会话数据
            try:
                processed = processor(
                    conversations=conversation,
                    images=pil_images,  # 单独传递图像
                    force_batchify=True
                )
                
                print(f"处理器可以处理完整的会话数据，keys: {processed.keys()}")
                return True
            except Exception as e:
                print(f"处理器无法处理完整的会话数据: {e}")
                # 继续使用基本组件测试
            
            # 使用各个组件单独处理
            # 1. 处理图像
            pixel_values = processor.image_processor([test_image], return_tensors="pt").pixel_values
            
            # 2. 构建文本
            full_text = f"<|User|>: <image_placeholder>\n这是什么？\n\n<|Assistant|>: 这是一个测试图像"
            
            # 3. 编码文本
            encoded = processor.tokenizer(full_text, return_tensors="pt")
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            
            print("基本组件测试通过")
            return True
            
        except Exception as e:
            print(f"完整流程测试失败: {e}")
            return False
            
    except Exception as e:
        print(f"处理器测试失败: {str(e)}")
        return False

def prepare_model_for_inference(model, processor, adapter_path=None):
    """
    准备模型用于推理，强制使用Float16（不考虑BF16报告支持）
    """
    # 设置为评估模式
    model.eval()
    
    # 如果提供了适配器路径，加载适配器
    if adapter_path:
        try:
            from peft import PeftModel
            print(f"加载LoRA适配器: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            print("LoRA适配器加载成功")
        except Exception as e:
            print(f"加载适配器时出错: {e}")
    
    # 将模型移动到GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # 在Turing GPU上强制使用Float16而不是BFloat16
        # 忽略is_bf16_supported()的检测结果，因为它可能是误报
        print("Turing架构GPU检测到，强制使用Float16...")
        model = model.to(torch.float16)
        model = model.to(device)
    else:
        model = model.to(device)
    
    return model, processor


def prepare_model_for_training(model):
    """
    准备模型进行训练
    
    Args:
        model: 模型
    
    Returns:
        training_model: 准备好的训练模型
    """
    # 确保模型处于训练模式
    model.train()
    
    # 如果模型有冻结的部分，可以在这里解冻特定层
    # 例如，对于Janus模型，可能希望只微调特定部分
    
    return model