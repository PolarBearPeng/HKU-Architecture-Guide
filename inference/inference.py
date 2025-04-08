"""
推理逻辑
"""

import torch
import torch.nn as nn
from PIL import Image
import os
import sys
sys.path.append(".")
from config.config import Config
from model.model_utils import load_model_and_processor

def force_convert_model_to_fp16(model):
    """
    强制将模型的所有BFloat16参数转换为Float16
    """
    print("强制转换模型所有BFloat16参数为Float16...")
    
    # 存储转换的参数数量
    converted = 0
    total = 0
    
    # 递归处理所有模块
    for module in model.modules():
        # 处理所有参数
        for param_name, param in module.named_parameters(recurse=False):
            total += 1
            if param.dtype == torch.bfloat16:
                converted += 1
                # 创建新的Float16参数并替换旧参数
                module._parameters[param_name] = nn.Parameter(
                    param.data.to(torch.float16), 
                    requires_grad=param.requires_grad
                )
        
        # 处理所有缓冲区
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if hasattr(buffer, 'dtype') and buffer.dtype == torch.bfloat16:
                module._buffers[buffer_name] = buffer.to(torch.float16)
    
    print(f"已转换 {converted}/{total} 个参数从BFloat16到Float16")
    return model

def prepare_model_for_inference(model, processor, adapter_path=None):
    """
    准备模型用于推理，强制使用Float16
    
    Args:
        model: 基础模型
        processor: 处理器  
        adapter_path: LoRA适配器路径，如果提供则加载适配器
        
    Returns:
        model: 准备好的模型
        processor: 处理器
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
        print("GPU检测到，强制使用Float16...")
        model = model.to(torch.float16)
        model = model.to(device)
        
        # 额外检查并强制转换任何可能仍为BFloat16的参数
        model = force_convert_model_to_fp16(model)
    else:
        model = model.to(device)
    
    return model, processor

class HKUBuildingRecognizer:
    def __init__(self, model_path=None, adapter_path=None):
        """
        初始化HKU建筑识别器
        
        Args:
            model_path: 基础模型路径，默认使用配置中的路径
            adapter_path: 适配器路径，默认使用输出目录
        """
        self.config = Config()
        
        # 设置模型路径和适配器路径
        model_path = model_path or self.config.MODEL_PATH
        adapter_path = adapter_path or self.config.OUTPUT_DIR
        
        print(f"加载基础模型: {model_path}")
        # 加载模型和处理器
        self.model, self.processor = load_model_and_processor(model_path)
        
        # 准备模型，强制使用FP16
        self.model, self.processor = prepare_model_for_inference(
            self.model, self.processor, adapter_path
        )
        
        # 获取设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
    @torch.inference_mode()
    def identify_building(self, image, question=None):
        """
        识别HKU建筑
        
        Args:
            image: PIL图像或图像路径
            question: 自定义问题，默认使用标准问题
            
        Returns:
            answer: 模型回答
        """
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 如果image是路径，加载图像
        if isinstance(image, str):
            print(f"加载图像: {image}")
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            # 如果是numpy数组，转换为PIL图像
            image = Image.fromarray(image).convert("RGB")
            
        # 默认问题
        if question is None:
            question = "这是香港大学的哪个建筑？请详细介绍它的特点、历史和功能。"
            
        # 构建对话
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
            }
        ]
        
        # 处理输入 - 参考源代码的方式处理
        print("处理输入...")
        pil_images = [image]  # 直接使用PIL图像
        prepare_inputs = self.processor(
            conversations=conversation, 
            images=pil_images,
            force_batchify=True
        )
        
        # 将处理后的输入移动到设备并转换为float16
        # 参考源代码，使用 .to() 方法而不是遍历字典
        prepare_inputs = prepare_inputs.to(self.device)
        
        # 额外确保pixel_values是float16
        if hasattr(prepare_inputs, 'pixel_values'):
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)
        
        # 获取底层模型，针对PEFT模型结构
        base_model = self.model
        if hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                base_model = self.model.base_model.model
            else:
                base_model = self.model.base_model
        
        try:
            print("使用Float16精度开始推理...")
            # 使用自动混合精度并强制Float16
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                # 准备模型输入 - 参考源代码使用**运算符展开参数
                inputs_embeds = base_model.prepare_inputs_embeds(**prepare_inputs)
                
                # 生成回答 - 参考源代码的参数传递方式
                outputs = base_model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    do_sample=self.config.DO_SAMPLE if hasattr(self.config, 'DO_SAMPLE') else (self.config.TEMPERATURE > 0),
                    use_cache=True,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                )
        except Exception as e:
            print(f"生成过程中出错: {e}")
            print("尝试备用方法...")
            
            # 再次检查并强制所有输入为Float16
            print("强制转换所有输入为Float16...")
            for attr_name in dir(prepare_inputs):
                if attr_name.startswith('_'):
                    continue
                    
                attr = getattr(prepare_inputs, attr_name)
                if isinstance(attr, torch.Tensor) and attr.dtype == torch.bfloat16:
                    print(f"转换输入 {attr_name} 从 {attr.dtype} 到 float16")
                    setattr(prepare_inputs, attr_name, attr.to(torch.float16))
            
            # 再次尝试，不使用autocast
            print("再次尝试生成，不使用autocast...")
            
            # 准备模型输入
            inputs_embeds = base_model.prepare_inputs_embeds(**prepare_inputs)
            
            # 生成回答
            outputs = base_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=self.config.DO_SAMPLE if hasattr(self.config, 'DO_SAMPLE') else (self.config.TEMPERATURE > 0),
                use_cache=True,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
            )
        
        # 解码回答
        answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return answer

def test_model():
    """测试模型推理"""
    # 创建识别器
    print("加载模型...")
    recognizer = HKUBuildingRecognizer()
    
    # 测试图像
    test_image_path = "examples/IMG_5635.jpeg"  # 使用您项目中的测试图像路径
    print(f"使用测试图像: {test_image_path}")
    
    # 执行推理
    answer = recognizer.identify_building(test_image_path)
    
    print("\n======== 推理结果 ========")
    print(answer)
    print("==========================")

if __name__ == "__main__":
    test_model()