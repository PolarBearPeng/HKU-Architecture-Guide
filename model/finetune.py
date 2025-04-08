"""
微调模型逻辑
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import sys
sys.path.append(".")
from config.config import Config
from data.dataset import HKUBuildingDataset, collate_fn
from model.model_utils import load_model_and_processor
from janus.models.processing_vlm import BatchedVLChatProcessorOutput
from janus.models import VLChatProcessor


class JanusTrainer(Trainer):
    """专为Janus模型设计的自定义训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """重写计算损失，专用于Janus模型，只使用成功的方法1"""
        
        # 确保所有输入在正确的设备上
        device = self.args.device
        dtype = torch.float16
        
        # 优化内存使用 - 将不必要的输入移到CPU
        tmp_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    tmp_inputs[k] = v.to(device=device, dtype=dtype)
                elif k in ["input_ids", "attention_mask", "labels", "images_seq_mask", "images_emb_mask"]:
                    tmp_inputs[k] = v.to(device)
                else:
                    # 其他不需要的张量保留在CPU
                    tmp_inputs[k] = v
            else:
                tmp_inputs[k] = v
        
        # 获取底层模型
        base_model = model.base_model.model if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') else model
        
        # 只使用方法1：Janus原始调用流程
        with torch.cuda.amp.autocast():  # 使用自动混合精度
            # 1. 准备输入嵌入
            inputs_embeds = base_model.prepare_inputs_embeds(
                pixel_values=tmp_inputs["pixel_values"],
                input_ids=tmp_inputs["input_ids"],
                images_seq_mask=tmp_inputs["images_seq_mask"],
                images_emb_mask=tmp_inputs["images_emb_mask"]
            )
            
            # 释放不再需要的内存
            del tmp_inputs["pixel_values"]
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # 2. 调用语言模型前向传递
            outputs = base_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=tmp_inputs["attention_mask"],
                labels=tmp_inputs["labels"],
                return_dict=True
            )
        
        # 获取损失
        loss = outputs.loss
        
        # 清理内存
        del inputs_embeds
        del tmp_inputs
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return (loss, outputs) if return_outputs else loss

def setup_peft_model(model):
    """
    设置模型用于参数高效微调(PEFT)
    
    Args:
        model: 原始模型
    
    Returns:
        peft_model: 准备好的PEFT模型
    """
    config = Config()
    
    # 检查并设置目标模块
    if not config.TARGET_MODULES:
        # 为Janus模型设置默认目标模块，如果配置中未指定
        # 这些模块名称可能需要根据您的模型架构进行调整
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    else:
        target_modules = config.TARGET_MODULES
    
    print(f"LoRA将应用于以下模块: {target_modules}")
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",  # 通常不对偏置应用LoRA
    )
    
    # 获取PEFT模型
    peft_model = get_peft_model(model, peft_config)
    
    return peft_model

def check_model_dtype(model):
    """检查模型各层的数据类型"""
    print("模型参数dtype检查:")
    
    # 存储不同类型的计数
    dtype_counts = {}
    
    # 遍历所有参数
    for name, param in model.named_parameters():
        dtype = param.dtype
        
        # 更新计数
        if dtype in dtype_counts:
            dtype_counts[dtype] += 1
        else:
            dtype_counts[dtype] = 1
            
        # 打印前几个参数的信息
        if len(dtype_counts) <= 5:
            print(f"参数 {name}: {dtype}")
    
    # 打印统计信息
    print("\n数据类型统计:")
    for dtype, count in dtype_counts.items():
        print(f"{dtype}: {count} 参数")
    
    # 检查主要dtype
    main_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0]
    print(f"\n主要数据类型: {main_dtype}")
    
    return dtype_counts


def check_specific_layers(model):
    """检查模型中特定层的数据类型"""
    print("\n检查特定层:")
    
    # 检查视觉模型部分
    if hasattr(model, 'vision_model'):
        vision_model = model.vision_model
        if hasattr(vision_model, 'patch_embed') and hasattr(vision_model.patch_embed, 'proj'):
            patch_embed = vision_model.patch_embed.proj
            print(f"视觉模型patch_embed.proj.weight dtype: {patch_embed.weight.dtype}")
            if hasattr(patch_embed, 'bias') and patch_embed.bias is not None:
                print(f"视觉模型patch_embed.proj.bias dtype: {patch_embed.bias.dtype}")
    
    # 如果是PEFT模型，检查基础模型
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model
            
        # 检查语言模型部分
        if hasattr(base_model, 'language_model'):
            lm = base_model.language_model
            # 检查第一个transformer块
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers') and len(lm.model.layers) > 0:
                layer0 = lm.model.layers[0]
                if hasattr(layer0, 'self_attn'):
                    attn = layer0.self_attn
                    if hasattr(attn, 'q_proj'):
                        print(f"语言模型self_attn.q_proj.weight dtype: {attn.q_proj.weight.dtype}")
                    if hasattr(attn, 'k_proj'):
                        print(f"语言模型self_attn.k_proj.weight dtype: {attn.k_proj.weight.dtype}")


def inspect_model_dtypes(model, prefix='', max_depth=3, current_depth=0):
    """递归检查模型中所有张量的数据类型"""
    if current_depth > max_depth:
        return {}
        
    results = {}
    
    # 检查当前模块的参数
    for name, param in model.named_parameters(recurse=False):
        full_name = f"{prefix}.{name}" if prefix else name
        results[full_name] = param.dtype
    
    # 递归检查子模块
    for child_name, child_module in model.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        child_results = inspect_model_dtypes(
            child_module, 
            prefix=child_prefix, 
            max_depth=max_depth, 
            current_depth=current_depth+1
        )
        results.update(child_results)
    
    return results

def finetune(model_path, output_dir, data_dir):
    """执行模型微调"""
    config = Config()
    model_path = model_path or config.MODEL_PATH
    output_dir = output_dir or config.OUTPUT_DIR
    data_dir = data_dir or config.TRAIN_DATA_DIR
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和处理器
    print(f"正在加载模型: {model_path}")
    model, processor = load_model_and_processor(model_path)

    # # 检查模型的数据类型
    # print("\n==== 检查模型数据类型 ====")
    # dtype_counts = check_model_dtype(model)
    # check_specific_layers(model)

    #  # 检查更详细的信息
    # print("\n==== 详细模型层检查 ====")
    # # 限制输出数量
    # detailed_dtypes = inspect_model_dtypes(model, max_depth=2)
    # for i, (name, dtype) in enumerate(detailed_dtypes.items()):
    #     print(f"{name}: {dtype}")
    #     if i >= 20:  # 只显示前20个
    #         print(f"... 还有 {len(detailed_dtypes) - 20} 个参数")
    #         break
    
    # # 根据检查结果决定数据类型
    # main_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0]
    # print(f"\n基于检查，模型主要使用的数据类型是: {main_dtype}")
    
    # 设置PEFT模型
    print("准备PEFT模型...")
    model = setup_peft_model(model)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    # 创建数据集
    print("正在创建数据集...")
    train_dataset = HKUBuildingDataset(data_dir, processor, split="train")
    eval_dataset = HKUBuildingDataset(data_dir, processor, split="val")
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # 降低批处理大小
        per_device_eval_batch_size=1,   # 降低评估批处理大小
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        save_strategy="epoch",
        save_steps=100,    # 减少保存频率
        eval_strategy="epoch",
        eval_steps=100,         # 减少评估频率
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        remove_unused_columns=False,
        fp16=True,  # 使用混合精度训练
        gradient_accumulation_steps=24,  # 增加梯度累积步数
        dataloader_drop_last=True,
        save_total_limit=2,  # 减少保存的检查点数量
        load_best_model_at_end=True,
        optim="adamw_torch",  # 使用PyTorch的AdamW优化器
    )
    # 创建自定义训练器
    trainer = JanusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print(f"保存最终模型到 {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    return model, processor