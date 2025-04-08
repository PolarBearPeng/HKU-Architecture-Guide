"""
数据集加载和处理
"""

import os
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import sys
sys.path.append(".")
from config.config import Config


class HKUBuildingDataset(Dataset):
    def __init__(self, data_dir, processor, split="train"):
        self.data_dir = data_dir
        self.processor = processor
        self.config = Config()
        self.split = split
        
        # 获取所有建筑文件夹
        self.building_folders = [f for f in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, f))]
        
        # 预处理数据：为每个建筑创建样本
        self.samples = self._prepare_samples()
        
        # 验证样本数量
        if len(self.samples) == 0:
            raise ValueError(f"没有找到任何有效样本! 请检查数据目录: {data_dir}")
        
        print(f"找到 {len(self.samples)} 个有效样本")
        
        # 分割训练和验证集
        random.shuffle(self.samples)
        if split == "train":
            self.samples = self.samples[:int(len(self.samples) * (1 - self.config.VALIDATION_SPLIT))]
            print(f"使用 {len(self.samples)} 个样本进行训练")
        else:
            self.samples = self.samples[int(len(self.samples) * (1 - self.config.VALIDATION_SPLIT)):]
            print(f"使用 {len(self.samples)} 个样本进行验证")
    
    def _prepare_samples(self):
        """准备训练样本"""
        samples = []
        for building in self.building_folders:
            building_path = os.path.join(self.data_dir, building)
            image_files = [f for f in os.listdir(building_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 获取建筑信息
            building_info = self.config.BUILDING_INFO.get(building, {})
            if not building_info:
                print(f"警告: 找不到建筑信息: {building}，跳过")
                continue
                
            # 构建描述
            building_description = self.config.BUILDING_DESCRIPTION_TEMPLATE.format(
                building_name=building_info.get("name", building),
                building_description=building_info.get("description", ""),
                building_features=building_info.get("features", ""),
                building_history=building_info.get("history", ""),
                building_functions=building_info.get("functions", "")
            )
            
            # 为每个图像创建一个样本
            for img_file in image_files:
                img_path = os.path.join(building_path, img_file)
                
                # 验证图像文件是否可访问
                if not os.path.exists(img_path):
                    print(f"警告: 图像文件不存在: {img_path}，跳过")
                    continue
                    
                # 测试是否可以打开图像
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                        if img_width < 10 or img_height < 10:
                            print(f"警告: 图像太小: {img_path}，跳过")
                            continue
                except Exception as e:
                    print(f"警告: 无法打开图像: {img_path}，错误: {e}，跳过")
                    continue
                    
                # 创建一些问题变体
                questions = [
                    f"这是什么建筑？请详细介绍。",
                    f"请识别这张图片中的建筑并给出详细信息。",
                    f"这是香港大学的哪栋建筑？请提供相关信息。"
                ]
                
                # 随机选择一个问题
                question = random.choice(questions)
                
                samples.append({
                    "image_path": img_path,
                    "question": question,
                    "answer": building_description,
                    "building_name": building_info.get("name", building)
                })
        
        return samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 获取样本
        sample = self.samples[idx]
        
        try:
            # 加载图像
            image_path = sample["image_path"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"无法打开图像 {image_path}: {e}")
                # 返回一个空图像（灰色背景）
                image = Image.new('RGB', (384, 384), color=(127, 127, 127))
            
            # 构建Janus格式的对话
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{sample['question']}",
                },
                {"role": "<|Assistant|>", "content": sample["answer"]},
            ]
            
            # 准备图像列表
            pil_images = [image]
            
            # 使用Janus处理器处理
            processed = self.processor(
                conversations=conversation,
                images=pil_images,  # 单独传递图像列表
                force_batchify=True
            )
            
            # 提取必要的张量
            input_ids = processed.input_ids[0]
            attention_mask = processed.attention_mask[0]
            pixel_values = processed.pixel_values[0].to(dtype=torch.float16)
            images_seq_mask = processed.images_seq_mask[0]
            images_emb_mask = processed.images_emb_mask[0]
            
            # 创建标签 (复制输入ID)
            labels = input_ids.clone()
            
            # 找到助手回复的开始位置，用于标记哪部分计算损失
            assistant_token = "<|Assistant|>"
            assistant_token_ids = self.processor.tokenizer.encode(assistant_token, add_special_tokens=False)
            
            if len(assistant_token_ids) > 0:
                assistant_token_id = assistant_token_ids[0]
                positions = (labels == assistant_token_id).nonzero(as_tuple=True)[0]
                
                if len(positions) > 0:
                    # 将用户输入部分设置为-100，不参与损失计算
                    assistant_start = positions[0]
                    labels[:assistant_start] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "images_seq_mask": images_seq_mask,
                "images_emb_mask": images_emb_mask,
                "labels": labels
            }
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            return self._create_empty_sample()
    
    def _create_empty_sample(self):
        """创建空样本用于错误处理"""
        # 针对Janus模型的默认值
        seq_len = 256
        image_size = 384
        num_image_tokens = 576  # Janus默认图像token数
        
        return {
            "input_ids": torch.zeros(seq_len, dtype=torch.long),
            "attention_mask": torch.zeros(seq_len, dtype=torch.long),
            "pixel_values": torch.zeros((3, image_size, image_size), dtype=torch.float16),
            "images_seq_mask": torch.zeros(seq_len, dtype=torch.bool),
            "images_emb_mask": torch.zeros((1, num_image_tokens), dtype=torch.bool),
            "labels": torch.full((seq_len,), -100, dtype=torch.long)
        }

def collate_fn(batch):
    """
    自定义数据整理函数，根据Janus的处理逻辑
    """
    # 过滤掉None值
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("批处理中没有有效样本")
    
    # 获取批处理大小和最大序列长度
    batch_size = len(batch)
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    
    # 获取第一个样本的图像相关维度
    first_item = batch[0]
    pixel_shape = first_item["pixel_values"].shape
    
    # Janus使用的pad_id (通常是0)
    pad_id = 0
    
    # 初始化批处理张量 - 左填充 (右对齐，符合Janus模型预期)
    input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    images_seq_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    # 获取图像嵌入掩码维度
    if hasattr(first_item["images_emb_mask"], "shape"):
        if len(first_item["images_emb_mask"].shape) == 1:
            # 1D掩码
            num_image_tokens = first_item["images_emb_mask"].size(0)
            images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens), dtype=torch.bool)
        elif len(first_item["images_emb_mask"].shape) == 2:
            # 2D掩码 
            n_images, num_image_tokens = first_item["images_emb_mask"].shape
            images_emb_mask = torch.zeros((batch_size, n_images, num_image_tokens), dtype=torch.bool)
        else:
            # 默认
            num_image_tokens = 576  # Janus默认值
            images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens), dtype=torch.bool)
    else:
        # 默认
        num_image_tokens = 576
        images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens), dtype=torch.bool)
    
    # 准备图像值张量 - 通常是[batch_size, 1, 3, height, width]
    if len(pixel_shape) == 3:  # [3, height, width]
        pixel_values = torch.zeros((batch_size, 1, *pixel_shape), dtype=torch.float16)
    else:  # 已经是[n_images, 3, height, width]
        pixel_values = torch.zeros((batch_size, *pixel_shape), dtype=torch.float16)
    
    # 填充每个样本 - 左填充 (右对齐)
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        # 右对齐
        input_ids[i, -seq_len:] = item["input_ids"]
        attention_mask[i, -seq_len:] = item["attention_mask"]
        labels[i, -seq_len:] = item["labels"]
        images_seq_mask[i, -seq_len:] = item["images_seq_mask"]
        
        # 图像相关张量
        if len(pixel_shape) == 3:  # [3, height, width]
            pixel_values[i, 0] = item["pixel_values"]
        else:  # 已经是[n_images, 3, height, width]
            pixel_values[i] = item["pixel_values"]
        
        # 处理images_emb_mask
        if hasattr(item["images_emb_mask"], "shape"):
            if len(item["images_emb_mask"].shape) == 1:
                # 1D掩码
                images_emb_mask[i, 0, :len(item["images_emb_mask"])] = item["images_emb_mask"]
            elif len(item["images_emb_mask"].shape) == 2:
                # 2D掩码
                images_emb_mask[i] = item["images_emb_mask"]
            # 对于其他情况，保持默认值
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "images_seq_mask": images_seq_mask,
        "images_emb_mask": images_emb_mask
    }