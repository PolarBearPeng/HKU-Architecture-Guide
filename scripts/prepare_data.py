"""
数据准备脚本 - 增强版本
支持多种图像处理模式
"""

import os
import json
import shutil
from PIL import Image
import argparse
import sys
from tqdm import tqdm
import concurrent.futures
import numpy as np
sys.path.append(".")
from config.config import Config
from config.building_info import BUILDING_INFO

def create_building_description(building_name):
    """创建建筑描述模板"""
    return {
        "name": f"{building_name}",
        "description": f"{building_name}是香港大学的一座重要建筑。",
        "features": "请填写建筑特点。",
        "history": "请填写历史背景。",
        "functions": "请填写使用功能。"
    }

def process_image(src_path, dst_path, target_size, mode='crop'):
    """
    处理单张图像
    
    Args:
        src_path: 源图像路径
        dst_path: 目标图像路径
        target_size: 目标尺寸 (width, height)
        mode: 处理模式
            - 'crop': 中心裁剪
            - 'pad': 保持宽高比并添加白边
            - 'stretch': 直接拉伸到目标尺寸
            - 'smart_crop': 智能裁剪以保留主体
    """
    try:
        img = Image.open(src_path).convert("RGB")
        orig_width, orig_height = img.size
        
        if mode == 'crop':
            # 调整大小后进行中心裁剪
            # 先将图像按照较短边缩放
            if orig_width / orig_height > target_size[0] / target_size[1]:
                # 原图更宽，先调整高度
                new_height = target_size[1]
                new_width = int(orig_width * (new_height / orig_height))
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # 从中间裁剪
                left = (new_width - target_size[0]) // 2
                img_cropped = img_resized.crop((left, 0, left + target_size[0], target_size[1]))
            else:
                # 原图更高，先调整宽度
                new_width = target_size[0]
                new_height = int(orig_height * (new_width / orig_width))
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # 从中间裁剪
                top = (new_height - target_size[1]) // 2
                img_cropped = img_resized.crop((0, top, target_size[0], top + target_size[1]))
            
            processed_img = img_cropped
            
        elif mode == 'pad':
            # 保持宽高比并填充
            ratio = min(target_size[0] / orig_width, target_size[1] / orig_height)
            new_size = (int(orig_width * ratio), int(orig_height * ratio))
            img_resized = img.resize(new_size, Image.LANCZOS)
            
            background = Image.new('RGB', target_size, (240, 240, 240))  # 使用浅灰色而非纯白
            offset = ((target_size[0] - new_size[0]) // 2,
                     (target_size[1] - new_size[1]) // 2)
            background.paste(img_resized, offset)
            
            processed_img = background
            
        elif mode == 'stretch':
            # 直接拉伸到目标尺寸
            processed_img = img.resize(target_size, Image.LANCZOS)
            
        elif mode == 'smart_crop':
            # 智能裁剪 - 简化版本
            # 这里使用一个基础的方法：假设图像中央区域更重要
            # 实际应用中可以使用更先进的物体检测算法
            
            # 首先缩放图像，保持宽高比
            ratio = max(target_size[0] / orig_width, target_size[1] / orig_height)
            new_size = (int(orig_width * ratio), int(orig_height * ratio))
            img_resized = img.resize(new_size, Image.LANCZOS)
            
            # 然后从中心裁剪
            left = (new_size[0] - target_size[0]) // 2
            top = (new_size[1] - target_size[1]) // 2
            img_cropped = img_resized.crop((
                left, top, 
                left + target_size[0], top + target_size[1]
            ))
            
            processed_img = img_cropped
        
        # 保存处理后的图像
        processed_img.save(dst_path, quality=90, optimize=True)
        return True
    except Exception as e:
        print(f"处理图像 {src_path} 时出错: {e}")
        return False

def prepare_data(data_dir, output_dir, target_size=(384, 288), mode='crop', num_workers=4):
    """
    准备训练数据
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        target_size: 目标图像尺寸
        mode: 图像处理模式
        num_workers: 并行处理的工作线程数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有建筑文件夹
    building_folders = [f for f in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, f))]
    
    # 创建建筑信息文件
    building_info_file = os.path.join(output_dir, "building_info_template.json")
    
    # 使用已有的建筑信息或创建模板
    building_info_output = {}
    for building in building_folders:
        if building in BUILDING_INFO:
            building_info_output[building] = BUILDING_INFO[building]
        else:
            standard_name = building.replace("_", " ")
            building_info_output[building] = create_building_description(standard_name)
        
        # 创建建筑输出目录
        building_output_dir = os.path.join(output_dir, building)
        os.makedirs(building_output_dir, exist_ok=True)
    
    # 保存建筑信息
    with open(building_info_file, 'w', encoding='utf-8') as f:
        json.dump(building_info_output, f, ensure_ascii=False, indent=4)
    
    print(f"建筑信息已保存至 {building_info_file}")
    
    # 计算原始图像的平均宽高比，辅助决定最佳目标尺寸
    width_sum, height_sum, count = 0, 0, 0
    for building in building_folders:
        building_dir = os.path.join(data_dir, building)
        images = [f for f in os.listdir(building_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in images[:5]:  # 只取样几张计算
            try:
                img_path = os.path.join(building_dir, img_file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    width_sum += width
                    height_sum += height
                    count += 1
            except Exception:
                pass
    
    if count > 0:
        avg_ratio = (width_sum / count) / (height_sum / count)
        print(f"原始图像的平均宽高比: {avg_ratio:.2f}")
        print(f"推荐的目标尺寸: {target_size[0]}x{int(target_size[0]/avg_ratio)}")
    
    # 并行处理图像
    tasks = []
    for building in building_folders:
        building_dir = os.path.join(data_dir, building)
        building_output_dir = os.path.join(output_dir, building)
        
        images = [f for f in os.listdir(building_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in images:
            src_path = os.path.join(building_dir, img_file)
            dst_path = os.path.join(building_output_dir, img_file)
            tasks.append((src_path, dst_path, target_size, mode))
    
    # 显示处理进度
    print(f"开始处理 {len(tasks)} 张图像，处理模式: {mode}...")
    success_count = 0
    
    with tqdm(total=len(tasks)) as pbar:
        # 使用线程池并行处理图像
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_image, src, dst, size, mode) 
                      for src, dst, size, mode in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)
    
    print(f"图像处理完成。成功处理: {success_count}/{len(tasks)}")
    print(f"数据准备完成。请检查 {output_dir} 目录")
    
    # 保存一个示例图片进行检查
    if len(tasks) > 0:
        sample_src, sample_dst, _, _ = tasks[0]
        sample_check_dst = os.path.join(output_dir, "sample_processed.jpg")
        shutil.copy(sample_dst, sample_check_dst)
        print(f"已保存处理示例图片: {sample_check_dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备HKU建筑数据集")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="处理后的数据输出目录")
    parser.add_argument("--width", type=int, default=384, help="目标图像宽度")
    parser.add_argument("--height", type=int, default=288, help="目标图像高度")
    parser.add_argument("--mode", type=str, choices=['crop', 'pad', 'stretch', 'smart_crop'], 
                       default='crop', help="图像处理模式")
    parser.add_argument("--workers", type=int, default=4, help="并行处理的工作线程数")
    
    args = parser.parse_args()
    prepare_data(args.data_dir, args.output_dir, (args.width, args.height), args.mode, args.workers)