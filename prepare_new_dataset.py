"""将LabelMe格式的盲道标注转换为YOLO格式"""

import os
import json
import shutil
import random

# 源数据路径
SOURCE_DIR = r"D:\working directory\Uhmw\25-26(2)\shixun\pic2\blind_dataset\blind_dataset(1)\images"
# 目标数据路径
TARGET_DIR = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset_v2"

def polygon_to_bbox(points, image_width, image_height):
    """将多边形转换为边界框"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # 转换为YOLO格式（归一化）
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return x_center, y_center, width, height

def convert_labelme_to_yolo(json_file, output_label_file, image_width, image_height):
    """将LabelMe JSON转换为YOLO txt格式"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    yolo_lines = []
    for shape in data.get('shapes', []):
        label = shape['label']
        points = shape['points']
        
        # 标签映射
        if label == 'blind_tile':
            class_id = 0  # blind_road
        elif label == 'vehicle':
            class_id = 1
        elif label == 'obstacle':
            class_id = 2
        elif label == 'person':
            class_id = 3
        else:
            print(f"未知标签: {label}")
            continue
        
        x_center, y_center, width, height = polygon_to_bbox(points, image_width, image_height)
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 写入YOLO格式标签
    with open(output_label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))
    
    return len(yolo_lines)

def prepare_dataset():
    """准备数据集"""
    print("=" * 60)
    print("Prepare Blind Road Detection Dataset")
    print("=" * 60)
    
    # 创建目录
    os.makedirs(os.path.join(TARGET_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels', 'val'), exist_ok=True)
    
    # 查找所有JSON文件
    json_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} annotation files")
    
    # 转换为YOLO格式
    valid_files = []
    for json_file in json_files:
        base_name = json_file.replace('.json', '')
        json_path = os.path.join(SOURCE_DIR, json_file)
        
        # 查找对应的图片
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(SOURCE_DIR, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"WARNING: Image not found: {base_name}")
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_width = data['imageWidth']
            image_height = data['imageHeight']
            
            # 生成YOLO标签
            label_name = f"{base_name}.txt"
            label_path = os.path.join(TARGET_DIR, 'labels', 'temp', label_name)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            num_objects = convert_labelme_to_yolo(json_path, label_path, image_width, image_height)
            
            if num_objects > 0:
                valid_files.append((image_path, label_path, base_name))
                print(f"OK {base_name}: {num_objects} objects")
            else:
                print(f"WARNING: No valid annotations: {base_name}")
                
        except Exception as e:
            print(f"ERROR: Conversion failed: {base_name} - {e}")
    
    print(f"\nValid files: {len(valid_files)}")
    
    # 随机分配训练集和验证集 (70% / 30%)
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * 0.7)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    
    print("\nCopying files...")
    for idx, (image_path, label_path, base_name) in enumerate(train_files):
        ext = os.path.splitext(image_path)[1]
        new_image_name = f"train_{idx:04d}{ext}"
        new_label_name = f"train_{idx:04d}.txt"
        
        shutil.copy(image_path, os.path.join(TARGET_DIR, 'images', 'train', new_image_name))
        shutil.copy(label_path, os.path.join(TARGET_DIR, 'labels', 'train', new_label_name))
    
    for idx, (image_path, label_path, base_name) in enumerate(val_files):
        ext = os.path.splitext(image_path)[1]
        new_image_name = f"val_{idx:04d}{ext}"
        new_label_name = f"val_{idx:04d}.txt"
        
        shutil.copy(image_path, os.path.join(TARGET_DIR, 'images', 'val', new_image_name))
        shutil.copy(label_path, os.path.join(TARGET_DIR, 'labels', 'val', new_label_name))
    
    print("Dataset preparation complete!")
    
    # 创建data.yaml
    yaml_content = f"""path: {TARGET_DIR}
train: images/train
val: images/val

nc: 4
names:
  - blind_road
  - vehicle
  - obstacle
  - person
"""
    yaml_path = os.path.join(TARGET_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"data.yaml created: {yaml_path}")
    
    return len(train_files), len(val_files)

if __name__ == "__main__":
    prepare_dataset()
