"""将LabelMe格式标注转换为YOLO格式"""

import os
import json
import shutil
import random

def polygon_to_bbox(points, image_width, image_height):
    """将多边形点转换为YOLO格式的边界框"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # 转换为归一化坐标
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return x_center, y_center, width, height

def convert_labelme_to_yolo(json_path, output_label_dir):
    """将单个LabelMe JSON转换为YOLO标签"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    image_name = data['imagePath']
    base_name = os.path.splitext(image_name)[0]
    
    labels = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        # 转换标签名称为类别ID
        if label == 'blind_tile' or label == 'blind_road':
            class_id = 0
        elif label == 'vehicle' or label == 'car':
            class_id = 1
        elif label == 'obstacle':
            class_id = 2
        elif label == 'person':
            class_id = 3
        else:
            print(f"未知标签: {label}，跳过")
            continue
        
        # 转换为YOLO格式
        x_center, y_center, width, height = polygon_to_bbox(points, image_width, image_height)
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 写入标签文件
    label_path = os.path.join(output_label_dir, f"{base_name}.txt")
    with open(label_path, 'w', encoding='utf-8') as f:
        for line in labels:
            f.write(line + '\n')
    
    return len(labels)

def batch_convert_and_copy(source_dir, dest_dir, train_ratio=0.7):
    """批量转换并复制已标注数据"""
    # 源目录结构
    source_images = os.path.join(source_dir, 'images')
    source_masks = os.path.join(source_dir, 'masks')
    
    # 目标目录结构
    train_img_dir = os.path.join(dest_dir, 'images', 'train')
    val_img_dir = os.path.join(dest_dir, 'images', 'val')
    train_label_dir = os.path.join(dest_dir, 'labels', 'train')
    val_label_dir = os.path.join(dest_dir, 'labels', 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = []
    for f in os.listdir(source_images):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(f)[0]
            json_path = os.path.join(source_images, f"{base_name}.json")
            if os.path.exists(json_path):
                image_files.append((f, json_path))
    
    print(f"发现 {len(image_files)} 个已标注的图片")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(image_files)
    
    # 分配到train和val
    train_count = int(len(image_files) * train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    print(f"训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")
    
    # 处理训练集
    train_label_count = 0
    for i, (img_file, json_path) in enumerate(train_files, 1):
        # 复制图片
        src_img = os.path.join(source_images, img_file)
        ext = os.path.splitext(img_file)[1]
        new_name = f"blind_road_train_extra_{i:04d}{ext}"
        dst_img = os.path.join(train_img_dir, new_name)
        shutil.copy(src_img, dst_img)
        
        # 转换标签
        json_base = os.path.splitext(os.path.basename(json_path))[0]
        temp_label = convert_labelme_to_yolo(json_path, train_label_dir)
        # 重命名标签文件
        old_label = os.path.join(train_label_dir, f"{json_base}.txt")
        new_label = os.path.join(train_label_dir, f"blind_road_train_extra_{i:04d}.txt")
        if os.path.exists(old_label):
            os.rename(old_label, new_label)
            train_label_count += temp_label
        
        print(f"训练集: {new_name} -> {temp_label} 个标签")
    
    # 处理验证集
    val_label_count = 0
    for i, (img_file, json_path) in enumerate(val_files, 1):
        # 复制图片
        src_img = os.path.join(source_images, img_file)
        ext = os.path.splitext(img_file)[1]
        new_name = f"blind_road_val_extra_{i:04d}{ext}"
        dst_img = os.path.join(val_img_dir, new_name)
        shutil.copy(src_img, dst_img)
        
        # 转换标签
        json_base = os.path.splitext(os.path.basename(json_path))[0]
        temp_label = convert_labelme_to_yolo(json_path, val_label_dir)
        # 重命名标签文件
        old_label = os.path.join(val_label_dir, f"{json_base}.txt")
        new_label = os.path.join(val_label_dir, f"blind_road_val_extra_{i:04d}.txt")
        if os.path.exists(old_label):
            os.rename(old_label, new_label)
            val_label_count += temp_label
        
        print(f"验证集: {new_name} -> {temp_label} 个标签")
    
    return {
        'total_images': len(image_files),
        'train_images': len(train_files),
        'val_images': len(val_files),
        'train_labels': train_label_count,
        'val_labels': val_label_count
    }

if __name__ == "__main__":
    source_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\pic2\blind_dataset"
    dest_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset"
    
    print("=" * 60)
    print("转换LabelMe标注到YOLO格式")
    print("=" * 60)
    
    result = batch_convert_and_copy(source_dir, dest_dir)
    
    print("\n" + "=" * 60)
    print("转换结果统计")
    print("=" * 60)
    print(f"总图片数: {result['total_images']}")
    print(f"训练集图片: {result['train_images']}")
    print(f"验证集图片: {result['val_images']}")
    print(f"训练集标签数: {result['train_labels']}")
    print(f"验证集标签数: {result['val_labels']}")
    print("=" * 60)