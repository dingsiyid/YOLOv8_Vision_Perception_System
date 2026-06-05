"""盲道自动标注脚本 - 简化版"""

import os
import cv2
import numpy as np

def detect_blind_road_color(image_path):
    """基于颜色检测盲道区域"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取: {image_path}")
        return []
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 黄色范围
    lower_yellow = np.array([10, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    img_height, img_width = img.shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        if width > 0.9 or height > 0.9:
            continue
        
        boxes.append([x_center, y_center, width, height])
    
    return boxes

def generate_label_file(image_path, output_dir):
    """生成YOLO格式标签文件"""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{filename}.txt")
    
    boxes = detect_blind_road_color(image_path)
    
    with open(label_path, 'w', encoding='utf-8') as f:
        for box in boxes:
            x_center, y_center, width, height = box
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return len(boxes)

def batch_generate_labels(input_dir, output_dir):
    """批量生成标签"""
    print(f"处理目录: {input_dir}")
    
    total = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            num = generate_label_file(image_path, output_dir)
            total += num
            print(f"{filename}: {num} 个盲道区域")
    
    return total

if __name__ == "__main__":
    print("=" * 60)
    print("盲道自动标注脚本 (简化版)")
    print("=" * 60)
    
    # 处理训练集
    print("\n1. 处理训练集...")
    train_img = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\train"
    train_label = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\labels\train"
    os.makedirs(train_label, exist_ok=True)
    train_total = batch_generate_labels(train_img, train_label)
    
    # 处理验证集
    print("\n2. 处理验证集...")
    val_img = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\val"
    val_label = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\labels\val"
    os.makedirs(val_label, exist_ok=True)
    val_total = batch_generate_labels(val_img, val_label)
    
    print("\n" + "=" * 60)
    print(f"训练集: {train_total} 个盲道标签")
    print(f"验证集: {val_total} 个盲道标签")
    print("自动标注完成！")
    print("=" * 60)