"""盲道自动标注脚本 - 基于颜色识别生成初始标签"""

import os
import cv2
import numpy as np

def detect_blind_road_color(image_path, lower_yellow=[10, 100, 100], upper_yellow=[40, 255, 255]):
    """
    基于颜色检测盲道区域
    :param image_path: 图片路径
    :param lower_yellow: 黄色下限 (HSV)
    :param upper_yellow: 黄色上限 (HSV)
    :return: 检测到的边界框列表 [x_center, y_center, width, height]
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return []
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 创建黄色掩码
    lower = np.array(lower_yellow, dtype=np.uint8)
    upper = np.array(upper_yellow, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # 形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    img_height, img_width = img.shape[:2]
    
    for contour in contours:
        # 过滤小轮廓
        area = cv2.contourArea(contour)
        if area < 1000:  # 最小面积阈值
            continue
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 转换为归一化坐标 (YOLO格式)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        # 过滤宽高比不合理的框
        if width > 0.8 or height > 0.8:
            continue
        
        boxes.append([x_center, y_center, width, height])
    
    return boxes

def detect_common_objects(image_path, conf_threshold=0.5):
    """
    使用YOLOv8预训练模型检测常见物体
    :param image_path: 图片路径
    :param conf_threshold: 置信度阈值
    :return: 检测到的物体列表 [(class_id, x_center, y_center, width, height)]
    """
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model(image_path, conf=conf_threshold)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                # 只保留人、自行车、汽车等相关类别
                # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
                if cls in [0, 1, 2, 3, 5, 7]:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    # 转换类别编号 (0: person, 1: vehicle)
                    if cls == 0:
                        boxes.append((3, x_center, y_center, width, height))  # person
                    else:
                        boxes.append((1, x_center, y_center, width, height))  # vehicle
        
        return boxes
    except Exception as e:
        print(f"物体检测失败: {e}")
        return []

def generate_label_file(image_path, output_dir, use_color_detection=True, use_object_detection=True):
    """
    生成YOLO格式的标签文件
    :param image_path: 图片路径
    :param output_dir: 标签输出目录
    :param use_color_detection: 是否使用颜色检测盲道
    :param use_object_detection: 是否使用YOLO检测物体
    """
    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{filename}.txt")
    
    labels = []
    
    # 颜色检测盲道
    if use_color_detection:
        blind_road_boxes = detect_blind_road_color(image_path)
        for box in blind_road_boxes:
            x_center, y_center, width, height = box
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # YOLO检测物体
    if use_object_detection:
        object_boxes = detect_common_objects(image_path)
        for cls, x_center, y_center, width, height in object_boxes:
            labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 写入标签文件
    os.makedirs(output_dir, exist_ok=True)
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')
    
    return len(labels)

def batch_generate_labels(input_dir, output_dir):
    """
    批量生成标签文件
    :param input_dir: 图片输入目录
    :param output_dir: 标签输出目录
    """
    print(f"开始处理目录: {input_dir}")
    print(f"标签输出目录: {output_dir}")
    
    total_images = 0
    total_labels = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            num_labels = generate_label_file(image_path, output_dir)
            total_images += 1
            total_labels += num_labels
            print(f"处理: {filename} -> {num_labels} 个标签")
    
    print(f"\n处理完成！")
    print(f"总图片数: {total_images}")
    print(f"总标签数: {total_labels}")
    
    return total_images, total_labels

def create_labelimg_config(output_dir):
    """
    创建LabelImg配置文件
    """
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation>
  <folder>VOC2007</folder>
  <filename>test.jpg</filename>
  <path>test.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
</annotation>
"""
    
    # 创建LabelImg配置文件
    labelimg_config = {
        "auto_save": True,
        "default_save_dir": output_dir.replace('\\', '/'),
        "classes": ["blind_road", "vehicle", "obstacle", "person"],
        "shortcut_keys": {
            "save": "Ctrl+S",
            "next_image": "D",
            "prev_image": "A",
            "create_box": "W",
            "delete_box": "Del",
            "copy_box": "Ctrl+C",
            "paste_box": "Ctrl+V",
            "zoom_in": "Ctrl++",
            "zoom_out": "Ctrl+-",
            "fit_window": "Ctrl+F",
            "label_list": "Ctrl+L"
        }
    }
    
    # 保存配置文件
    config_path = os.path.join(output_dir, 'labelimg_config.json')
    import json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(labelimg_config, f, indent=2, ensure_ascii=False)
    
    print(f"LabelImg配置文件已创建: {config_path}")
    
    return config_path

def create_labelimg_shortcut_config():
    """
    创建LabelImg快捷键配置
    """
    import platform
    if platform.system() == 'Windows':
        # Windows配置路径
        appdata = os.getenv('APPDATA')
        labelimg_dir = os.path.join(appdata, 'labelImg')
        os.makedirs(labelimg_dir, exist_ok=True)
        
        shortcut_content = """[shortcuts]
save=ctrl+s
nextImage=d
prevImage=a
createBox=w
deleteBox=del
copyBox=ctrl+c
pasteBox=ctrl+v
zoomIn=ctrl++
zoomOut=ctrl+-
fitWindow=ctrl+f
labelList=ctrl+l
verifyImage=ctrl+v
quit=ctrl+q
"""
        shortcut_path = os.path.join(labelimg_dir, 'shortcuts.ini')
        with open(shortcut_path, 'w', encoding='utf-8') as f:
            f.write(shortcut_content)
        
        print(f"LabelImg快捷键配置已创建: {shortcut_path}")
        return shortcut_path
    else:
        print("⚠️ 快捷键配置仅支持Windows系统")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("盲道数据集自动标注脚本")
    print("=" * 60)
    
    # 处理训练集
    print("\n1. 处理训练集...")
    train_img_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\train"
    train_label_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\labels\train"
    batch_generate_labels(train_img_dir, train_label_dir)
    
    # 处理验证集
    print("\n2. 处理验证集...")
    val_img_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\val"
    val_label_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\labels\val"
    batch_generate_labels(val_img_dir, val_label_dir)
    
    # 创建LabelImg配置
    print("\n3. 创建LabelImg配置...")
    create_labelimg_config(train_label_dir)
    create_labelimg_shortcut_config()
    
    print("\n" + "=" * 60)
    print("自动标注完成！")
    print("=" * 60)
    print("\n接下来：")
    print("1. 使用LabelImg打开 images/train 目录")
    print("2. 设置保存目录为 labels/train")
    print("3. 按 W 开始标注，按 D 切换下一张")
    print("4. 检查并修正自动生成的标签")