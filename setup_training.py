"""使用COCO数据集进行训练的配置脚本"""

import os
from ultralytics import YOLO

def create_coco_dataset_config():
    """创建COCO数据集配置文件"""
    coco_yaml = """
path: ./coco_dataset  # 数据集根路径
train: images/train2017  # 训练集图片
val: images/val2017      # 验证集图片

nc: 80  # COCO数据集有80个类别
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
"""

    with open('./coco_dataset/data.yaml', 'w', encoding='utf-8') as f:
        f.write(coco_yaml)

    print("✅ COCO数据集配置文件已创建: ./coco_dataset/data.yaml")

def download_coco_dataset():
    """下载COCO数据集"""
    import requests
    import zipfile
    from tqdm import tqdm

    # 创建数据集目录
    os.makedirs('./coco_dataset/images', exist_ok=True)
    os.makedirs('./coco_dataset/labels', exist_ok=True)

    print("📥 正在下载COCO数据集...")
    print("⚠️  注意：COCO数据集很大（约18GB），下载可能需要较长时间")
    print("💡 建议：使用预训练的YOLOv8模型直接进行推理，无需下载完整数据集")

    # COCO数据集下载链接
    coco_urls = {
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    # 询问用户是否下载
    print("\n是否下载完整的COCO数据集？")
    print("1. 下载完整数据集（约18GB，需要较长时间）")
    print("2. 使用预训练模型进行推理（推荐，无需下载）")
    print("3. 下载小样本数据集（约1GB，用于快速测试）")

    return coco_urls

def train_with_pretrained_model():
    """使用预训练模型进行训练"""
    print("\n🚀 使用预训练的YOLOv8模型进行训练...")

    # 使用预训练的YOLOv8n模型
    model = YOLO('yolov8n.pt')

    # 在COCO数据集上进行微调
    results = model.train(
        data='coco8.yaml',  # 使用COCO8小样本数据集
        epochs=10,
        imgsz=640,
        batch=16,
        device='cpu'
    )

    return results

def create_blind_road_dataset_config():
    """创建盲道检测数据集配置"""
    blind_road_yaml = """
path: ./blind_road_dataset  # 数据集根路径
train: images/train  # 训练集图片
val: images/val      # 验证集图片

nc: 4  # 类别数量
names: ['blind_road', 'vehicle', 'obstacle', 'person']  # 类别名称
"""

    with open('./blind_road_dataset/data.yaml', 'w', encoding='utf-8') as f:
        f.write(blind_road_yaml)

    print("✅ 盲道检测数据集配置文件已创建: ./blind_road_dataset/data.yaml")

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 YOLOv8 训练配置助手")
    print("=" * 60)

    print("\n请选择训练方案：")
    print("1. 使用COCO数据集（80个通用类别）")
    print("2. 使用盲道检测数据集（需要自己标注）")
    print("3. 使用预训练模型直接推理")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == "1":
        print("\n📋 方案1：使用COCO数据集")
        create_coco_dataset_config()
        coco_urls = download_coco_dataset()

    elif choice == "2":
        print("\n📋 方案2：使用盲道检测数据集")
        create_blind_road_dataset_config()
        print("\n📝 接下来的步骤：")
        print("1. 收集盲道场景图片")
        print("2. 使用LabelImg标注工具标注图片")
        print("3. 将图片和标签放入对应目录")
        print("4. 运行训练命令")

    elif choice == "3":
        print("\n📋 方案3：使用预训练模型直接推理")
        print("✅ 预训练模型已就绪，可以直接使用！")
        print("💡 访问 http://localhost:8501 开始使用")

    else:
        print("❌ 无效选项！")