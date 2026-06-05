"""快速训练脚本 - 使用COCO8数据集"""

from ultralytics import YOLO
import os

def quick_train():
    """使用COCO8数据集进行快速训练"""
    print("=" * 60)
    print("🚀 开始快速训练 - COCO8数据集")
    print("=" * 60)

    # 检查COCO8数据集是否存在
    coco8_yaml = 'coco8.yaml'
    if not os.path.exists(coco8_yaml):
        print(f"❌ 未找到 {coco8_yaml} 文件")
        print("💡 正在从Ultralytics下载COCO8数据集...")

    # 加载预训练模型
    print("\n📥 加载预训练的YOLOv8n模型...")
    model = YOLO('yolov8n.pt')

    # 训练参数配置
    print("\n⚙️  训练参数配置：")
    print("   - 数据集: COCO8 (8张图片)")
    print("   - 训练轮数: 10 epochs")
    print("   - 图像尺寸: 640x640")
    print("   - 批次大小: 16")
    print("   - 设备: CPU")

    # 开始训练
    print("\n🏋️  开始训练...")
    results = model.train(
        data='coco8.yaml',  # COCO8数据集
        epochs=10,          # 训练轮数
        imgsz=640,          # 图像尺寸
        batch=16,           # 批次大小
        device='cpu',       # 使用CPU
        project='runs',     # 项目目录
        name='quick_train', # 实验名称
        exist_ok=True,      # 覆盖已存在的实验
        patience=5,         # 早停耐心值
        save=True,          # 保存检查点
        plots=True,         # 生成图表
        verbose=True        # 详细输出
    )

    print("\n✅ 训练完成！")
    print(f"📁 模型保存位置: runs/quick_train/weights/best.pt")
    print(f"📊 训练结果: runs/quick_train/")

    return results

def train_blind_road_detection():
    """训练盲道检测模型"""
    print("=" * 60)
    print("🚀 开始训练 - 盲道检测")
    print("=" * 60)

    # 检查数据集配置
    data_yaml = './blind_road_dataset/data.yaml'
    if not os.path.exists(data_yaml):
        print(f"❌ 未找到数据集配置文件: {data_yaml}")
        print("💡 请先准备盲道检测数据集")
        return None

    # 加载预训练模型
    print("\n📥 加载预训练的YOLOv8n模型...")
    model = YOLO('yolov8n.pt')

    # 训练参数配置
    print("\n⚙️  训练参数配置：")
    print("   - 数据集: 盲道检测")
    print("   - 训练轮数: 50 epochs")
    print("   - 图像尺寸: 640x640")
    print("   - 批次大小: 16")
    print("   - 设备: CPU")

    # 开始训练
    print("\n🏋️  开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',
        project='runs',
        name='blind_road_detection',
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        verbose=True
    )

    print("\n✅ 训练完成！")
    print(f"📁 模型保存位置: runs/blind_road_detection/weights/best.pt")
    print(f"📊 训练结果: runs/blind_road_detection/")

    return results

if __name__ == "__main__":
    import sys

    print("\n请选择训练模式：")
    print("1. 快速训练（COCO8数据集，8张图片，约5分钟）")
    print("2. 盲道检测训练（需要自己准备数据集）")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请输入选项 (1/2): ").strip()

    if choice == "1":
        results = quick_train()

    elif choice == "2":
        results = train_blind_road_detection()

    else:
        print("❌ 无效选项！")
        sys.exit(1)