"""盲道占用检测训练脚本"""

from ultralytics import YOLO
import os

def train_blind_road_detector():
    """训练盲道检测模型"""
    print("=" * 60)
    print("🚀 启动盲道占用检测模型训练")
    print("=" * 60)
    
    # 加载预训练模型
    print("\n📥 加载预训练的YOLOv8n模型...")
    model = YOLO('yolov8n.pt')
    
    # 数据集配置
    data_yaml = './blind_road_dataset/data.yaml'
    
    if not os.path.exists(data_yaml):
        print(f"❌ 数据集配置文件未找到: {data_yaml}")
        return None
    
    print("\n⚙️ 训练参数配置：")
    print("   - 数据集: blind_road_dataset")
    print("   - 训练轮数: 50 epochs")
    print("   - 图像尺寸: 640x640")
    print("   - 批次大小: 8")
    print("   - 设备: CPU")
    
    # 开始训练
    print("\n🏋️  开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,           # 减小批次大小避免内存问题
        device='cpu',
        project='runs',
        name='blind_road_detection',
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
        augment=True,      # 启用数据增强
        mixup=0.5,         # MixUp增强
        copy_paste=0.5,    # Copy-Paste增强
        mosaic=0.5         # Mosaic增强
    )
    
    print("\n✅ 训练完成！")
    print(f"📁 模型保存位置: runs/blind_road_detection/weights/best.pt")
    
    return results

def evaluate_model():
    """评估训练好的模型"""
    print("\n📊 评估模型性能...")
    
    model_path = 'runs/blind_road_detection/weights/best.pt'
    data_yaml = './blind_road_dataset/data.yaml'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件未找到: {model_path}")
        return None
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device='cpu')
    
    print("\n📈 评估结果:")
    print(f"   - mAP@0.5: {results.box.map:.4f}")
    print(f"   - mAP@0.5:0.95: {results.box.map50_95:.4f}")
    print(f"   - Precision: {results.box.mp:.4f}")
    print(f"   - Recall: {results.box.mr:.4f}")
    
    return results

if __name__ == "__main__":
    # 训练模型
    train_results = train_blind_road_detector()
    
    # 评估模型
    if train_results:
        evaluate_model()
    
    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 使用Web界面加载训练好的模型")
    print("2. 上传图片测试盲道检测效果")
    print("3. 继续标注更多数据提升模型精度")