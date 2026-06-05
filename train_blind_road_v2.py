"""训练盲道检测模型 - 新数据集"""

from ultralytics import YOLO
import os

def train_blind_road_model():
    """使用新数据集训练盲道检测模型"""
    
    print("=" * 60)
    print("Train Blind Road Detection Model (New Dataset)")
    print("=" * 60)
    
    # 数据集配置
    data_yaml = "./blind_road_dataset_v2/data.yaml"
    
    # 检查数据集是否存在
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset config not found: {data_yaml}")
        return
    
    print(f"Dataset config: {data_yaml}")
    
    # 加载预训练模型
    print("\nLoading pretrained model...")
    model = YOLO('yolov8n.pt')
    
    # 训练模型
    print("\nStarting training...")
    print("-" * 60)
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device='cpu',
        project='runs',
        name='blind_road_v2',
        exist_ok=True,
        patience=20,
        augment=True,
        mixup=0.3,
        copy_paste=0.3,
        mosaic=0.8,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best model saved: {results.save_dir}/weights/best.pt")
    print(f"Last model saved: {results.save_dir}/weights/last.pt")
    
    # 复制最佳模型到主目录
    best_model = os.path.join(results.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model):
        import shutil
        shutil.copy(best_model, 'blind_road_detector_v2.pt')
        print(f"\nModel copied to: blind_road_detector_v2.pt")
    
    return results

if __name__ == "__main__":
    train_blind_road_model()
