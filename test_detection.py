"""盲道占用检测测试 - 展示检测结果"""

from ultralytics import YOLO
import cv2
import os

# 加载训练好的模型
model = YOLO('blind_road_detector.pt')

# 类别配置
CLASS_NAMES = ['blind_road', 'vehicle', 'obstacle', 'person']
CLASS_COLORS = {
    'blind_road': (0, 255, 255),   # 黄色
    'vehicle': (255, 0, 0),         # 红色
    'obstacle': (0, 0, 255),        # 蓝色
    'person': (0, 255, 0)           # 绿色
}

def test_blind_road_detection(image_path):
    """测试盲道检测"""
    print(f"\n{'='*60}")
    print(f"测试图片: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    # 推理
    results = model(img, conf=0.25)
    
    # 统计
    detected = {name: 0 for name in CLASS_NAMES}
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = CLASS_NAMES[cls]
            color = CLASS_COLORS[class_name]
            
            detected[class_name] += 1
            
            # 绘制
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 输出结果
    print(f"\n检测结果:")
    for name, count in detected.items():
        print(f"  {name}: {count} 个")
    
    is_occupied = detected['vehicle'] > 0 or detected['obstacle'] > 0 or detected['person'] > 0
    if detected['blind_road'] > 0:
        print(f"\n盲道状态: {'[警告] 被占用' if is_occupied else '[正常] 畅通'}")
    
    # 保存结果
    output_path = image_path.replace('\\val\\', '\\results\\').replace('\\train\\', '\\results\\')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"\n结果已保存: {output_path}")
    
    return detected

if __name__ == "__main__":
    # 测试验证集
    val_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\val"
    
    if os.path.exists(val_dir):
        files = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"找到 {len(files)} 张测试图片")
        
        # 测试前3张
        for f in files[:3]:
            test_blind_road_detection(os.path.join(val_dir, f))
    else:
        print(f"目录不存在: {val_dir}")
