"""盲道检测演示脚本"""

from ultralytics import YOLO
import cv2
import os

def detect_blind_road(image_path, model_path='blind_road_detector.pt'):
    """
    使用训练好的盲道检测模型进行推理
    :param image_path: 图片路径
    :param model_path: 模型路径
    :return: 检测结果图像
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 类别名称
    class_names = ['blind_road', 'vehicle', 'obstacle', 'person']
    colors = [
        (0, 255, 255),   # blind_road - 黄色
        (255, 0, 0),     # vehicle - 红色
        (0, 0, 255),     # obstacle - 蓝色
        (0, 255, 0)      # person - 绿色
    ]
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    # 推理
    results = model(img, conf=0.25)
    
    # 处理检测结果
    for result in results:
        for box in result.boxes:
            # 获取边界框
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取类别和置信度
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 绘制边界框
            color = colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            label = f"{class_names[cls]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def batch_detect(input_dir, output_dir):
    """批量检测图片"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            result_img = detect_blind_road(image_path)
            
            if result_img is not None:
                output_path = os.path.join(output_dir, f"detected_{filename}")
                cv2.imwrite(output_path, result_img)
                print(f"检测完成: {filename} -> {output_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 盲道检测演示")
    print("=" * 60)
    
    # 测试图片
    test_images = [
        r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\val\blind_road_val_0001.jpg"
    ]
    
    # 批量检测验证集
    input_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\val"
    output_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\results"
    
    print(f"\n📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("\n🔍 开始检测...")
    
    batch_detect(input_dir, output_dir)
    
    print("\n✅ 检测完成！")
    print(f"📊 检测结果已保存到: {output_dir}")