"""清理多余的标签文件"""

import os

def clean_extra_labels(image_dir, label_dir):
    """删除多余的标签文件"""
    # 获取图片文件名（不含扩展名）
    image_names = set()
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(f)[0]
            image_names.add(name)
    
    # 删除多余的标签文件
    deleted_count = 0
    for f in os.listdir(label_dir):
        if f.endswith('.txt'):
            name = os.path.splitext(f)[0]
            if name not in image_names:
                os.remove(os.path.join(label_dir, f))
                print("删除: %s" % f)
                deleted_count += 1
    
    return deleted_count

if __name__ == "__main__":
    base_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset"
    
    print("=" * 60)
    print("清理多余标签文件")
    print("=" * 60)
    
    # 清理训练集
    print("\n1. 清理训练集标签...")
    train_image_dir = os.path.join(base_dir, 'images/train')
    train_label_dir = os.path.join(base_dir, 'labels/train')
    deleted_train = clean_extra_labels(train_image_dir, train_label_dir)
    print("训练集删除: %d 个文件" % deleted_train)
    
    # 清理验证集
    print("\n2. 清理验证集标签...")
    val_image_dir = os.path.join(base_dir, 'images/val')
    val_label_dir = os.path.join(base_dir, 'labels/val')
    deleted_val = clean_extra_labels(val_image_dir, val_label_dir)
    print("验证集删除: %d 个文件" % deleted_val)
    
    print("\n" + "=" * 60)
    print("清理完成！")
    print("=" * 60)