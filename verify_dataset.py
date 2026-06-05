"""验证盲道数据集标签"""

import os

def count_labels(label_dir):
    """统计标签数量"""
    total_labels = 0
    label_files = []
    
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            label_files.append(filename)
            with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_labels += len(lines)
    
    return label_files, total_labels

def check_dataset_structure(base_dir):
    """检查数据集结构"""
    print("=" * 60)
    print("数据集结构验证")
    print("=" * 60)
    
    # 检查目录
    dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    for d in dirs:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            files = os.listdir(path)
            print("OK %s: %d 个文件" % (d, len(files)))
        else:
            print("ERR %s: 目录不存在" % d)
    
    # 检查data.yaml
    yaml_path = os.path.join(base_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        print("OK data.yaml: 存在")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("ERR data.yaml: 不存在")

def verify_label_files(image_dir, label_dir):
    """验证标签文件与图片文件的对应关系"""
    print("\n" + "=" * 60)
    print("标签文件验证")
    print("=" * 60)
    
    image_files = set()
    label_files = set()
    
    # 获取图片文件
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(f)[0]
            image_files.add(name)
    
    # 获取标签文件
    for f in os.listdir(label_dir):
        if f.endswith('.txt'):
            name = os.path.splitext(f)[0]
            label_files.add(name)
    
    # 检查匹配情况
    missing_labels = image_files - label_files
    extra_labels = label_files - image_files
    
    print("图片文件: %d" % len(image_files))
    print("标签文件: %d" % len(label_files))
    
    if missing_labels:
        print("\nERR 缺少标签的图片: %d" % len(missing_labels))
        for name in missing_labels:
            print("  - %s" % name)
    else:
        print("OK 所有图片都有对应的标签")
    
    if extra_labels:
        print("\nERR 多余的标签文件: %d" % len(extra_labels))
        for name in extra_labels:
            print("  - %s.txt" % name)
    else:
        print("OK 没有多余的标签文件")

if __name__ == "__main__":
    base_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset"
    
    # 检查数据集结构
    check_dataset_structure(base_dir)
    
    # 验证训练集
    print("\n--- 训练集 ---")
    verify_label_files(
        os.path.join(base_dir, 'images/train'),
        os.path.join(base_dir, 'labels/train')
    )
    
    # 验证验证集
    print("\n--- 验证集 ---")
    verify_label_files(
        os.path.join(base_dir, 'images/val'),
        os.path.join(base_dir, 'labels/val')
    )
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)