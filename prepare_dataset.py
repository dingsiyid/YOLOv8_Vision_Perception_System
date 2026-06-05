"""盲道数据集处理脚本 - 完整版本"""

import os
import shutil
import random
from pathlib import Path

def process_blind_road_images(src_dir, dest_dir, train_ratio=0.7):
    """处理盲道图片：去重、重命名、分配到train/val目录"""
    
    # 创建目标目录结构
    train_img_dir = os.path.join(dest_dir, 'images', 'train')
    val_img_dir = os.path.join(dest_dir, 'images', 'val')
    train_label_dir = os.path.join(dest_dir, 'labels', 'train')
    val_label_dir = os.path.join(dest_dir, 'labels', 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 获取所有图片文件
    all_files = []
    for f in os.listdir(src_dir):
        f_path = os.path.join(src_dir, f)
        if os.path.isfile(f_path) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_files.append(f_path)
    
    print("发现 %d 个图片文件" % len(all_files))
    
    # 去重：保留原始文件，删除副本标记
    unique_files = {}
    for f_path in all_files:
        filename = os.path.basename(f_path)
        # 去除 "- 副本 (数字)" 或 "- 副本"
        base_name = filename
        if " - 副本" in base_name:
            # 找到 "- 副本" 的位置
            idx = base_name.find(" - 副本")
            # 获取扩展名
            ext = os.path.splitext(base_name)[1]
            # 获取原始名称（到 "- 副本" 之前）
            original_name = base_name[:idx] + ext
        else:
            original_name = base_name
        
        # 只保留第一次出现的文件
        if original_name not in unique_files:
            unique_files[original_name] = f_path
    
    unique_list = list(unique_files.values())
    print("去重后剩余 %d 个唯一文件" % len(unique_list))
    
    # 随机打乱
    random.seed(42)  # 设置种子保证结果可重复
    random.shuffle(unique_list)
    
    # 分配到train和val
    train_count = int(len(unique_list) * train_ratio)
    train_files = unique_list[:train_count]
    val_files = unique_list[train_count:]
    
    print("分配结果：训练集 %d 张，验证集 %d 张" % (len(train_files), len(val_files)))
    
    # 删除旧文件
    for f in os.listdir(train_img_dir):
        os.remove(os.path.join(train_img_dir, f))
    for f in os.listdir(val_img_dir):
        os.remove(os.path.join(val_img_dir, f))
    
    # 重命名并复制文件
    for i, f_path in enumerate(train_files, 1):
        ext = os.path.splitext(f_path)[1]
        new_name = "blind_road_train_%04d%s" % (i, ext)
        shutil.copy(f_path, os.path.join(train_img_dir, new_name))
        print("复制训练集: %s" % new_name)
    
    for i, f_path in enumerate(val_files, 1):
        ext = os.path.splitext(f_path)[1]
        new_name = "blind_road_val_%04d%s" % (i, ext)
        shutil.copy(f_path, os.path.join(val_img_dir, new_name))
        print("复制验证集: %s" % new_name)
    
    print("\n图片处理完成！")
    print("训练集图片: %s" % train_img_dir)
    print("验证集图片: %s" % val_img_dir)
    
    return {
        'total_files': len(all_files),
        'unique_files': len(unique_list),
        'train_count': len(train_files),
        'val_count': len(val_files)
    }

def update_data_yaml(dest_dir):
    """更新数据集配置文件"""
    yaml_content = """path: ./blind_road_dataset  # 数据集根路径
train: images/train  # 训练集图片
val: images/val      # 验证集图片

nc: 4  # 类别数量
names: ['blind_road', 'vehicle', 'obstacle', 'person']  # 类别名称
"""
    
    yaml_path = os.path.join(dest_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("更新配置文件: %s" % yaml_path)

def create_labeling_instructions(dest_dir):
    """创建标注说明文档"""
    instructions = """# 盲道数据集标注说明

## 数据集结构

blind_road_dataset/
├── data.yaml              # 数据集配置文件
├── images/
│   ├── train/            # 训练集图片
│   └── val/              # 验证集图片
└── labels/
    ├── train/            # 训练集标签
    └── val/              # 验证集标签

## 类别定义

| class_id | 类别名称 | 说明 |
|---------|---------|------|
| 0 | blind_road | 盲道区域（黄色盲道砖） |
| 1 | vehicle | 占用盲道的车辆（自行车、电动车、汽车等） |
| 2 | obstacle | 占用盲道的障碍物（箱子、垃圾、障碍物等） |
| 3 | person | 行人（在盲道上或附近的行人） |

## 标注格式

每个标签文件（.txt）的格式：
class_id x_center y_center width height

其中：
- class_id: 类别编号（0-3）
- x_center, y_center: 边界框中心点的归一化坐标（0-1）
- width, height: 边界框的归一化宽度和高度（0-1）

## 使用LabelImg标注

1. 安装LabelImg:
   pip install labelImg

2. 启动LabelImg:
   labelImg

3. 设置目录:
   - Open Dir: 选择 images/train 或 images/val
   - Change Save Dir: 选择 labels/train 或 labels/val

4. 开始标注:
   - 按 w 开始标注
   - 选择类别并绘制边界框
   - 按 d 保存并切换到下一张图片

## 标注注意事项

1. 盲道区域: 标注所有可见的盲道砖区域
2. 占用检测: 标注所有在盲道上的车辆和障碍物
3. 行人: 标注在盲道上或即将进入盲道的行人
4. 边界框: 尽量贴近物体边缘
5. 文件名: 标签文件名必须与图片文件名一致

## 参考资料

- LabelImg官方文档: https://github.com/HumanSignal/labelImg
- YOLO格式说明: https://docs.ultralytics.com/datasets/detect/
"""
    
    readme_path = os.path.join(dest_dir, 'ANNOTATION_GUIDE.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("创建标注说明文档: %s" % readme_path)

if __name__ == "__main__":
    src_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\pic"
    dest_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset"
    
    print("=" * 60)
    print("盲道数据集处理脚本")
    print("=" * 60)
    
    # 处理图片
    result = process_blind_road_images(src_dir, dest_dir)
    
    # 更新配置文件
    update_data_yaml(dest_dir)
    
    # 创建标注说明
    create_labeling_instructions(dest_dir)
    
    print("\n" + "=" * 60)
    print("处理结果统计")
    print("=" * 60)
    print("原始文件数: %d" % result['total_files'])
    print("去重后文件数: %d" % result['unique_files'])
    print("训练集文件数: %d" % result['train_count'])
    print("验证集文件数: %d" % result['val_count'])
    print("=" * 60)