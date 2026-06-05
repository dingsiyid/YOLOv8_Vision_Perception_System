"""
YOLOv8 Model Trainer Module
独立的模型训练模块，支持全量训练和迁移学习
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_custom_dataset(data_path):
    """
    加载自定义数据集
    :param data_path: 数据集目录路径
    :return: (success, result) - success为True时result是统计信息，否则是错误信息
    """
    if not os.path.exists(data_path):
        return False, f"数据集路径不存在: {data_path}"
    
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_name in required_dirs:
        full_path = os.path.join(data_path, dir_name)
        if not os.path.exists(full_path):
            return False, f"缺少必需目录: {full_path}"
    
    # 统计数据
    train_images = len([f for f in os.listdir(os.path.join(data_path, 'images/train')) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    val_images = len([f for f in os.listdir(os.path.join(data_path, 'images/val')) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    train_labels = len([f for f in os.listdir(os.path.join(data_path, 'labels/train')) 
                       if f.lower().endswith('.txt')])
    val_labels = len([f for f in os.listdir(os.path.join(data_path, 'labels/val')) 
                     if f.lower().endswith('.txt')])
    
    # 获取类别分布统计
    class_dist = analyze_class_distribution(data_path)
    
    return True, {
        'train_images': train_images,
        'val_images': val_images,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'total': train_images + val_images,
        'data_path': data_path,
        'class_distribution': class_dist
    }


def analyze_class_distribution(data_path):
    """
    分析数据集中各类别的分布情况
    :param data_path: 数据集路径
    :return: 类别分布字典
    """
    class_dist = {'train': {}, 'val': {}}
    
    for split in ['train', 'val']:
        label_dir = os.path.join(data_path, 'labels', split)
        if os.path.exists(label_dir):
            for label_file in os.listdir(label_dir):
                if label_file.lower().endswith('.txt'):
                    label_path = os.path.join(label_dir, label_file)
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                class_id = int(line.split()[0])
                                class_dist[split][class_id] = class_dist[split].get(class_id, 0) + 1
    
    return class_dist


def create_data_yaml(data_path, class_names):
    """
    创建 YOLO 数据集配置文件
    :param data_path: 数据集路径
    :param class_names: 类别名称列表
    :return: yaml文件路径
    """
    yaml_content = f"""path: {data_path}  # 数据集根路径
train: images/train  # 训练集图片路径
val: images/val      # 验证集图片路径

nc: {len(class_names)}  # 类别数量
names: {class_names}    # 类别名称
"""
    yaml_path = os.path.join(data_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    return yaml_path


def preprocess_dataset(data_path, target_size=(640, 640)):
    """
    数据集预处理：检查图片尺寸和格式，可选调整大小
    :param data_path: 数据集路径
    :param target_size: 目标尺寸
    :return: 预处理报告
    """
    report = {
        'total_images': 0,
        'resized_images': 0,
        'invalid_images': 0,
        'aspect_ratio_stats': [],
        'issues': []
    }
    
    for split in ['train', 'val']:
        img_dir = os.path.join(data_path, 'images', split)
        if not os.path.exists(img_dir):
            continue
        
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                report['total_images'] += 1
                img_path = os.path.join(img_dir, img_file)
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height
                        report['aspect_ratio_stats'].append(aspect_ratio)
                        
                        # 检查是否需要调整大小
                        if (width, height) != target_size:
                            report['resized_images'] += 1
                            
                except Exception as e:
                    report['invalid_images'] += 1
                    report['issues'].append(f"Invalid image: {img_file} - {str(e)}")
    
    # 计算统计信息
    if report['aspect_ratio_stats']:
        report['avg_aspect_ratio'] = np.mean(report['aspect_ratio_stats']).round(2)
        report['min_aspect_ratio'] = np.min(report['aspect_ratio_stats']).round(2)
        report['max_aspect_ratio'] = np.max(report['aspect_ratio_stats']).round(2)
    
    return report


def train_model(base_model_name, data_yaml_path, epochs=50, batch_size=16, imgsz=640, 
                device='cpu', output_dir='runs/detect', augment=True, patience=10):
    """
    训练/微调 YOLOv8 模型
    :param base_model_name: 基础模型名称（如 'yolov8n.pt'）
    :param data_yaml_path: 数据集配置文件路径
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param imgsz: 输入图像大小
    :param device: 训练设备 ('cpu' 或 'cuda')
    :param output_dir: 输出目录
    :param augment: 是否启用数据增强
    :param patience: 早停耐心值
    :return: (success, result)
    """
    try:
        # 加载预训练模型
        model = YOLO(base_model_name)
        
        # 获取数据集信息
        data_info = parse_data_yaml(data_yaml_path)
        
        # 训练模型
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=4,
            pretrained=True,
            augment=augment,
            patience=patience,
            verbose=True,
            project=output_dir,
            name='train',
            exist_ok=True
        )
        
        # 获取训练结果
        final_results = {
            'model_path': results.save_dir,
            'metrics': results.metrics,
            'epoch': results.epoch,
            'train_time': results.trainer.epoch_time,
            'class_names': data_info.get('names', []),
            'num_classes': data_info.get('nc', 0)
        }
        
        return True, final_results
    except Exception as e:
        return False, str(e)


def fine_tune_model(pretrained_model_path, data_yaml_path, epochs=30, batch_size=8, imgsz=640, 
                    device='cpu', freeze_layers=10, lr0=0.001):
    """
    迁移学习 - 微调预训练模型
    :param pretrained_model_path: 预训练模型路径
    :param data_yaml_path: 数据集配置文件路径
    :param epochs: 微调轮数
    :param batch_size: 批次大小
    :param imgsz: 输入图像大小
    :param device: 训练设备
    :param freeze_layers: 冻结层数
    :param lr0: 初始学习率
    :return: (success, result)
    """
    try:
        # 加载预训练模型
        model = YOLO(pretrained_model_path)
        
        # 获取数据集信息
        data_info = parse_data_yaml(data_yaml_path)
        
        # 冻结前几层，只训练头部
        model.freeze(layers=freeze_layers)
        print(f"✅ Frozen {freeze_layers} layers, training {len(model.model) - freeze_layers} layers")
        
        # 微调模型
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=4,
            pretrained=True,
            augment=False,
            patience=5,
            verbose=True,
            lr0=lr0,
            project='runs/detect',
            name='fine_tune',
            exist_ok=True
        )
        
        final_results = {
            'model_path': results.save_dir,
            'metrics': results.metrics,
            'epoch': results.epoch,
            'class_names': data_info.get('names', []),
            'num_classes': data_info.get('nc', 0),
            'frozen_layers': freeze_layers
        }
        
        return True, final_results
    except Exception as e:
        return False, str(e)


def parse_data_yaml(yaml_path):
    """
    解析数据配置文件
    :param yaml_path: yaml文件路径
    :return: 配置字典
    """
    import yaml
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return {}


def generate_confusion_matrix(model_path, data_yaml_path, device='cpu'):
    """
    生成混淆矩阵
    :param model_path: 训练好的模型路径
    :param data_yaml_path: 数据集配置文件路径
    :param device: 运行设备
    :return: (success, confusion_matrix_fig)
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 获取类别名称
        data_info = parse_data_yaml(data_yaml_path)
        class_names = data_info.get('names', [])
        
        # 验证并获取结果
        results = model.val(
            data=data_yaml_path,
            device=device,
            verbose=False
        )
        
        # 获取混淆矩阵
        conf_matrix = results.confusion_matrix
        
        # 确保类别名称与矩阵维度匹配
        if len(class_names) != conf_matrix.shape[0]:
            class_names = [str(i) for i in range(conf_matrix.shape[0])]
        
        # 获取混淆矩阵数据
        matrix_data = conf_matrix.numpy() if hasattr(conf_matrix, 'numpy') else np.array(conf_matrix)
        
        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if HAS_SEABORN:
            # 使用 seaborn 绘制热力图
            sns.heatmap(
                matrix_data,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                annot_kws={"size": 10}
            )
        else:
            # 使用 matplotlib 绘制热力图
            im = ax.imshow(matrix_data, cmap='Blues', interpolation='nearest')
            
            # 添加数值标注
            for i in range(matrix_data.shape[0]):
                for j in range(matrix_data.shape[1]):
                    ax.text(j, i, str(matrix_data[i, j]), 
                            ha='center', va='center', fontsize=10)
            
            # 设置刻度标签
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            
            # 添加颜色条
            fig.colorbar(im, ax=ax)
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        
        return True, fig, class_names
    except Exception as e:
        return False, str(e), []


def generate_training_plots(results_csv_path):
    """
    生成训练指标图表
    :param results_csv_path: 训练结果CSV文件路径
    :return: (success, figures_dict)
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(results_csv_path)
        
        figures = {}
        
        # 1. mAP曲线
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df['epoch'], df['metrics/mAP50'], label='mAP@0.5', color='#569CD6', linewidth=2)
        ax1.plot(df['epoch'], df['metrics/mAP50-95'], label='mAP@0.5:0.95', color='#4EC9B0', linewidth=2)
        ax1.set_title('mAP Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['mAP'] = fig1
        
        # 2. Precision-Recall曲线
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='#C586C0', linewidth=2)
        ax2.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='#DCDCAA', linewidth=2)
        ax2.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['precision_recall'] = fig2
        
        # 3. Loss曲线
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='#569CD6', linewidth=2)
        ax3.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='#C586C0', linewidth=2)
        ax3.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='#4EC9B0', linewidth=2)
        ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['loss'] = fig3
        
        # 4. Validation Loss曲线
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#569CD6', linewidth=2)
        ax4.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='#C586C0', linewidth=2)
        ax4.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', color='#4EC9B0', linewidth=2)
        ax4.set_title('Validation Loss', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['val_loss'] = fig4
        
        # 5. 学习率曲线
        if 'lr/pg0' in df.columns:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            ax5.plot(df['epoch'], df['lr/pg0'], label='PG0 LR', color='#569CD6', linewidth=2)
            ax5.plot(df['epoch'], df['lr/pg1'], label='PG1 LR', color='#C586C0', linewidth=2)
            ax5.plot(df['epoch'], df['lr/pg2'], label='PG2 LR', color='#4EC9B0', linewidth=2)
            ax5.set_title('Learning Rate', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('LR')
            ax5.legend(fontsize=12)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            figures['lr'] = fig5
        
        # 6. 综合指标对比图
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        epochs = df['epoch']
        ax6.plot(epochs, df['metrics/mAP50'], label='mAP@0.5', color='#569CD6', linewidth=2)
        ax6.plot(epochs, df['metrics/precision(B)'], label='Precision', color='#C586C0', linewidth=2)
        ax6.plot(epochs, df['metrics/recall(B)'], label='Recall', color='#4EC9B0', linewidth=2)
        ax6.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Score')
        ax6.legend(fontsize=12)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['overall'] = fig6
        
        return True, figures
    except Exception as e:
        return False, str(e)


def save_training_results(result_dir, confusion_matrix_fig=None, training_plots=None):
    """
    保存训练结果和可视化图像
    :param result_dir: 结果保存目录
    :param confusion_matrix_fig: 混淆矩阵图像
    :param training_plots: 训练图表字典
    :return: None
    """
    # 创建可视化目录
    viz_dir = os.path.join(result_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 保存混淆矩阵
    if confusion_matrix_fig:
        confusion_matrix_fig.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close(confusion_matrix_fig)
    
    # 保存训练图表
    if training_plots:
        for name, fig in training_plots.items():
            fig.savefig(os.path.join(viz_dir, f'{name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)


def get_train_results(model_path):
    """
    获取训练结果信息
    :param model_path: 模型保存目录路径
    :return: 结果字典
    """
    results = {}
    
    # 检查是否存在结果文件
    results_path = os.path.join(model_path, 'results.csv')
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        results['df'] = df
        results['best_mAP'] = df['metrics/mAP50'].max()
        results['best_epoch'] = df['metrics/mAP50'].idxmax() + 1  # 转换为1-based索引
        results['final_mAP'] = df['metrics/mAP50'].iloc[-1]
        results['final_precision'] = df['metrics/precision(B)'].iloc[-1]
        results['final_recall'] = df['metrics/recall(B)'].iloc[-1]
    
    # 检查是否存在最佳模型
    best_model_path = os.path.join(model_path, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        results['best_model'] = best_model_path
    
    # 检查是否存在最后模型
    last_model_path = os.path.join(model_path, 'weights', 'last.pt')
    if os.path.exists(last_model_path):
        results['last_model'] = last_model_path
    
    return results


def evaluate_model(model_path, data_yaml_path, device='cpu'):
    """
    评估模型性能
    :param model_path: 模型路径
    :param data_yaml_path: 数据集配置路径
    :param device: 运行设备
    :return: (success, metrics)
    """
    try:
        model = YOLO(model_path)
        
        results = model.val(
            data=data_yaml_path,
            device=device,
            verbose=False
        )
        
        metrics = {
            'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else None,
            'mAP50_95': float(results.box.map) if hasattr(results.box, 'map') else None,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else None,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else None,
            'f1': float(results.box.mf1) if hasattr(results.box, 'mf1') else None,
            'speed': results.speed if hasattr(results, 'speed') else None
        }
        
        return True, metrics
    except Exception as e:
        return False, str(e)


def generate_class_distribution_plot(data_path, class_names):
    """
    生成类别分布直方图
    :param data_path: 数据集路径
    :param class_names: 类别名称列表
    :return: (success, fig)
    """
    try:
        class_dist = analyze_class_distribution(data_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 训练集分布
        train_counts = [class_dist['train'].get(i, 0) for i in range(len(class_names))]
        ax1.bar(class_names, train_counts, color='#569CD6')
        ax1.set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 验证集分布
        val_counts = [class_dist['val'].get(i, 0) for i in range(len(class_names))]
        ax2.bar(class_names, val_counts, color='#C586C0')
        ax2.set_title('Validation Set Class Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return True, fig
    except Exception as e:
        return False, str(e)


def generate_prediction_examples(model_path, data_yaml_path, device='cpu', num_samples=5):
    """
    生成预测示例可视化
    :param model_path: 模型路径
    :param data_yaml_path: 数据集配置路径
    :param device: 运行设备
    :param num_samples: 样本数量
    :return: (success, fig)
    """
    try:
        model = YOLO(model_path)
        data_info = parse_data_yaml(data_yaml_path)
        
        # 获取验证集图片
        val_img_dir = os.path.join(data_info['path'], 'images', 'val')
        img_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        sample_files = img_files[:min(num_samples, len(img_files))]
        
        if not sample_files:
            return False, "No validation images found"
        
        # 计算子图布局
        cols = min(num_samples, 3)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i, img_file in enumerate(sample_files):
            img_path = os.path.join(val_img_dir, img_file)
            results = model.predict(img_path, device=device, verbose=False)
            
            # 绘制带预测结果的图像
            ax = axes[i]
            result_img = results[0].plot()
            ax.imshow(result_img)
            ax.set_title(f"Prediction: {img_file}", fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for j in range(len(sample_files), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        return True, fig
    except Exception as e:
        return False, str(e)


if __name__ == '__main__':
    # 示例用法
    print("=== YOLOv8 Model Trainer ===")
    
    # 示例：加载数据集
    dataset_path = './dataset'
    success, data_info = load_custom_dataset(dataset_path)
    if success:
        print(f"✅ 数据集加载成功：{data_info['total']} 张图片")
        print(f"   训练集：{data_info['train_images']} 张")
        print(f"   验证集：{data_info['val_images']} 张")
    else:
        print(f"❌ 数据集加载失败：{data_info}")
    
    # 示例：创建数据配置
    class_names = ['blind_road', 'vehicle', 'obstacle', 'person']
    yaml_path = create_data_yaml(dataset_path, class_names)
    print(f"✅ 数据配置文件已创建：{yaml_path}")
    
    # 示例：训练模型（取消注释以运行）
    # print("\n🚀 开始训练模型...")
    # success, result = train_model('yolov8n.pt', yaml_path, epochs=10, batch_size=8)
    # if success:
    #     print(f"✅ 训练完成，模型保存至：{result['model_path']}")
    # else:
    #     print(f"❌ 训练失败：{result}")
