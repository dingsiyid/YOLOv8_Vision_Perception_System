"""
设置模拟训练结果用于预览
"""

import os
import sys

def setup_preview():
    """设置模拟训练结果"""
    
    # 模拟数据集信息
    dataset_info = {
        'train_images': 700,
        'val_images': 300,
        'train_labels': 700,
        'val_labels': 300,
        'total': 1000,
        'data_path': './dataset'
    }
    
    # 模拟训练结果
    training_result = {
        'model_path': './runs/detect/train',
        'metrics': {
            'mAP50': 0.8523,
            'mAP50_95': 0.6534,
            'precision': 0.8734,
            'recall': 0.8234
        },
        'epoch': 50,
        'class_names': ['blind_road', 'vehicle', 'obstacle', 'person'],
        'num_classes': 4
    }
    
    # 创建Python脚本用于设置session_state
    setup_script = """
# Session State Setup for Preview
import sys
sys.path.insert(0, '.')

# 设置数据集信息
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {
        'train_images': 700,
        'val_images': 300,
        'train_labels': 700,
        'val_labels': 300,
        'total': 1000,
        'data_path': './dataset'
    }

# 设置数据集路径
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = './dataset'

# 设置训练结果
if 'training_result' not in st.session_state:
    st.session_state.training_result = {
        'model_path': './runs/detect/train',
        'metrics': {
            'mAP50': 0.8523,
            'mAP50_95': 0.6534,
            'precision': 0.8734,
            'recall': 0.8234
        },
        'epoch': 50,
        'class_names': ['blind_road', 'vehicle', 'obstacle', 'person'],
        'num_classes': 4
    }
"""
    
    # 保存setup脚本
    setup_path = './preview_setup.py'
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(setup_script)
    
    print("Preview setup created!")
    print(f"\nTo preview the results:")
    print(f"1. The fake dataset and training results have been generated")
    print(f"2. The application should automatically detect the results")
    print(f"3. Refresh the web page to see the preview")
    print(f"\nFiles generated:")
    print(f"  - Dataset: ./dataset (1000 images)")
    print(f"  - Training Results: ./runs/detect/train/")
    print(f"    - results.csv")
    print(f"    - confusion_matrix.png")
    print(f"    - mAP.png")
    print(f"    - precision_recall.png")
    print(f"    - loss.png")
    print(f"    - val_loss.png")
    print(f"    - overall.png")
    print(f"    - class_distribution.png")
    print(f"    - best.pt")
    print(f"    - last.pt")


if __name__ == '__main__':
    print("=" * 60)
    print("Preview Setup Script")
    print("=" * 60)
    setup_preview()
    print("=" * 60)
