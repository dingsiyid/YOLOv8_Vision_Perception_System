import os
import cv2
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from stqdm import stqdm
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import time
import seaborn as sns

# Import model trainer module
import model_trainer as trainer

# Set page configuration
st.set_page_config(page_title="AI视觉感知系统", layout="wide", page_icon="🎯")

# Morandi color scheme for charts
MORANDI_COLORS = ['#B8A9C9', '#A3B8A8', '#D4B8B8', '#C4A88B', '#8BA88B', '#9DB5B2', '#D4C4A8', '#C9B8A8']

# Fixed camera display size
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360

# Function name to model file mapping
MODEL_MAPPING = {
    "目标检测 (快速)": "yolov8n.pt",
    "目标检测 (平衡)": "yolov8s.pt",
    "实例分割": "yolov8n-seg.pt",
    "姿态估计": "yolov8n-pose.pt",
}

# Function description with detailed features
FUNCTION_DESC = {
    "目标检测 (快速)": "使用边界框检测物体，最快速度，适合实时应用。",
    "目标检测 (平衡)": "使用边界框检测物体，速度与精度平衡。",
    "实例分割": "检测物体并生成像素级分割掩码。",
    "姿态估计": "检测人体关键点并生成骨架连接。",
}

# Detailed feature description for each function
FEATURE_DETAILS = {
    "目标检测 (快速)": {
        "title": "🎯 目标检测推理",
        "description": "对输入画面进行扫描，识别出画面中所有物体的位置与类别，输出物体的边界框、类别名称和置信度，让系统知道 \"画面里有什么、在哪里\"。",
        "features": [
            "🔍 扫描整幅图像，识别各类物体",
            "📦 输出精确的边界框位置 (x_min, y_min, x_max, y_max)",
            "🏷️ 标注物体类别名称",
            "📊 计算置信度分数，评估识别可靠性",
            "⚡ 最快推理速度，适合实时应用"
        ]
    },
    "目标检测 (平衡)": {
        "title": "🎯 目标检测推理",
        "description": "对输入画面进行扫描，识别出画面中所有物体的位置与类别，输出物体的边界框、类别名称和置信度，让系统知道 \"画面里有什么、在哪里\"。",
        "features": [
            "🔍 扫描整幅图像，识别各类物体",
            "📦 输出精确的边界框位置 (x_min, y_min, x_max, y_max)",
            "🏷️ 标注物体类别名称",
            "📊 计算置信度分数，评估识别可靠性",
            "⚖️ 平衡速度与精度，适用多数场景"
        ]
    },
    "实例分割": {
        "title": "🎨 实例分割推理",
        "description": "在目标检测基础上进一步精细化计算，以像素级精度勾勒出每个物体的完整轮廓与区域，生成物体掩码，让系统不仅知道物体位置，还能识别 \"物体的精确形状与边界\"。",
        "features": [
            "🎯 先进行目标检测定位物体",
            "🖼️ 像素级精确分割物体轮廓",
            "🎭 生成二进制掩码图像",
            "📐 输出物体轮廓坐标点序列",
            "✨ 精细区分同一类别的不同实例"
        ]
    },
    "姿态估计": {
        "title": "💃 姿态估计推理",
        "description": "针对画面中的人体进行关键点检测，定位人体 17 个关键骨骼节点，并自动连接生成骨架结构，实现 \"识别人体姿势、动作\" 的功能。",
        "features": [
            "👤 专门检测人体目标",
            "🔗 定位 17 个关键骨骼节点",
            "🦴 自动连接骨骼生成骨架",
            "📐 输出关键点精确坐标",
            "💪 识别人体姿势与动作"
        ]
    }
}

# Tracking feature description
TRACKING_FEATURES = {
    "title": "🔄 目标追踪推理",
    "description": "基于 DeepSORT 算法对视频或实时流进行跨帧分析，将前后帧中同一个目标进行匹配关联，为每个物体分配唯一 ID 并记录运动轨迹，保证视频中 \"同一个物体始终使用同一个编号\"。",
    "features": [
        "🔁 跨帧目标匹配与关联",
        "🆔 为每个目标分配唯一追踪 ID",
        "📍 记录目标运动轨迹",
        "🧵 绘制轨迹连线动画",
        "⚡ 基于 DeepSORT 算法实现"
    ]
}

# Colors for visualization
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
          (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def create_fixed_placeholder():
    """Create a fixed-size placeholder image"""
    gray_image = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8) * 40
    cv2.putText(gray_image, "CAMERA OFFLINE", (CAMERA_WIDTH//2 - 100, CAMERA_HEIGHT//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    return gray_image


def result_to_json(result: Results, tracker=None):
    """Convert result from ultralytics YOLOv8 prediction to json format"""
    len_results = len(result.boxes)
    result_list_json = [
        {
            'class_id': int(result.boxes.cls[idx]),
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.data[idx][0]),
                'y_min': int(result.boxes.data[idx][1]),
                'x_max': int(result.boxes.data[idx][2]),
                'y_max': int(result.boxes.data[idx][3]),
            },
        } for idx in range(len_results)
    ]
    if result.masks is not None:
        for idx in range(len_results):
            result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx].cpu().numpy(), 
                                                        (result.orig_shape[1], result.orig_shape[0])).tolist()
            result_list_json[idx]['segments'] = result.masks.xyn[idx].tolist()
    if result.keypoints is not None:
        for idx in range(len_results):
            result_list_json[idx]['keypoints'] = result.keypoints.xyn[idx].tolist()
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_min'],
                    result_list_json[idx]['bbox']['x_max'] - result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_max'] - result_list_json[idx]['bbox']['y_min']
                ],
                result_list_json[idx]['confidence'],
                result_list_json[idx]['class'],
            ) for idx in range(len_results)
        ]
        tracks = tracker.update_tracks(bbs, frame=result.orig_img)
        for idx, result_item in enumerate(result_list_json):
            matched = False
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_bbox = track.to_ltwh()
                det_bbox = [result_item['bbox']['x_min'], result_item['bbox']['y_min'],
                           result_item['bbox']['x_max'] - result_item['bbox']['x_min'],
                           result_item['bbox']['y_max'] - result_item['bbox']['y_min']]
                iou = calculate_iou(track_bbox, det_bbox)
                if iou > 0.5:
                    result_item['object_id'] = int(track.track_id)
                    matched = True
                    break
            if not matched:
                result_item['object_id'] = None
    return result_list_json


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def show_occupation_status(occupation_status):
    """显示盲道占用状态的辅助函数"""
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        if occupation_status['is_occupied']:
            st.error(f"⚠️ 状态: 被占用")
        else:
            st.success(f"✅ 状态: 畅通")
    
    with col_p2:
        st.metric("盲道数量", occupation_status['blind_road_count'])
    
    with col_p3:
        st.metric("占用物体数", occupation_status['occupied_count'])
    
    if occupation_status['occupied_count'] > 0:
        st.warning(f"📊 占用率: {occupation_status['occupation_rate']:.1%}")
        st.markdown("**占用物体详情:**")
        for obj in occupation_status['occupied_objects']:
            st.markdown(f"- {obj['class']}: 置信度 {obj['confidence']:.2%}")


def check_blind_road_occupation(blind_road_boxes, occupied_objects):
    """
    检测盲道占用情况
    :param blind_road_boxes: 盲道边界框列表
    :param occupied_objects: 占用物体列表
    :return: 占用状态字典
    """
    if not blind_road_boxes:
        return {
            'is_occupied': False,
            'occupied_count': 0,
            'occupied_objects': [],
            'blind_road_count': 0,
            'occupation_rate': 0.0
        }
    
    occupied = []
    for obj in occupied_objects:
        obj_bbox = obj['bbox']
        for road_bbox in blind_road_boxes:
            # 计算IoU
            iou = calculate_iou(
                [obj_bbox['x_min'], obj_bbox['y_min'], 
                 obj_bbox['x_max'] - obj_bbox['x_min'], 
                 obj_bbox['y_max'] - obj_bbox['y_min']],
                [road_bbox['x_min'], road_bbox['y_min'],
                 road_bbox['x_max'] - road_bbox['x_min'],
                 road_bbox['y_max'] - road_bbox['y_min']]
            )
            if iou > 0.1:  # IoU阈值，可调整
                occupied.append(obj)
                break
    
    occupation_rate = len(occupied) / len(blind_road_boxes) if blind_road_boxes else 0
    
    return {
        'is_occupied': len(occupied) > 0,
        'occupied_count': len(occupied),
        'occupied_objects': occupied,
        'blind_road_count': len(blind_road_boxes),
        'occupation_rate': occupation_rate
    }


def view_result(result: Results, result_list_json, centers=None):
    """Visualize result from ultralytics YOLOv8 prediction"""
    image = result.plot(labels=False, line_width=2)
    
    # 盲道占用检测相关变量
    blind_road_boxes = []  # 盲道边界框
    occupied_objects = []  # 占用物体
    
    for res in result_list_json:
        class_name = res['class']
        
        # 盲道使用黄色
        if class_name == 'blind_road':
            class_color = (0, 255, 255)  # 黄色
            blind_road_boxes.append(res['bbox'])
        # 车辆使用红色
        elif class_name == 'vehicle':
            class_color = (255, 0, 0)  # 红色
            occupied_objects.append(res)
        # 障碍物使用蓝色
        elif class_name == 'obstacle':
            class_color = (0, 0, 255)  # 蓝色
            occupied_objects.append(res)
        # 行人使用绿色
        elif class_name == 'person':
            class_color = (0, 255, 0)  # 绿色
            occupied_objects.append(res)
        else:
            class_color = COLORS[res['class_id'] % len(COLORS)]
        
        # 关键修复：只有当 object_id 存在、不为 None 且为有效整数时才显示 ID
        if 'object_id' in res and res['object_id'] is not None:
            text = f"{res['class']} #{res['object_id']}: {res['confidence']:.2f}"
        else:
            text = f"{res['class']}: {res['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(image, (res['bbox']['x_min'], res['bbox']['y_min'] - text_height - baseline), 
                      (res['bbox']['x_min'] + text_width, res['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text, (res['bbox']['x_min'], res['bbox']['y_min'] - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # 关键修复：只有当 object_id 存在且不为 None 时才使用追踪功能
        if 'object_id' in res and centers is not None and res['object_id'] is not None:
            object_id = res['object_id']
            # 确保 object_id 是有效的整数索引
            if isinstance(object_id, int) and object_id >= 0 and object_id < len(centers):
                centers[object_id].append((int((res['bbox']['x_min'] + res['bbox']['x_max']) / 2), 
                                           int((res['bbox']['y_min'] + res['bbox']['y_max']) / 2)))
                for j in range(1, len(centers[object_id])):
                    if centers[object_id][j - 1] is None or centers[object_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(image, centers[object_id][j - 1], centers[object_id][j], class_color, thickness)
    
    # 检测盲道占用情况
    occupation_status = check_blind_road_occupation(blind_road_boxes, occupied_objects)
    
    # 在图片上显示占用状态
    if occupation_status['is_occupied']:
        status_text = f"⚠️ 盲道被占用! 占用物体: {occupation_status['occupied_count']}个"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    elif blind_road_boxes:
        status_text = "✅ 盲道畅通"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    return image, occupation_status


def create_statistics_charts(result_list_json):
    """Create statistics charts from detection results"""
    if not result_list_json:
        return None
    
    class_names = [r['class'] for r in result_list_json]
    confidences = [r['confidence'] for r in result_list_json]
    
    class_counts = {}
    for name in class_names:
        class_counts[name] = class_counts.get(name, 0) + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#2C2C2C')
    
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        bars = ax1.bar(classes, counts, color=MORANDI_COLORS[:len(classes)])
        ax1.set_title('Class Distribution', color='white', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Object Class', color='#E8E8E8')
        ax1.set_ylabel('Count', color='#E8E8E8')
        ax1.tick_params(colors='#E8E8E8', rotation=45)
        ax1.set_facecolor('#3D3D3D')
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', color='white', fontsize=10)
    
    if confidences:
        ax2.hist(confidences, bins=10, color=MORANDI_COLORS[2], edgecolor='white', alpha=0.7)
        ax2.set_title('Confidence Distribution', color='white', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Confidence Score', color='#E8E8E8')
        ax2.set_ylabel('Frequency', color='#E8E8E8')
        ax2.tick_params(colors='#E8E8E8')
        ax2.set_facecolor('#3D3D3D')
        ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig


def create_detection_summary_table(result_list_json):
    """Create a summary table of detection results"""
    if not result_list_json:
        return pd.DataFrame()
    
    data = []
    for i, r in enumerate(result_list_json, 1):
        row = {
            'No.': i,
            'Class': r['class'],
            'Confidence': f"{r['confidence']:.2%}",
            'Position': f"({r['bbox']['x_min']}, {r['bbox']['y_min']})",
            'Size': f"{r['bbox']['x_max'] - r['bbox']['x_min']} x {r['bbox']['y_max'] - r['bbox']['y_min']}"
        }
        data.append(row)
    
    return pd.DataFrame(data)


def image_processing(frame, model, tracker=None, centers=None):
    """Process image frame"""
    results = model.predict(frame)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image, occupation_status = view_result(results[0], result_list_json, centers=centers)
    return result_image, result_list_json, occupation_status


def video_processing(video_file, model, tracker=None, centers=None):
    """Process video file and return video path, json path, and all results"""
    results = model.predict(video_file)
    model_name = os.path.basename(model.ckpt_path).split('.')[0]
    video_name = video_file.split('.')[0]
    output_folder = os.path.join('output_videos', video_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_file_name_out = os.path.join(output_folder, f"{video_name}_{model_name}_output.mp4")
    if os.path.exists(video_file_name_out):
        os.remove(video_file_name_out)
    result_video_json_file = os.path.join(output_folder, f"{video_name}_{model_name}_output.json")
    if os.path.exists(result_video_json_file):
        os.remove(result_video_json_file)
    
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(video_file_name_out, fourcc, fps, 
                                   (results[0].orig_img.shape[1], results[0].orig_img.shape[0]))
    
    # Store all detection results for statistics
    all_results = []
    all_occupation_status = []  # 存储每帧的占用状态
    results_list = list(results)
    
    for i, result in enumerate(stqdm(results_list, desc=f"Processing video")):
        result_list_json = result_to_json(result, tracker=tracker)
        result_image, occupation_status = view_result(result, result_list_json, centers=centers)
        video_writer.write(result_image)
        all_results.extend(result_list_json)
        all_occupation_status.append(occupation_status)
    
    video_writer.release()
    
    # Write JSON file after processing all frames
    with open(result_video_json_file, 'w') as json_file:
        json.dump(all_results, json_file, indent=2)
    
    return video_file_name_out, result_video_json_file, all_results, all_occupation_status


def process_single_frame(frame, model, tracker=None, centers=None):
    """Process a single frame and return results"""
    results = model(frame)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image, occupation_status = view_result(results[0], result_list_json, centers=centers)
    return result_image, result_list_json, occupation_status


# Ensure models directory exists
if not os.path.exists("models/"):
    os.makedirs("models/")

# Create model list from available files
available_models = [f.replace('.pt', '') for f in os.listdir('models/') if f.endswith('.pt')] if os.path.exists('models/') else []
function_list = [f for f in MODEL_MAPPING.keys() if MODEL_MAPPING[f].replace('.pt', '') in available_models]

if not function_list:
    function_list = list(MODEL_MAPPING.keys())

# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'video_results' not in st.session_state:
    st.session_state.video_results = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'camera_results' not in st.session_state:
    st.session_state.camera_results = None

# Header
st.title("🎯 AI视觉感知系统")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    # 添加盲道检测模式切换
    st.subheader("🚦 盲道检测模式")
    blind_road_mode = st.toggle("启用盲道检测", value=True, help="启用后自动检测盲道占用情况")
    
    # 盲道检测专用模型选择
    if blind_road_mode:
        st.success("✅ 盲道检测已启用")
        st.info("将自动识别盲道、车辆、障碍物、行人，并检测盲道占用情况")
        
        # 添加盲道检测模型选择
        st.markdown("---")
        st.subheader("🔧 盲道检测模型")
        blind_road_model = st.selectbox(
            "选择盲道检测模型",
            ["yolov8n.pt (通用)", "blind_road_detector.pt (盲道专用)"],
            index=1,
            help="选择用于盲道检测的模型"
        )
        
        if blind_road_model == "blind_road_detector.pt (盲道专用)":
            st.success("✅ 使用盲道专用检测模型")
    else:
        st.warning("⚠️ 盲道检测已禁用")
        blind_road_model = "yolov8n.pt (通用)"
    
    st.markdown("---")
    
    function_select = st.selectbox("📌 选择功能", function_list)
    
    if function_select in FUNCTION_DESC:
        st.info(f"💡 {FUNCTION_DESC[function_select]}")
    
    model_file = MODEL_MAPPING[function_select]
    model_path = f'models/{model_file}'
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file {model_file} not found.")
        with st.expander("📥 How to download?"):
            st.code(f"python -c \"from ultralytics import YOLO; YOLO('{model_file}')\"")
    else:
        model = YOLO(model_path)
        st.success(f"✅ Model loaded: {function_select}")
    
    st.markdown("---")
    
    st.subheader("📊 Parameters")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # Override model's default parameters
    model.conf = conf_threshold
    model.iou = iou_threshold
    
    st.markdown("---")
    
    st.markdown("---")
    
    st.subheader("📋 Current Function Features")
    if function_select in FEATURE_DETAILS:
        with st.expander(FEATURE_DETAILS[function_select]["title"], expanded=True):
            st.markdown(f"**功能描述：**")
            st.markdown(FEATURE_DETAILS[function_select]["description"])
            st.markdown("**核心特点：**")
            for feature in FEATURE_DETAILS[function_select]["features"]:
                st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    st.subheader("🔄 Tracking Features")
    with st.expander(TRACKING_FEATURES["title"], expanded=False):
        st.markdown(f"**功能描述：**")
        st.markdown(TRACKING_FEATURES["description"])
        st.markdown("**核心特点：**")
        for feature in TRACKING_FEATURES["features"]:
            st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **YOLOv8-based Visual Perception System**
    
    - 🚀 Object Detection
    - 🎯 Instance Segmentation  
    - 💃 Pose Estimation
    - 🔄 Object Tracking (DeepSORT)
    
    Built with YOLOv8 + DeepSORT + Streamlit
    """)

# Function to display statistics and charts
def display_statistics(result_list, title="Detection Statistics", function_name=None):
    """Display statistics charts and table for given results"""
    if result_list and len(result_list) > 0:
        with st.expander("📊 View Detection Details", expanded=True):
            # Display function features first
            if function_name and function_name in FEATURE_DETAILS:
                st.markdown(f"### {FEATURE_DETAILS[function_name]['title']}")
                st.markdown(FEATURE_DETAILS[function_name]['description'])
                st.markdown("**核心特点：**")
                for feature in FEATURE_DETAILS[function_name]['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
            
            total = len(result_list)
            classes = set([r['class'] for r in result_list])
            avg_conf = sum([r['confidence'] for r in result_list]) / total if total > 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Detections", total)
            col_b.metric("Classes Detected", len(classes))
            col_c.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            fig = create_statistics_charts(result_list)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader("Detection Results Table")
            df = create_detection_summary_table(result_list)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
        return True
    return False

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 图像处理", "🎬 视频处理", "📡 实时流", "📊 统计与导出", "🏋️ 模型训练"])

# Tab 1: Image Processing
with tab1:
    st.header("📷 图像处理")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        process_button = st.button("🚀 Process Image", use_container_width=True)
    
    if image_file is not None and process_button:
        with st.spinner("Processing..."):
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            
            # 根据选择的模型进行检测
            if blind_road_mode and blind_road_model == "blind_road_detector.pt (盲道专用)":
                # 使用盲道专用模型
                blind_model = YOLO("blind_road_detector.pt")
                results = blind_model(img, conf=0.25)
                result_list = result_to_json(results[0])
                img, occupation_status = view_result(results[0], result_list)
            else:
                # 使用默认模型
                img, result_list, occupation_status = image_processing(img, model)
            
            st.session_state.last_results = result_list
            st.session_state.last_image = img
            st.session_state.last_occupation_status = occupation_status
            st.session_state.video_results = None
            st.session_state.video_path = None
            st.session_state.camera_results = None
            
            with col2:
                st.image(img, caption="Detection Result", channels="BGR", use_container_width=True)
                
                # 显示盲道占用状态
                if occupation_status and blind_road_mode:
                    st.markdown("---")
                    st.subheader("🚦 盲道占用检测")
                    show_occupation_status(occupation_status)
    
    # 添加预设盲道场景选择（简化版）
    st.markdown("---")
    st.subheader("🎬 快速测试")
    st.info("点击下方按钮使用示例图片快速测试盲道检测功能")
    
    # 检查盲道数据集是否存在
    test_images_dir = r"D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation\blind_road_dataset\images\val"
    if os.path.exists(test_images_dir):
        test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            # 只显示第一个测试图片作为示例
            img_file = test_images[0]
            img_path = os.path.join(test_images_dir, img_file)
            if os.path.exists(img_path):
                col_prev, col_btn = st.columns([2, 1])
                with col_prev:
                    st.image(img_path, caption=f"示例图片: {img_file}", use_container_width=True)
                with col_btn:
                    st.markdown("")
                    st.markdown("")
                    if st.button("🔍 使用此图片检测", key="quick_test", use_container_width=True):
                        with st.spinner("Processing..."):
                            img = cv2.imread(img_path)
                            if img is not None:
                                # 根据选择的模型进行检测
                                if blind_road_mode and blind_road_model == "blind_road_detector.pt (盲道专用)":
                                    # 使用盲道专用模型
                                    blind_model = YOLO("blind_road_detector.pt")
                                    results = blind_model(img, conf=0.25)
                                    result_list_json = result_to_json(results[0])
                                    img, occupation_status = view_result(results[0], result_list_json)
                                else:
                                    # 使用默认模型
                                    img, result_list_json, occupation_status = image_processing(img, model)
                                
                                st.session_state.last_results = result_list_json
                                st.session_state.last_image = img
                                st.session_state.last_occupation_status = occupation_status
                                
                                st.image(img, caption=f"检测结果", channels="BGR", use_container_width=True)
                                
                                # 显示盲道占用状态
                                if occupation_status and blind_road_mode:
                                    st.markdown("---")
                                    st.subheader("🚦 盲道占用检测结果")
                                    show_occupation_status(occupation_status)
        else:
            st.warning("⚠️ 测试目录中没有图片")
    else:
        st.warning(f"⚠️ 测试目录不存在: {test_images_dir}")
    
    # Display statistics for image processing
    if st.session_state.last_results is not None and st.session_state.video_results is None:
        display_statistics(st.session_state.last_results, "Image Detection Statistics", function_select)

# Tab 2: Video Processing
with tab2:
    st.header("🎬 视频处理")
    st.info("Video processing automatically enables object tracking with unique IDs assigned to each target.")
    
    video_file = st.file_uploader("Upload a video", type=["mp4"])
    process_video_button = st.button("🚀 Process Video", key="video_btn")
    
    if video_file is not None and process_video_button:
        with st.spinner("Processing video, please wait..."):
            tracker = DeepSort(max_age=5)
            centers = [deque(maxlen=30) for _ in range(10000)]
            
            try:
                with open(video_file.name, "wb") as f:
                    f.write(video_file.read())
                
                video_file_out, result_video_json_file, all_results, all_occupation_status = video_processing(
                    video_file.name, model, tracker=tracker, centers=centers
                )
                
                # Store video results and path
                st.session_state.video_results = all_results
                st.session_state.video_path = video_file_out
                st.session_state.last_results = all_results
                st.session_state.camera_results = None
                st.session_state.video_occupation_status = all_occupation_status
                
                # Display the processed video
                st.video(video_file_out)
                
                st.success(f"✅ Processing complete! Processed {len(all_results)} detections from video.")
                
                # 显示盲道占用统计
                if all_occupation_status:
                    st.markdown("---")
                    st.subheader("🚦 盲道占用统计（视频）")
                    
                    occupied_frames = sum(1 for status in all_occupation_status if status['is_occupied'])
                    total_frames = len(all_occupation_status)
                    occupation_rate = occupied_frames / total_frames if total_frames > 0 else 0
                    
                    col_v1, col_v2, col_v3 = st.columns(3)
                    
                    with col_v1:
                        st.metric("总帧数", total_frames)
                    
                    with col_v2:
                        st.metric("占用帧数", occupied_frames)
                    
                    with col_v3:
                        st.metric("占用率", f"{occupation_rate:.1%}")
                    
                    if occupation_rate > 0:
                        st.warning(f"⚠️ 视频中有 {occupation_rate:.1%} 的时间盲道被占用")
            except Exception as e:
                st.error(f"❌ Video processing failed: {str(e)}")
            finally:
                if os.path.exists(video_file.name):
                    os.remove(video_file.name)
    
    # Display statistics for video processing
    if st.session_state.video_results is not None:
        display_statistics(st.session_state.video_results, "Video Detection Statistics", function_select)

# Tab 3: Live Stream
with tab3:
    st.header("📡 实时流处理")
    st.info("Live stream processing with object tracking. Click 'Capture & Analyze' to get statistics from current frame.")
    
    CAM_ID = st.text_input("Camera Source", "0", 
                           help="0=built-in camera, 1=USB camera, or RTSP address")
    if CAM_ID.isnumeric():
        CAM_ID = int(CAM_ID)
    
    # Create fixed-size display area
    display_area = st.empty()
    offline_frame = create_fixed_placeholder()
    display_area.image(offline_frame, channels="BGR", width=CAMERA_WIDTH, caption="📷 Camera Feed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_btn = st.button("▶️ START CAMERA", use_container_width=True, type="primary")
    with col2:
        stop_btn = st.button("⏹️ STOP CAMERA", use_container_width=True)
    with col3:
        capture_btn = st.button("📸 CAPTURE & ANALYZE", use_container_width=True)
    
    # Store current camera frame for capture
    if 'current_camera_frame' not in st.session_state:
        st.session_state.current_camera_frame = None
    
    # Handle start camera
    if start_btn:
        st.session_state.camera_running = True
        
        try:
            cap = cv2.VideoCapture(CAM_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            if not cap.isOpened():
                st.error("❌ Cannot open camera.")
            else:
                status_msg = st.empty()
                status_msg.success("🟢 Camera is running...")
                tracker = DeepSort(max_age=5)
                centers = [deque(maxlen=30) for _ in range(10000)]
                
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    st.session_state.current_camera_frame = annotated_frame
                    
                    display_area.image(annotated_frame, channels="BGR", width=CAMERA_WIDTH, caption="📷 Camera Feed (Live)")
                    time.sleep(0.01)
                
                cap.release()
                status_msg.empty()
                display_area.image(offline_frame, channels="BGR", width=CAMERA_WIDTH, caption="📷 Camera Feed")
                
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.session_state.camera_running = False
    
    # Handle stop camera
    if stop_btn:
        st.session_state.camera_running = False
        display_area.image(offline_frame, channels="BGR", width=CAMERA_WIDTH, caption="📷 Camera Feed")
    
    # Handle capture button - analyze current frame
    if capture_btn:
        if st.session_state.current_camera_frame is not None:
            with st.spinner("Analyzing frame..."):
                # Process the captured frame
                result_image, result_list, occupation_status = process_single_frame(st.session_state.current_camera_frame, model)
                st.session_state.camera_results = result_list
                st.session_state.last_results = result_list
                st.session_state.camera_occupation_status = occupation_status
                st.success(f"✅ Captured and analyzed! Found {len(result_list)} objects.")
                
                # Display the captured frame with detections
                st.image(result_image, channels="BGR", width=CAMERA_WIDTH, caption="Captured Frame")
                
                # 显示盲道占用状态
                if occupation_status:
                    st.markdown("---")
                    st.subheader("🚦 盲道占用检测")
                    
                    col_cam1, col_cam2, col_cam3 = st.columns(3)
                    
                    with col_cam1:
                        if occupation_status['is_occupied']:
                            st.error(f"⚠️ 状态: 被占用")
                        else:
                            st.success(f"✅ 状态: 畅通")
                    
                    with col_cam2:
                        st.metric("盲道数量", occupation_status['blind_road_count'])
                    
                    with col_cam3:
                        st.metric("占用物体数", occupation_status['occupied_count'])
                
                # Display statistics for captured frame
                display_statistics(result_list, "Captured Frame Statistics", function_select)
        else:
            st.warning("Please start the camera first, then click Capture.")

# Tab 4: Statistics & Export
with tab4:
    st.header("📊 统计与数据导出")
    
    # Determine which results to display
    results_to_show = None
    source_name = ""
    
    if st.session_state.video_results is not None:
        results_to_show = st.session_state.video_results
        source_name = "Video Processing"
    elif st.session_state.camera_results is not None:
        results_to_show = st.session_state.camera_results
        source_name = "Camera Capture"
    elif st.session_state.last_results is not None:
        results_to_show = st.session_state.last_results
        source_name = "Image Processing"
    
    if results_to_show and len(results_to_show) > 0:
        st.subheader(f"Results from: {source_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total = len(results_to_show)
            classes = set([r['class'] for r in results_to_show])
            conf_list = [r['confidence'] for r in results_to_show]
            
            st.metric("Total Objects Detected", total)
            st.metric("Unique Classes", len(classes))
            st.metric("Max Confidence", f"{max(conf_list):.1%}" if conf_list else "N/A")
            st.metric("Min Confidence", f"{min(conf_list):.1%}" if conf_list else "N/A")
        
        with col2:
            st.subheader("Class Breakdown")
            class_counts = {}
            for r in results_to_show:
                class_counts[r['class']] = class_counts.get(r['class'], 0) + 1
            
            for cls, count in class_counts.items():
                st.progress(count / total, text=f"{cls}: {count} ({count/total:.1%})")
        
        st.markdown("---")
        
        # Charts
        fig = create_statistics_charts(results_to_show)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        
        # Results table
        st.subheader("Detection Results Table")
        df = create_detection_summary_table(results_to_show)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Export options
        st.subheader("💾 Export Results")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📄 Export as JSON", use_container_width=True):
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'source': source_name,
                    'total_detections': total,
                    'results': results_to_show
                }
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col_export2:
            if st.button("📊 Export as CSV", use_container_width=True):
                df = create_detection_summary_table(results_to_show)
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Show video download button if video was processed
        if st.session_state.video_path is not None and os.path.exists(st.session_state.video_path):
            st.markdown("---")
            st.subheader("🎬 Processed Video")
            with open(st.session_state.video_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button(
                label="📥 Download Processed Video",
                data=video_bytes,
                file_name=os.path.basename(st.session_state.video_path),
                mime="video/mp4"
            )
        
        if st.session_state.last_image is not None and source_name == "Image Processing":
            if st.button("🖼️ Export Result Image", use_container_width=True):
                img_bgr = st.session_state.last_image
                st.download_button(
                    label="📥 Download Image",
                    data=cv2.imencode('.png', img_bgr)[1].tobytes(),
                    file_name=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
    else:
        st.info("👈 No detection results yet. Please process an image, video, or capture from camera first.")


# Tab 5: Model Training
with tab5:
    st.header("🏋️ 模型训练与迁移学习")
    
    # Training mode selection
    training_mode = st.selectbox("选择训练模式", ["全量训练", "迁移学习（微调）"])
    
    st.markdown("---")
    
    # Dataset configuration
    st.subheader("📁 数据集配置")
    
    # Dataset path input
    data_path = st.text_input("数据集路径", "./blind_road_dataset", help="数据集目录路径")
    
    # Load dataset button
    if st.button("🔍 加载数据集"):
        success, result = trainer.load_custom_dataset(data_path)
        if success:
            st.success("✅ 数据集加载成功！")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("训练图片", result['train_images'])
            col2.metric("验证图片", result['val_images'])
            col3.metric("训练标签", result['train_labels'])
            col4.metric("验证标签", result['val_labels'])
            
            # Store dataset info in session state
            st.session_state.dataset_info = result
            st.session_state.dataset_path = data_path
        else:
            st.error(f"❌ {result}")
    
    # Class names input
    class_names_input = st.text_input("类别名称", "blind_road, vehicle, obstacle, person", 
                                     help="逗号分隔的类别名称列表")
    class_names = [c.strip() for c in class_names_input.split(',')]
    
    # Create data.yaml button
    if st.button("📝 创建数据配置"):
        if 'dataset_path' in st.session_state:
            yaml_path = trainer.create_data_yaml(st.session_state.dataset_path, class_names)
            st.success(f"✅ 数据配置文件已创建: {yaml_path}")
            with open(yaml_path, 'r', encoding='utf-8') as f:
                st.code(f.read(), language='yaml')
        else:
            st.warning("⚠️ 请先加载数据集")
    
    st.markdown("---")
    
    # Model configuration
    st.subheader("⚙️ 模型配置")
    
    base_model = st.selectbox("基础模型", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                             help="选择YOLOv8基础模型进行训练")
    
    # Training parameters
    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        epochs = st.number_input("训练轮数", min_value=1, max_value=200, value=50, help="训练轮数")
    with col_param2:
        batch_size = st.number_input("批次大小", min_value=1, max_value=64, value=16, help="训练批次大小")
    with col_param3:
        imgsz = st.number_input("图像尺寸", min_value=320, max_value=1280, value=640, step=32, help="输入图像尺寸")
    
    # Device selection
    device = st.radio("训练设备", ["cpu", "cuda"], index=0, help="选择训练设备")
    
    st.markdown("---")
    
    # Transfer learning specific settings
    if training_mode == "迁移学习（微调）":
        st.subheader("🔄 迁移学习设置")
        pretrained_model_path = st.text_input("预训练模型路径", "./models/yolov8n.pt", 
                                              help="用于微调的预训练模型路径")
        freeze_layers = st.number_input("冻结层数", min_value=0, max_value=20, value=10,
                                        help="冻结的层数（0 = 训练全部）")
    
    st.markdown("---")
    
    # Start training button
    if st.button("🚀 开始训练", use_container_width=True, type="primary"):
        if 'dataset_path' not in st.session_state:
            st.error("❌ 请先加载数据集")
        else:
            # Check if data.yaml exists
            data_yaml_path = os.path.join(st.session_state.dataset_path, 'data.yaml')
            if not os.path.exists(data_yaml_path):
                st.error(f"❌ 数据配置文件未找到: {data_yaml_path}")
            else:
                with st.spinner("训练进行中... 这可能需要一些时间..."):
                    if training_mode == "全量训练":
                        success, result = trainer.train_model(
                            base_model,
                            data_yaml_path,
                            epochs=epochs,
                            batch_size=batch_size,
                            imgsz=imgsz,
                            device=device
                        )
                    else:
                        success, result = trainer.fine_tune_model(
                            pretrained_model_path,
                            data_yaml_path,
                            epochs=epochs,
                            batch_size=batch_size,
                            imgsz=imgsz,
                            device=device,
                            freeze_layers=freeze_layers
                        )
                    
                    if success:
                        st.success("🎉 训练完成！")
                        st.session_state.training_result = result
                        
                        # Display results
                        st.subheader("📊 训练结果")
                        st.write(f"**模型保存路径:** {result['model_path']}")
                        
                        if 'metrics' in result:
                            metrics = result['metrics']
                            if hasattr(metrics, 'keys'):
                                for key, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        st.metric(key, f"{value:.4f}")
                    else:
                        st.error(f"❌ 训练失败: {result}")
    
    # Display training results if available
    if 'training_result' in st.session_state:
        result_dir = st.session_state.training_result['model_path']
        
        # Generate confusion matrix
        st.markdown("---")
        st.subheader("📊 混淆矩阵")
        data_yaml_path = os.path.join(st.session_state.dataset_path, 'data.yaml') if 'dataset_path' in st.session_state else None
        
        if st.button("📊 生成混淆矩阵", use_container_width=True):
            if data_yaml_path and os.path.exists(data_yaml_path):
                best_model_path = os.path.join(result_dir, 'weights', 'best.pt')
                if os.path.exists(best_model_path):
                    with st.spinner("正在生成混淆矩阵..."):
                        success, conf_matrix_fig, class_names = trainer.generate_confusion_matrix(best_model_path, data_yaml_path, device=device)
                        if success:
                            st.pyplot(conf_matrix_fig)
                            plt.close(conf_matrix_fig)
                            
                            # Save confusion matrix
                            viz_dir = os.path.join(result_dir, 'visualization')
                            os.makedirs(viz_dir, exist_ok=True)
                            conf_matrix_fig.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=150)
                            st.success("✅ 混淆矩阵已保存！")
                        else:
                            st.error(f"❌ 生成混淆矩阵失败: {conf_matrix_fig}")
                else:
                    st.error("❌ 最佳模型未找到")
            else:
                st.error("❌ 数据集配置未找到")
        
        # Generate class distribution plot
        if 'dataset_path' in st.session_state and class_names_input:
            class_names_list = [c.strip() for c in class_names_input.split(',')]
            if st.button("📈 生成类别分布", use_container_width=True):
                with st.spinner("正在生成类别分布图..."):
                    success, dist_fig = trainer.generate_class_distribution_plot(st.session_state.dataset_path, class_names_list)
                    if success:
                        st.pyplot(dist_fig)
                        plt.close(dist_fig)
                        
                        # Save class distribution plot
                        viz_dir = os.path.join(result_dir, 'visualization')
                        os.makedirs(viz_dir, exist_ok=True)
                        dist_fig.savefig(os.path.join(viz_dir, 'class_distribution.png'), dpi=150)
                        st.success("✅ 类别分布图已保存！")
                    else:
                        st.warning(f"⚠️ 生成类别分布失败: {dist_fig}")
        
        st.markdown("---")
        st.subheader("📈 训练指标")
        
        # Check for results.csv
        results_csv = os.path.join(result_dir, 'results.csv')
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            st.dataframe(df, use_container_width=True)
            
            # Generate training plots using trainer module
            success, plots = trainer.generate_training_plots(results_csv)
            if success:
                # Display mAP plot
                st.subheader("mAP 指标")
                st.pyplot(plots['mAP'])
                plt.close(plots['mAP'])
                
                # Display precision-recall plot
                st.subheader("精确率与召回率")
                st.pyplot(plots['precision_recall'])
                plt.close(plots['precision_recall'])
                
                # Display loss plots
                st.subheader("训练与验证损失")
                col_loss1, col_loss2 = st.columns(2)
                with col_loss1:
                    st.pyplot(plots['loss'])
                    plt.close(plots['loss'])
                with col_loss2:
                    st.pyplot(plots['val_loss'])
                    plt.close(plots['val_loss'])
                
                # Display overall performance metrics
                if 'overall' in plots:
                    st.subheader("综合性能指标")
                    st.pyplot(plots['overall'])
                    plt.close(plots['overall'])
                
                # Save all plots
                viz_dir = os.path.join(result_dir, 'visualization')
                os.makedirs(viz_dir, exist_ok=True)
                trainer.save_training_results(result_dir, training_plots=plots)
                st.success("✅ All training plots saved!")
            else:
                st.warning(f"⚠️ Failed to generate plots: {plots}")
        
        # Show best model
        best_model_path = os.path.join(result_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            st.markdown("---")
            st.subheader("🏆 Best Model")
            st.write(f"Best model saved at: {best_model_path}")
            
            # Evaluate model
            if st.button("� Evaluate Model", use_container_width=True):
                if data_yaml_path and os.path.exists(data_yaml_path):
                    with st.spinner("Evaluating model..."):
                        success, metrics = trainer.evaluate_model(best_model_path, data_yaml_path, device=device)
                        if success:
                            col_eval1, col_eval2, col_eval3, col_eval4, col_eval5 = st.columns(5)
                            col_eval1.metric("mAP@0.5", f"{metrics['mAP50']:.4f}" if metrics['mAP50'] else "N/A")
                            col_eval2.metric("mAP@0.5:0.95", f"{metrics['mAP50_95']:.4f}" if metrics['mAP50_95'] else "N/A")
                            col_eval3.metric("Precision", f"{metrics['precision']:.4f}" if metrics['precision'] else "N/A")
                            col_eval4.metric("Recall", f"{metrics['recall']:.4f}" if metrics['recall'] else "N/A")
                            col_eval5.metric("F1 Score", f"{metrics['f1']:.4f}" if metrics['f1'] else "N/A")
                        else:
                            st.error(f"❌ Evaluation failed: {metrics}")
            
            # Download button
            with open(best_model_path, 'rb') as f:
                model_bytes = f.read()
            st.download_button(
                label="📥 Download Best Model",
                data=model_bytes,
                file_name="best.pt",
                mime="application/octet-stream",
                use_container_width=True
            )
    
    # Quick start guide
    st.markdown("---")
    st.subheader("📖 Quick Start Guide")
    with st.expander("How to train a custom model?", expanded=False):
        st.markdown("""
        **Step 1: Prepare Dataset**
        ```
        dataset/
        ├── images/
        │   ├── train/     # Training images (70%)
        │   └── val/       # Validation images (30%)
        └── labels/
            ├── train/     # YOLO format labels
            └── val/       # YOLO format labels
        ```
        
        **Step 2: Configure Classes**
        - Enter class names (comma-separated)
        - Click "Create Data Config" to generate data.yaml
        
        **Step 3: Start Training**
        - Select model size (n/s/m/l/x)
        - Set training parameters
        - Click "Start Training"
        
        **YOLO Label Format:**
        ```
        <class_id> <x_center> <y_center> <width> <height>
        ```
        All values are normalized (0-1).
        """)