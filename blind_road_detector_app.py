"""
盲道占用检测系统 - 简化版
专注于盲道检测功能
"""

import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics.engine.results import Results

# 设置页面配置
st.set_page_config(page_title="盲道占用检测系统", layout="wide", page_icon="🦯")

# 类别配置
CLASS_NAMES = ['blind_road', 'vehicle', 'obstacle', 'person']
CLASS_COLORS = {
    'blind_road': (0, 255, 255),   # 黄色 - 盲道
    'vehicle': (255, 0, 0),         # 红色 - 车辆
    'obstacle': (0, 0, 255),        # 蓝色 - 障碍物
    'person': (0, 255, 0)           # 绿色 - 行人
}

def result_to_json(result: Results):
    """将YOLO结果转换为JSON格式"""
    result_list = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
        
        result_list.append({
            'class': class_name,
            'class_id': class_id,
            'confidence': float(box.conf[0]),
            'bbox': {
                'x_min': int(x1),
                'y_min': int(y1),
                'x_max': int(x2),
                'y_max': int(y2)
            }
        })
    return result_list

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
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

def check_blind_road_occupation(blind_road_boxes, occupied_objects):
    """检测盲道占用情况"""
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
            iou = calculate_iou(
                [obj_bbox['x_min'], obj_bbox['y_min'], 
                 obj_bbox['x_max'] - obj_bbox['x_min'], 
                 obj_bbox['y_max'] - obj_bbox['y_min']],
                [road_bbox['x_min'], road_bbox['y_min'],
                 road_bbox['x_max'] - road_bbox['x_min'],
                 road_bbox['y_max'] - road_bbox['y_min']]
            )
            if iou > 0.1:
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

def view_result(result: Results, result_list_json):
    """可视化检测结果"""
    image = result.plot(labels=False, line_width=2)
    
    blind_road_boxes = []
    occupied_objects = []
    
    for res in result_list_json:
        class_name = res['class']
        
        if class_name == 'blind_road':
            class_color = CLASS_COLORS['blind_road']
            blind_road_boxes.append(res['bbox'])
        elif class_name in ['vehicle', 'obstacle', 'person']:
            class_color = CLASS_COLORS.get(class_name, (255, 255, 255))
            occupied_objects.append(res)
        else:
            class_color = (255, 255, 255)
        
        text = f"{class_name}: {res['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(image, (res['bbox']['x_min'], res['bbox']['y_min'] - text_height - baseline), 
                      (res['bbox']['x_min'] + text_width, res['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text, (res['bbox']['x_min'], res['bbox']['y_min'] - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
    # 检测盲道占用
    occupation_status = check_blind_road_occupation(blind_road_boxes, occupied_objects)
    
    # 在图片上显示占用状态
    if occupation_status['is_occupied']:
        status_text = f"⚠️ 盲道被占用! 占用物体: {occupation_status['occupied_count']}个"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    elif blind_road_boxes:
        status_text = "✅ 盲道畅通"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    return image, occupation_status

def main():
    # 标题
    st.title("🦯 盲道占用检测系统")
    st.markdown("---")
    
    # 侧边栏 - 模型选择
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 模型选择
        model_option = st.selectbox(
            "选择检测模型",
            ["blind_road_detector.pt (盲道专用)", "yolov8n.pt (通用)"],
            index=0
        )
        
        # 置信度阈值
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)
        
        st.markdown("---")
        st.info("""
        **使用说明：**
        1. 上传盲道场景图片
        2. 点击检测按钮
        3. 查看盲道占用情况
        
        **检测类别：**
        - 🟡 盲道 (blind_road)
        - 🔴 车辆 (vehicle)
        - 🔵 障碍物 (obstacle)
        - 🟢 行人 (person)
        """)
    
    # 主内容区
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 上传图片")
        image_file = st.file_uploader("选择盲道场景图片", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            st.image(image_file, caption="上传的图片", use_container_width=True)
    
    with col2:
        st.subheader("🔍 检测结果")
        
        if image_file is not None:
            if st.button("🚀 开始检测", use_container_width=True):
                with st.spinner("正在检测..."):
                    # 读取图片
                    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                    
                    # 加载模型
                    model_name = model_option.split()[0]
                    model = YOLO(model_name)
                    model.conf = conf_threshold
                    
                    # 推理
                    results = model(img)
                    result_list_json = result_to_json(results[0])
                    result_image, occupation_status = view_result(results[0], result_list_json)
                    
                    # 显示结果
                    st.image(result_image, caption="检测结果", channels="BGR", use_container_width=True)
                    
                    # 显示盲道占用状态
                    st.markdown("---")
                    st.subheader("🚦 盲道占用检测")
                    
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
        else:
            st.info("请先上传图片")
    
    # 快速测试
    st.markdown("---")
    st.subheader("🎬 快速测试")
    
    test_images_dir = "./blind_road_dataset/images/val"
    if os.path.exists(test_images_dir):
        test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            img_path = os.path.join(test_images_dir, test_images[0])
            col_prev, col_btn = st.columns([3, 1])
            
            with col_prev:
                st.image(img_path, caption=f"示例图片", use_container_width=True)
            
            with col_btn:
                if st.button("🔍 使用示例检测", use_container_width=True):
                    with st.spinner("正在检测..."):
                        img = cv2.imread(img_path)
                        model = YOLO("blind_road_detector.pt")
                        model.conf = conf_threshold
                        
                        results = model(img)
                        result_list_json = result_to_json(results[0])
                        result_image, occupation_status = view_result(results[0], result_list_json)
                        
                        st.image(result_image, caption="检测结果", channels="BGR", use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("🚦 盲道占用检测")
                        
                        col_t1, col_t2, col_t3 = st.columns(3)
                        
                        with col_t1:
                            if occupation_status['is_occupied']:
                                st.error(f"⚠️ 状态: 被占用")
                            else:
                                st.success(f"✅ 状态: 畅通")
                        
                        with col_t2:
                            st.metric("盲道数量", occupation_status['blind_road_count'])
                        
                        with col_t3:
                            st.metric("占用物体数", occupation_status['occupied_count'])
        else:
            st.warning("测试目录中没有图片")
    else:
        st.warning("测试目录不存在")

if __name__ == "__main__":
    main()
