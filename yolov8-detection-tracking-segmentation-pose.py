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

# Set page configuration
st.set_page_config(page_title="AI Visual Perception System", layout="wide", page_icon="🎯")

# Morandi color scheme for charts
MORANDI_COLORS = ['#B8A9C9', '#A3B8A8', '#D4B8B8', '#C4A88B', '#8BA88B', '#9DB5B2', '#D4C4A8', '#C9B8A8']

# Fixed camera display size
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360

# Function name to model file mapping
MODEL_MAPPING = {
    "Object Detection (Fast)": "yolov8n.pt",
    "Object Detection (Balanced)": "yolov8s.pt",
    "Instance Segmentation": "yolov8n-seg.pt",
    "Pose Estimation": "yolov8n-pose.pt",
}

# Function description
FUNCTION_DESC = {
    "Object Detection (Fast)": "Detect objects with bounding boxes. Fastest speed, suitable for real-time applications.",
    "Object Detection (Balanced)": "Detect objects with bounding boxes. Balanced speed and accuracy.",
    "Instance Segmentation": "Detect objects and generate pixel-level segmentation masks.",
    "Pose Estimation": "Detect human body keypoints and generate skeleton connections.",
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


def view_result(result: Results, result_list_json, centers=None):
    """Visualize result from ultralytics YOLOv8 prediction"""
    image = result.plot(labels=False, line_width=2)
    for res in result_list_json:
        class_color = COLORS[res['class_id'] % len(COLORS)]
        text = f"{res['class']} #{res['object_id']}: {res['confidence']:.2f}" if 'object_id' in res else f"{res['class']}: {res['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(image, (res['bbox']['x_min'], res['bbox']['y_min'] - text_height - baseline), 
                      (res['bbox']['x_min'] + text_width, res['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text, (res['bbox']['x_min'], res['bbox']['y_min'] - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if 'object_id' in res and centers is not None:
            centers[res['object_id']].append((int((res['bbox']['x_min'] + res['bbox']['x_max']) / 2), 
                                               int((res['bbox']['y_min'] + res['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[res['object_id']])):
                if centers[res['object_id']][j - 1] is None or centers[res['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image, centers[res['object_id']][j - 1], centers[res['object_id']][j], class_color, thickness)
    return image


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
    result_image = view_result(results[0], result_list_json, centers=centers)
    return result_image, result_list_json


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
    results_list = list(results)
    
    for i, result in enumerate(stqdm(results_list, desc=f"Processing video")):
        result_list_json = result_to_json(result, tracker=tracker)
        result_image = view_result(result, result_list_json, centers=centers)
        video_writer.write(result_image)
        all_results.extend(result_list_json)
    
    video_writer.release()
    
    # Write JSON file after processing all frames
    with open(result_video_json_file, 'w') as json_file:
        json.dump(all_results, json_file, indent=2)
    
    return video_file_name_out, result_video_json_file, all_results


def process_single_frame(frame, model, tracker=None, centers=None):
    """Process a single frame and return results"""
    results = model(frame)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image = view_result(results[0], result_list_json, centers=centers)
    return result_image, result_list_json


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
st.title("🎯 AI Visual Perception System")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    function_select = st.selectbox("📌 Select Function", function_list)
    
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
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **YOLOv8-based Visual Perception System**
    
    - 🚀 Object Detection
    - 🎯 Instance Segmentation  
    - 💃 Pose Estimation
    - 🔄 Object Tracking
    
    Built with YOLOv8 + DeepSORT
    """)

# Function to display statistics and charts
def display_statistics(result_list, title="Detection Statistics"):
    """Display statistics charts and table for given results"""
    if result_list and len(result_list) > 0:
        with st.expander("📊 View Detection Details", expanded=True):
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
tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Processing", "🎬 Video Processing", "📡 Live Stream", "📊 Statistics & Export"])

# Tab 1: Image Processing
with tab1:
    st.header("📷 Image Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        process_button = st.button("🚀 Process Image", use_container_width=True)
    
    if image_file is not None and process_button:
        with st.spinner("Processing..."):
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            img, result_list = image_processing(img, model)
            
            st.session_state.last_results = result_list
            st.session_state.last_image = img
            st.session_state.video_results = None
            st.session_state.video_path = None
            st.session_state.camera_results = None
            
            with col2:
                st.image(img, caption="Detection Result", channels="BGR", use_container_width=True)
    
    # Display statistics for image processing
    if st.session_state.last_results is not None and st.session_state.video_results is None:
        display_statistics(st.session_state.last_results)

# Tab 2: Video Processing
with tab2:
    st.header("🎬 Video Processing")
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
                
                video_file_out, result_video_json_file, all_results = video_processing(
                    video_file.name, model, tracker=tracker, centers=centers
                )
                
                # Store video results and path
                st.session_state.video_results = all_results
                st.session_state.video_path = video_file_out
                st.session_state.last_results = all_results
                st.session_state.camera_results = None
                
                # Display the processed video
                st.video(video_file_out)
                
                st.success(f"✅ Processing complete! Processed {len(all_results)} detections from video.")
            except Exception as e:
                st.error(f"❌ Video processing failed: {str(e)}")
            finally:
                if os.path.exists(video_file.name):
                    os.remove(video_file.name)
    
    # Display statistics for video processing
    if st.session_state.video_results is not None:
        display_statistics(st.session_state.video_results, "Video Detection Statistics")

# Tab 3: Live Stream
with tab3:
    st.header("📡 Live Stream Processing")
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
                _, result_list = process_single_frame(st.session_state.current_camera_frame, model)
                st.session_state.camera_results = result_list
                st.session_state.last_results = result_list
                st.success(f"✅ Captured and analyzed! Found {len(result_list)} objects.")
                
                # Display the captured frame with detections
                st.image(st.session_state.current_camera_frame, channels="BGR", width=CAMERA_WIDTH, caption="Captured Frame")
                
                # Display statistics for captured frame
                display_statistics(result_list, "Captured Frame Statistics")
        else:
            st.warning("Please start the camera first, then click Capture.")

# Tab 4: Statistics & Export
with tab4:
    st.header("📊 Statistics & Data Export")
    
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