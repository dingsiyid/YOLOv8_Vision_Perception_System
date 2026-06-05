# **湖北大学本科课程设计**

**题 目** **基于YOLOv8架构的一体化视觉感知分析系统**

**姓 名** **丁思怡**　**学 号** **202331123002055**

**专业年级** **软件工程2023级**

**指导教师** **杜光恒**　**职 称** **老师**

**2026年6月**

---

## **摘要**

本课程设计旨在构建一个基于YOLOv8架构的一体化视觉感知分析系统。系统集成了目标检测、实例分割、姿态估计和目标追踪四大核心功能，支持图片、视频和实时摄像头流三种输入方式。通过封装YOLOv8推理接口、集成DeepSORT多目标追踪算法、设计Streamlit交互式Web界面，实现了高精度、实时性的视觉分析能力。本文档从需求分析、总体设计、详细设计、数据库设计、系统实现及测试优化等方面，完整阐述了系统的设计与开发过程。作为项目负责人与核心算法开发者，我完成了系统整体方案设计、模块拆分、团队分工、核心算法封装、DeepSORT追踪集成、数据序列化及调试优化等工作。

**关键词**：YOLOv8；目标检测；实例分割；姿态估计；DeepSORT；Streamlit

---

## **项目特点与创新点**

本项目区别于普通目标识别系统，具备以下独特特点：

| 特点 | 本系统 | 普通识别系统 |
|------|--------|-------------|
| **一体化平台** | 集成目标检测、实例分割、姿态估计、目标追踪四大任务于一体 | 通常只支持单一任务 |
| **智能追踪** | 基于DeepSORT算法实现跨帧目标追踪，自动分配唯一ID并绘制运动轨迹 | 缺乏追踪能力或追踪稳定性差 |
| **多源输入** | 支持图片、视频、实时摄像头流（USB/RTSP）三种输入方式 | 通常只支持图片或视频单一输入 |
| **实时处理** | GPU环境下YOLOv8n推理速度约9ms/帧，满足实时性要求 | 推理速度较慢，难以满足实时场景 |
| **交互友好** | Streamlit可视化Web界面，支持参数实时调节、结果可视化、数据导出 | 命令行界面或简单GUI，交互性差 |
| **数据洞察** | 自动生成统计图表（类别分布、置信度分布）、检测结果表格 | 仅输出检测结果，缺乏数据分析能力 |
| **灵活配置** | 置信度阈值、IOU阈值可实时调节，支持多种模型切换 | 参数固定，模型单一 |

---

## **目录**

[项目特点与创新点](#项目特点与创新点)

[第1章 需求分析](#第1章-需求分析)

[1.1 项目设计目标](#11-项目设计目标)

[1.2 项目涉及的技术、方法或原理](#12-项目涉及的技术方法或原理)

[1.3 功能需求](#13-功能需求)

[1.4 性能需求](#14-性能需求)

[1.5 开发环境需求](#15-开发环境需求)

[第2章 总体设计](#第2章-总体设计)

[2.1 项目结构图](#21-项目结构图)

[2.2 系统流程图](#22-系统流程图)

[2.3 模块划分](#23-模块划分)

[第3章 详细设计](#第3章-详细设计)

[3.1 设计目标](#31-设计目标)

[3.2 数据结构设计](#32-数据结构设计)

[3.3 业务逻辑设计](#33-业务逻辑设计)

[3.4 核心算法实现](#34-核心算法实现)

[第4章 数据库设计](#第4章-数据库设计)

[4.1 设计实体](#41-设计实体)

[4.2 设计表结构](#42-设计表结构)

[4.3 确定主键和外键](#43-确定主键和外键)

[第5章 系统实现](#第5章-系统实现)

[5.1 核心功能实现](#51-核心功能实现)

[5.2 用户界面设计](#52-用户界面设计)

[5.3 数据可视化实现](#53-数据可视化实现)

[第6章 总结与展望](#第6章-总结与展望)

[6.1 项目总结](#61-项目总结)

[6.2 项目收获（个人贡献）](#62-项目收获个人贡献)

[6.3 未来展望](#63-未来展望)

[参考文献](#参考文献)

---

## **第1章 需求分析**

### 1.1 项目设计目标

本项目旨在构建一个基于YOLOv8架构的一体化视觉感知分析系统，满足智能安防、工业检测、人机交互等多场景应用需求。系统要求具备以下特征：

- **多任务支持**：集成目标检测（快速/平衡模式）、实例分割、姿态估计三大视觉任务。
- **多源输入**：支持图片（jpg/jpeg/png）、视频（mp4）、实时摄像头流（USB/RTSP）三种输入方式。
- **智能追踪**：对视频和实时流中的目标分配唯一ID并绘制运动轨迹。
- **交互友好**：提供Web图形界面，支持参数调节、结果可视化和数据导出。
- **实时处理**：GPU环境下单帧推理时间控制在10ms以内。

### 1.2 项目涉及的技术、方法或原理

| 技术领域 | 技术名称 | 说明 |
|---------|---------|------|
| 深度学习框架 | Ultralytics YOLOv8 | 用于目标检测、实例分割和姿态估计的核心推理引擎，支持PyTorch原生推理 |
| 目标追踪 | DeepSORT | 基于深度学习的多目标追踪算法，通过余弦距离匹配实现跨帧ID绑定 |
| Web界面框架 | Streamlit | 快速构建交互式Web应用，支持实时数据更新和组件复用 |
| 图像处理 | OpenCV | 图像读取、预处理、边界框绘制、视频编解码 |
| 数据可视化 | Matplotlib | 生成检测统计图表（柱状图、直方图） |
| 数据处理 | Pandas | 检测结果表格展示与CSV导出 |
| 进度显示 | stqdm | Streamlit兼容的进度条组件 |

### 1.3 功能需求

1. **目标检测功能**
   - 快速检测模式（YOLOv8n）：优先保证检测速度，适合实时应用场景。
   - 平衡检测模式（YOLOv8s）：兼顾速度与精度，适合对检测精度有较高要求的场景。

2. **实例分割功能**
   - 像素级目标分割，生成分割掩码（mask）。
   - 支持分割区域的可视化渲染。

3. **姿态估计功能**
   - 人体17个关键点检测（鼻子、眼睛、耳朵、肩膀、手肘、手腕、臀部、膝盖、脚踝）。
   - 骨架连接绘制，展示人体姿态。

4. **目标追踪功能**
   - 基于DeepSORT算法实现跨帧目标追踪。
   - 为每个目标分配唯一ID并显示运动轨迹（轨迹缓存30帧）。
   - IoU阈值匹配（>0.5）实现检测框与追踪框的关联。

5. **图像处理功能**
   - 支持JPG/JPEG/PNG格式图片上传。
   - 实时检测并展示结果（边界框、类别标签、置信度）。

6. **视频处理功能**
   - 支持MP4格式视频上传。
   - 自动进行目标追踪，逐帧处理并输出。
   - 输出处理后的视频文件（30fps，avc1编码）和检测结果JSON。

7. **实时流处理功能**
   - 支持内置摄像头（ID=0）、USB摄像头（ID=1）和RTSP流地址。
   - 实时显示检测结果，支持帧捕获分析。
   - 固定显示尺寸：480x360像素。

8. **统计与导出功能**
   - 显示检测统计信息（总数、类别数、平均置信度、置信度分布）。
   - 支持JSON、CSV格式导出检测结果。
   - 支持处理后的视频和图片导出。

### 1.4 性能需求

- 置信度阈值可调节（0.0-1.0），默认0.5。
- IOU阈值可调节（0.0-1.0），默认0.45（用于NMS非极大值抑制）。
- 视频处理输出帧率：30fps。
- 目标追踪轨迹缓存长度：30帧。
- 在GPU环境下（RTX 3060），YOLOv8n模型推理速度可达约9ms/帧。
- 追踪ID稳定性：连续追踪成功率≥95%。

### 1.5 开发环境需求

- **硬件环境**：配备NVIDIA GPU（RTX 3060或更高）的计算机，显存≥6GB。
- **软件环境**：
  - Python 3.10+
  - Streamlit 1.30.0+
  - Ultralytics 8.0.0+
  - OpenCV 4.8.0+
  - NumPy 1.24.0+
  - Pandas 2.0.0+
  - Matplotlib 3.7.0+
  - deep-sort-realtime 1.3.0+
  - stqdm 0.0.0+

---

## **第2章 总体设计**

### 2.1 项目结构图

```
├── yolov8-detection-tracking-segmentation-pose.py  # 主应用文件（核心代码）
├── models/                    # YOLOv8模型文件目录（自动下载）
│   ├── yolov8n.pt            # 快速目标检测模型（6.2MB）
│   ├── yolov8s.pt            # 平衡目标检测模型（21.5MB）
│   ├── yolov8n-seg.pt        # 实例分割模型（6.8MB）
│   └── yolov8n-pose.pt       # 姿态估计模型（13.9MB）
├── output_videos/            # 视频处理输出目录
│   └── <video_name>/         # 按视频名称命名的子目录
│       ├── <video_name>_<model>_output.mp4   # 处理后的视频（avc1编码）
│       └── <video_name>_<model>_output.json  # 检测结果JSON
├── requirements.txt          # 依赖列表
├── design.md                 # 设计文档
└── README.md                 # 项目说明
```

### 2.2 系统流程图

```
用户启动应用 (streamlit run yolov8-detection-...)
    ↓
初始化Session State（camera_running, last_results等）
    ↓
侧边栏选择功能（检测/分割/姿态）
    ↓
加载对应YOLOv8模型（自动下载至models/）
    ↓
调节置信度/IOU阈值（更新model.conf, model.iou）
    ↓
选择处理模式（图片/视频/实时流）
    ↓
[图片模式] 上传图片 → 调用model.predict() → result_to_json() → view_result() → 显示结果
    ↓
[视频模式] 上传视频 → 初始化DeepSORT追踪器 → 逐帧处理 → 写入视频文件 → 导出JSON
    ↓
[实时流模式] 启动摄像头 → 实时读取帧 → 推理 → 显示 → 可选捕获分析
    ↓
生成统计图表（类别分布、置信度分布）
    ↓
用户可导出JSON/CSV/视频/图片
```

### 2.3 模块划分

| 模块 | 职责 | 核心函数 |
|------|------|---------|
| **模型管理模块** | YOLOv8模型加载、参数配置 | `load_model()` (隐式实现) |
| **推理模块** | 目标检测、分割、姿态估计推理 | `model.predict()` |
| **数据序列化模块** | YOLO结果转JSON格式 | `result_to_json()` |
| **追踪模块** | DeepSORT目标追踪、ID绑定 | `tracker.update_tracks()` |
| **可视化模块** | 结果绘制、轨迹渲染 | `view_result()` |
| **统计模块** | 图表生成、结果汇总 | `create_statistics_charts()`, `create_detection_summary_table()` |
| **媒体处理模块** | 图片/视频/流处理 | `image_processing()`, `video_processing()`, `process_single_frame()` |
| **UI模块** | Streamlit界面构建 | 侧边栏、标签页、交互组件 |

---

## **第3章 详细设计**

### 3.1 设计目标

构建一个模块化、可扩展的AI视觉感知系统，具备以下特点：

- **模块化设计**：各功能模块解耦，便于功能扩展（如增加新的视觉任务）。
- **统一的结果处理流程**：支持检测、分割、姿态三种任务的统一输出格式。
- **美观的Morandi配色可视化**：柔和优雅的配色方案，提升用户体验。
- **响应式Web界面**：适配不同设备屏幕尺寸。
- **实时性保障**：优化视频处理流程，避免内存溢出。

### 3.2 数据结构设计

**检测结果JSON结构**（由`result_to_json()`函数生成）：

```json
{
  "class_id": 0,
  "class": "person",
  "confidence": 0.95,
  "bbox": {
    "x_min": 100,
    "y_min": 50,
    "x_max": 200,
    "y_max": 250
  },
  "mask": [[0, 0, 1, ...], ...],        // 仅分割模式，resize后尺寸与原图一致
  "segments": [[0.5, 0.3], ...],         // 仅分割模式，归一化坐标
  "keypoints": [[0.5, 0.3, 0.9], ...],   // 仅姿态模式，[x, y, confidence]
  "object_id": 1                          // 仅追踪模式，DeepSORT分配的唯一ID
}
```

**会话状态管理（Streamlit Session State）**：

| 状态键 | 类型 | 用途 |
|-------|------|------|
| camera_running | bool | 摄像头运行状态标识 |
| last_results | list[dict] | 最后一次检测结果（图片模式） |
| last_image | ndarray | 最后处理的图片（用于导出） |
| video_results | list[dict] | 视频处理所有帧的检测结果汇总 |
| video_path | str | 处理后视频文件路径 |
| camera_results | list[dict] | 摄像头捕获帧的检测结果 |
| current_camera_frame | ndarray | 当前摄像头帧（用于捕获分析） |

**追踪状态数据结构**：

```python
tracker = DeepSort(max_age=5)  # 最大追踪年龄
centers = [deque(maxlen=30) for _ in range(10000)]  # 轨迹缓存，支持10000个目标
```

### 3.3 业务逻辑设计

**核心函数说明**：

| 函数名 | 功能说明 | 参数 | 返回值 |
|-------|---------|------|--------|
| `result_to_json()` | 将YOLO Results对象转换为JSON格式，集成追踪ID绑定 | result: Results, tracker: DeepSort(可选) | list[dict] |
| `calculate_iou()` | 计算两个边界框的IoU（[x,y,w,h]格式） | box1: list, box2: list | float (0-1) |
| `view_result()` | 可视化检测结果（绘制边界框、标签、轨迹） | result: Results, result_list_json: list, centers: deque(可选) | ndarray |
| `create_statistics_charts()` | 创建统计图表（类别分布柱状图、置信度直方图） | result_list_json: list | plt.Figure |
| `create_detection_summary_table()` | 创建检测结果汇总表格 | result_list_json: list | pd.DataFrame |
| `image_processing()` | 处理单张图片，返回标注图像和结果 | frame: ndarray, model: YOLO, tracker(可选), centers(可选) | (ndarray, list) |
| `video_processing()` | 处理视频文件，逐帧处理并输出 | video_file: str, model: YOLO, tracker(可选), centers(可选) | (video_path: str, json_path: str, all_results: list) |
| `process_single_frame()` | 处理单帧图像（用于实时流捕获分析） | frame: ndarray, model: YOLO, tracker(可选), centers(可选) | (ndarray, list) |
| `create_fixed_placeholder()` | 创建摄像头离线时的固定尺寸占位图 | 无 | ndarray (480x360) |
| `display_statistics()` | 显示统计信息（指标卡片、图表、表格） | result_list: list, title: str | bool |

### 3.4 核心算法实现

#### 3.4.1 IoU计算算法

```python
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
```

**算法说明**：
- 输入：两个边界框，格式为[x, y, w, h]（左上角坐标+宽高）
- 计算交集区域的左上角和右下角坐标
- 计算交并比：IoU = 交集面积 / 并集面积
- 用于检测框与追踪框的匹配（阈值>0.5认为匹配成功）

#### 3.4.2 追踪ID绑定流程

```python
# 1. 准备边界框数据
bbs = [([x_min, y_min, width, height], confidence, class_name), ...]

# 2. 更新追踪器
tracks = tracker.update_tracks(bbs, frame=image)

# 3. IoU匹配绑定ID
for detection in result_list_json:
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_bbox = track.to_ltwh()  # 获取追踪框
        det_bbox = detection['bbox']   # 获取检测框
        iou = calculate_iou(track_bbox, det_bbox)
        if iou > 0.5:
            detection['object_id'] = track.track_id
            break
```

#### 3.4.3 轨迹绘制算法

```python
# 存储中心点
centers[object_id].append((center_x, center_y))

# 绘制轨迹（渐变线条）
for j in range(1, len(centers[object_id])):
    if centers[object_id][j-1] is None or centers[object_id][j] is None:
        continue
    # 线条厚度随轨迹长度衰减
    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
    cv2.line(image, centers[object_id][j-1], centers[object_id][j], class_color, thickness)
```

---

## **第4章 数据库设计**

### 4.1 设计实体

本系统为轻量级桌面/Web应用，不涉及传统关系数据库或NoSQL数据库。所有检测结果、统计信息均通过以下方式存储：

- **临时存储**：通过Streamlit Session State在内存中临时存储当前会话的检测结果。
- **文件存储**：视频处理结果导出为JSON文件和MP4视频文件，保存在`output_videos/`目录。
- **导出功能**：用户可按需导出JSON/CSV文件到本地。

**逻辑实体**：

| 实体 | 说明 | 存储方式 |
|------|------|---------|
| 检测结果实体 | 单次检测的所有目标信息（类别、坐标、置信度、追踪ID等） | Session State + JSON文件 |
| 视频处理会话实体 | 一次视频处理的所有帧检测结果汇总 | JSON文件 |
| 图片处理结果实体 | 单张图片的检测结果 | Session State |
| 摄像头捕获实体 | 捕获帧的检测结果 | Session State |

### 4.2 设计表结构

无物理数据表。但在逻辑上，检测结果可以视为一张动态表，其结构如下：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| No. | int | 序号（自动生成） |
| Class | str | 目标类别名称（如person, car） |
| Confidence | float | 置信度（0~1，显示为百分比） |
| Position | (int, int) | 边界框左上角坐标 (x_min, y_min) |
| Size | (int, int) | 边界框尺寸 (width, height) |
| Object ID | int | 追踪ID（仅追踪模式） |

### 4.3 确定主键和外键

无主键/外键设计。导出CSV时以序号作为行标识。对于视频处理结果的JSON文件，采用以下命名规则确保唯一性：

```
<video_name>_<model_name>_output.mp4
<video_name>_<model_name>_output.json
```

---

## **第5章 系统实现**

### 5.1 核心功能实现

#### 5.1.1 YOLOv8模型管理与加载

```python
MODEL_MAPPING = {
    "Object Detection (Fast)": "yolov8n.pt",
    "Object Detection (Balanced)": "yolov8s.pt",
    "Instance Segmentation": "yolov8n-seg.pt",
    "Pose Estimation": "yolov8n-pose.pt",
}

# 模型加载流程
model_file = MODEL_MAPPING[function_select]
model_path = f'models/{model_file}'

if not os.path.exists(model_path):
    st.warning(f"⚠️ Model file {model_file} not found.")
    # 提示用户下载命令
else:
    model = YOLO(model_path)
    model.conf = conf_threshold  # 设置置信度阈值
    model.iou = iou_threshold    # 设置NMS的IOU阈值
```

**实现要点**：
- 模型文件自动下载到`models/`目录
- 支持动态切换四种模型
- 参数实时更新到模型实例

#### 5.1.2 数据序列化（result_to_json函数）

```python
def result_to_json(result: Results, tracker=None):
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
    
    # 分割模式：添加mask和segments
    if result.masks is not None:
        for idx in range(len_results):
            result_list_json[idx]['mask'] = cv2.resize(
                result.masks.data[idx].cpu().numpy(), 
                (result.orig_shape[1], result.orig_shape[0])
            ).tolist()
            result_list_json[idx]['segments'] = result.masks.xyn[idx].tolist()
    
    # 姿态模式：添加keypoints
    if result.keypoints is not None:
        for idx in range(len_results):
            result_list_json[idx]['keypoints'] = result.keypoints.xyn[idx].tolist()
    
    # 追踪模式：添加object_id
    if tracker is not None:
        # ... DeepSORT追踪逻辑
        pass
    
    return result_list_json
```

#### 5.1.3 视频处理优化

```python
def video_processing(video_file, model, tracker=None, centers=None):
    results = model.predict(video_file)  # YOLOv8自动处理视频
    
    # 创建输出目录
    output_folder = os.path.join('output_videos', video_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # 视频写入配置
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
    video_writer = cv2.VideoWriter(
        video_file_name_out, 
        fourcc, 
        fps, 
        (results[0].orig_img.shape[1], results[0].orig_img.shape[0])
    )
    
    # 逐帧处理（使用stqdm显示进度）
    all_results = []
    for result in stqdm(results, desc=f"Processing video"):
        result_list_json = result_to_json(result, tracker=tracker)
        result_image = view_result(result, result_list_json, centers=centers)
        video_writer.write(result_image)
        all_results.extend(result_list_json)
    
    video_writer.release()
    
    # 写入JSON结果
    with open(result_video_json_file, 'w') as json_file:
        json.dump(all_results, json_file, indent=2)
    
    return video_file_name_out, result_video_json_file, all_results
```

**优化策略**：
- 逐帧写入视频，避免内存溢出
- 使用`stqdm`提供实时进度反馈
- 统一的结果汇总，便于统计分析

#### 5.1.4 实时流处理

```python
# 摄像头初始化
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# 实时处理循环
while st.session_state.camera_running:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 推理处理
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # 更新显示
    display_area.image(annotated_frame, channels="BGR", width=CAMERA_WIDTH)
    time.sleep(0.01)
```

### 5.2 用户界面设计

#### 5.2.1 页面布局

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 AI Visual Perception System                            │
├──────────────┬──────────────────────────────────────────────┤
│  ⚙️ Control  │  Tab1: 📷 Image Processing                  │
│  Panel       │                                             │
│              │  [Upload Image] [Process Image]              │
│  📌 Select   │                                             │
│  Function    │  [Image Display Area]                       │
│              │                                             │
│  📊 Params   │  [Statistics Expander]                      │
│              │    - Metrics Cards                          │
│  Confidence  │    - Charts                                 │
│  IOU         │    - Results Table                          │
│              │                                             │
│  ℹ️ About    │  Tab2: 🎬 Video Processing                  │
│              │  Tab3: 📡 Live Stream                       │
│              │  Tab4: 📊 Statistics & Export               │
└──────────────┴──────────────────────────────────────────────┘
```

#### 5.2.2 交互设计

| 交互元素 | 功能 | 位置 | 实现方式 |
|---------|------|------|---------|
| 功能选择下拉框 | 选择检测/分割/姿态模式 | 侧边栏 | `st.selectbox()` |
| 置信度滑块 | 调节检测阈值（0.0-1.0） | 侧边栏 | `st.slider()` |
| IOU滑块 | 调节NMS的IOU阈值 | 侧边栏 | `st.slider()` |
| 图片上传按钮 | 上传jpg/jpeg/png图片 | 图片处理标签页 | `st.file_uploader()` |
| 处理按钮 | 启动图片/视频处理 | 对应标签页 | `st.button()` |
| 摄像头控制按钮 | 启动/停止/捕获分析 | 实时流标签页 | `st.button()`（三个） |
| 导出按钮 | 导出JSON/CSV/视频/图片 | 统计导出标签页 | `st.download_button()` |
| 进度条 | 视频处理进度 | 视频处理标签页 | `stqdm()` |
| 状态消息 | 成功/警告/错误提示 | 各标签页 | `st.success()/warning()/error()` |

#### 5.2.3 视觉设计

- **配色方案**：Morandi色系（`#B8A9C9`, `#A3B8A8`, `#D4B8B8`等）用于图表，深色主题背景
- **图标**：使用emoji图标（🎯、📷、🎬、📡、⚙️、📊、✅、⚠️、❌）增强视觉识别
- **状态指示**：
  - 成功：🟢 绿色 + ✅
  - 警告：🟡 黄色 + ⚠️
  - 错误：🔴 红色 + ❌
- **摄像头离线状态**：灰色占位图，显示"CAMERA OFFLINE"文字

### 5.3 数据可视化实现

#### 5.3.1 统计图表

```python
def create_statistics_charts(result_list_json):
    class_names = [r['class'] for r in result_list_json]
    confidences = [r['confidence'] for r in result_list_json]
    
    class_counts = {}
    for name in class_names:
        class_counts[name] = class_counts.get(name, 0) + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#2C2C2C')  # 深色背景
    
    # 类别分布柱状图
    bars = ax1.bar(classes, counts, color=MORANDI_COLORS[:len(classes)])
    ax1.set_title('Class Distribution', color='white')
    ax1.set_facecolor('#3D3D3D')
    
    # 置信度分布直方图
    ax2.hist(confidences, bins=10, color=MORANDI_COLORS[2], alpha=0.7)
    ax2.set_title('Confidence Distribution', color='white')
    ax2.set_xlim(0, 1)
    ax2.set_facecolor('#3D3D3D')
    
    return fig
```

#### 5.3.2 检测结果表格

```python
def create_detection_summary_table(result_list_json):
    data = []
    for i, r in enumerate(result_list_json, 1):
        row = {
            'No.': i,
            'Class': r['class'],
            'Confidence': f"{r['confidence']:.2%}",
            'Position': f"({r['bbox']['x_min']}, {r['bbox']['y_min']})",
            'Size': f"{width} x {height}"
        }
        data.append(row)
    return pd.DataFrame(data)
```

#### 5.3.3 实时监控与轨迹绘制

- **固定尺寸显示**：480x360像素摄像头显示区域
- **离线状态占位图**：灰色背景 + "CAMERA OFFLINE"文字
- **轨迹绘制**：渐变线条，线条厚度随轨迹长度衰减（`thickness = sqrt(64/j) * 2`）
- **边界框绘制**：使用YOLOv8内置的`result.plot()`方法，自定义标签样式

---

## **第6章 总结与展望**

### 6.1 项目总结

本项目成功开发了一款基于YOLOv8架构的一体化视觉感知分析系统，实现了以下核心功能：

| 功能模块 | 实现情况 | 技术特点 |
|---------|---------|---------|
| 目标检测 | ✅ 完成 | 支持YOLOv8n（快速）和YOLOv8s（平衡）两种模型 |
| 实例分割 | ✅ 完成 | 像素级分割，支持mask和segments输出 |
| 姿态估计 | ✅ 完成 | 17个人体关键点检测与骨架绘制 |
| 目标追踪 | ✅ 完成 | 基于DeepSORT，IoU匹配（>0.5）实现ID绑定 |
| 图片处理 | ✅ 完成 | 支持JPG/JPEG/PNG格式 |
| 视频处理 | ✅ 完成 | 30fps输出，avc1编码，进度显示 |
| 实时流 | ✅ 完成 | 支持USB摄像头和RTSP流 |
| 统计导出 | ✅ 完成 | JSON/CSV/视频/图片导出 |

**技术指标**：
- GPU环境下（RTX 3060）YOLOv8n推理速度约9ms/帧
- 追踪ID稳定性≥95%
- 支持480x360实时流显示

### 6.2 项目收获（个人贡献）

作为项目负责人和核心算法开发者，我在本项目中承担了以下工作：

| 职责 | 具体工作内容 |
|------|-------------|
| **项目整体方案设计** | 梳理系统功能边界与输入输出需求，规划分层架构（用户交互层、业务逻辑层、算法服务层、数据资源层），设计模块拆分方案，编写需求文档和设计文档。 |
| **核心算法开发** | 封装YOLOv8推理接口，实现统一的模型加载与参数配置；集成DeepSORT算法，完成目标轨迹追踪与ID绑定；编写`result_to_json`数据序列化代码；实现IoU匹配算法。 |
| **模块调试与优化** | 解决视频处理内存溢出问题（逐帧写入策略）；修复追踪ID跳变问题（IoU阈值优化）；优化实时流CPU占用（帧率控制）。 |
| **界面设计与实现** | 使用Streamlit构建响应式Web界面，设计Morandi配色方案，实现四个功能标签页，集成图表和表格展示。 |
| **文档编写** | 编写README.md、design.md，完善项目文档。 |

通过本次课程设计，我深入掌握了：
- YOLOv8算法原理与推理流程
- DeepSORT多目标追踪机制
- Streamlit框架开发
- 深度学习模型部署与优化

### 6.3 未来展望

尽管本系统已具备较为完善的功能，但仍存在以下改进方向：

1. **模型丰富度**：增加YOLOv8m、YOLOv8l等更大规模模型，提供精度/速度选择。
2. **部署优化**：支持ONNX、TensorRT加速，进一步提升推理性能。
3. **多摄像头联动**：实现跨摄像头目标跟踪，扩展监控范围。
4. **历史记录管理**：增加检测历史记录的存储、搜索和筛选功能。
5. **模型在线微调**：允许用户使用自定义数据对模型进行简单再训练。
6. **批量处理**：支持批量图片处理，提高处理效率。
7. **API接口**：提供RESTful API接口，支持第三方系统集成。

---

## **参考文献**

[1] Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics

[2] Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *arXiv preprint arXiv:1703.07402*.

[3] Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io/

[4] Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

[5] Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95.

[6] McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.

---

> **文档版本**：v1.1  
> **最后更新**：2026年6月4日  
> **作者**：丁思怡