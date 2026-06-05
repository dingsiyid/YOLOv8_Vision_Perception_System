# 🎯 YOLOv8 AI视觉感知系统

基于YOLOv8构建的综合计算机视觉应用，支持目标检测、实例分割、姿态估计和目标追踪功能。

## ✨ 功能特性

| 功能 | 描述 | 模型 |
|------|------|------|
| 🚀 **目标检测 (Fast)** | 快速目标检测，适合实时应用 | YOLOv8n |
| ⚡ **目标检测 (Balanced)** | 平衡速度与精度的目标检测 | YOLOv8s |
| 🎯 **实例分割** | 像素级目标分割 | YOLOv8n-seg |
| 💃 **姿态估计** | 人体关键点检测与骨架连接 | YOLOv8n-pose |
| 🔄 **目标追踪** | 基于DeepSORT的跨帧目标追踪 | DeepSORT |

## 🛠️ 技术栈

- **深度学习框架**: Ultralytics YOLOv8
- **Web界面**: Streamlit
- **目标追踪**: DeepSORT
- **图像处理**: OpenCV
- **数据可视化**: Matplotlib
- **数据处理**: Pandas

## 📦 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

模型会在首次使用时自动下载，也可以提前下载：

```bash
# 下载所有模型
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt'); YOLO('yolov8n-seg.pt'); YOLO('yolov8n-pose.pt')"
```

模型要保存到 `models/` 目录。

## 🚀 运行应用

```bash
streamlit run yolov8-detection-tracking-segmentation-pose.py
```

应用启动后，在浏览器中访问显示的URL（通常是 `http://localhost:8501`）。

## 📖 使用说明

### 1. 选择功能

在左侧控制面板中选择要使用的功能：
- **Object Detection (Fast)**: 快速目标检测
- **Object Detection (Balanced)**: 平衡性能的目标检测
- **Instance Segmentation**: 实例分割
- **Pose Estimation**: 姿态估计

### 2. 调整参数

- **Confidence Threshold**: 置信度阈值（0.0-1.0），过滤低置信度检测结果
- **IOU Threshold**: IOU阈值（0.0-1.0），用于NMS（非极大值抑制）

### 3. 图片处理

1. 切换到 **Image Processing** 标签
2. 上传图片（支持 jpg, jpeg, png）
3. 点击 **Process Image** 按钮
4. 查看检测结果和统计信息

### 4. 视频处理

1. 切换到 **Video Processing** 标签
2. 上传视频（支持 mp4）
3. 点击 **Process Video** 按钮
4. 等待处理完成后查看结果视频和统计信息

> **注意**: 视频处理会自动启用目标追踪功能，每个目标会分配唯一ID并显示运动轨迹。

### 5. 实时流处理

1. 切换到 **Live Stream** 标签
2. 设置摄像头源（0=内置摄像头，1=USB摄像头，或RTSP地址）
3. 点击 **START CAMERA** 开始实时流
4. 点击 **CAPTURE & ANALYZE** 捕获并分析当前帧
5. 点击 **STOP CAMERA** 停止流

### 6. 统计与导出

1. 切换到 **Statistics & Export** 标签
2. 查看检测统计信息（总数、类别、置信度分布等）
3. 导出结果为 JSON、CSV 或视频文件

## 📁 项目结构

```
.
├── models/                    # YOLOv8模型文件目录
│   ├── yolov8n.pt            # 快速目标检测模型
│   ├── yolov8s.pt            # 平衡目标检测模型
│   ├── yolov8n-seg.pt        # 实例分割模型
│   └── yolov8n-pose.pt       # 姿态估计模型
├── output_videos/            # 视频处理输出目录
│   └── <video_name>/         # 按视频名称命名的子目录
│       ├── <video_name>_<model>_output.mp4   # 处理后的视频
│       └── <video_name>_<model>_output.json  # 检测结果JSON
├── yolov8-detection-tracking-segmentation-pose.py  # 主应用文件
├── requirements.txt          # 依赖列表
├── Dockerfile                # Docker配置
└── README.md                 # 项目说明
```

## 🔧 配置说明

### 摄像头配置

- **内置摄像头**: `0`
- **USB摄像头**: `1`（或更高数字）
- **RTSP流**: `rtsp://username:password@ip:port/stream`

### 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **Confidence Threshold** | 0.5 | 0.0-1.0 | 检测置信度阈值，低于此值的检测结果会被过滤 |
| **IOU Threshold** | 0.45 | 0.0-1.0 | NMS的IOU阈值，用于去除冗余检测框 |

## 📊 输出格式

### JSON输出格式

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "source": "Video Processing",
  "total_detections": 150,
  "results": [
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
      "object_id": 1,
      "keypoints": [...]  // 姿态估计时包含
    }
  ]
}
```

### CSV输出格式

| No. | Class | Confidence | Position | Size |
|-----|-------|------------|----------|------|
| 1 | person | 95.00% | (100, 50) | 100 x 200 |

## 🐳 Docker部署

```bash
# 构建镜像
docker build -t yolov8-vision .

# 运行容器
docker run -p 8501:8501 yolov8-vision
```

## � 团队成员

| 姓名 | 职责 | 主要贡献 |
|------|------|----------|
| **丁思怡** | 项目整体方案设计与核心算法开发 | 梳理系统整体功能边界与输入输出需求，规划系统分层架构与模块拆分方案，统筹团队整体分工。封装YOLO目标检测、实例分割、姿态估计三大推理接口，接入DeepSORT算法实现目标轨迹追踪与ID绑定，编写数据序列化代码，独立完成各算法模块调试与问题修复。 |
| **师睿荥** | 技术调研与开发环境搭建 | 系统调研YOLOv8多模型、DeepSORT目标追踪、Streamlit界面开发核心技术，确定整体技术路线。搭建Python3.10虚拟开发环境，安装全部项目依赖并生成配置文件，下载各类预训练权重，完成模型推理测试，搭建完整可用的项目运行环境。 |
| **刘思琦** | Web界面搭建与系统功能集成 | 设计并搭建Streamlit页面整体布局，完成图片上传、在线推理功能对接。实现视频批量处理与摄像头实时流检测功能，集成数据统计图表展示、结果导出等功能，完成前端界面与后端算法的整体联调与功能整合。 |
| **蔡依想** | 系统测试优化与文档整理 | 对系统所有功能进行全面黑盒测试与性能测速，优化模型推理速度与内存占用问题。统一规范代码格式，补充代码注释，修复项目遗留Bug，撰写完整测试报告，完善项目README文档，规整全部项目资料。 |

## �📝 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue

---

**注意**: 首次运行时，模型文件会自动下载，可能需要一些时间，请确保网络连接正常。