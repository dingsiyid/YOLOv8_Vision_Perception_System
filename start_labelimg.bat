@echo off
echo ============================================================
echo 启动LabelImg - 盲道数据集标注工具
echo ============================================================
echo.
echo 快捷键说明:
echo   W - 创建边界框
echo   D - 下一张图片
echo   A - 上一张图片
echo   Ctrl+S - 保存标签
echo   Del - 删除边界框
echo   Ctrl+C - 复制边界框
echo   Ctrl+V - 粘贴边界框
echo.

:: 设置工作目录
cd /d "D:\working directory\Uhmw\25-26(2)\shixun\YOLOv8-Object-Detection-Tracking-Image-Segmentation-Pose-Estimation"

:: 激活虚拟环境（如果需要）
:: call venv\Scripts\activate

:: 启动LabelImg，设置默认打开目录和保存目录
python -m labelImg "blind_road_dataset\images\train" "blind_road_dataset\labels\train"

pause