"""
Performance Testing Tool for YOLOv8
Measures inference speed, memory usage, and FPS for different models
"""

import os
import time
import cv2
import numpy as np
import psutil
from ultralytics import YOLO

# ==================== CONFIGURATION ====================

MODEL_PATHS = {
    "yolov8n": "models/yolov8n.pt",
    "yolov8n-seg": "models/yolov8n-seg.pt",
    "yolov8n-pose": "models/yolov8n-pose.pt",
}

TEST_IMAGE_SIZE = (640, 640)
TEST_ITERATIONS = 50
WARMUP_ITERATIONS = 5
BATCH_SIZES = [1, 2, 4, 8, 16]

# Color codes for terminal output
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'
COLOR_RESET = '\033[0m'


# ==================== UTILITY FUNCTIONS ====================

def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{COLOR_BLUE}{'='*60}{COLOR_RESET}")
    print(f"{COLOR_BLUE}📊 {title}{COLOR_RESET}")
    print(f"{COLOR_BLUE}{'='*60}{COLOR_RESET}")


def print_result(label, value, unit=""):
    """Print a formatted result line"""
    print(f"  {label:30s}: {value:.2f} {unit}")


# ==================== PERFORMANCE TESTS ====================

def test_inference_speed(model, test_image, iterations=TEST_ITERATIONS):
    """
    Test model inference speed
    
    Args:
        model: YOLO model instance
        test_image: Input image as numpy array
        iterations: Number of test iterations
    
    Returns:
        dict: Speed test results
    """
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = model(test_image, verbose=False)
    
    # Test
    times = []
    for _ in range(iterations):
        start = time.time()
        results = model(test_image, verbose=False)
        end = time.time()
        times.append((end - start) * 1000)
    
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "fps": 1000 / (sum(times) / len(times)),
    }


def test_memory_usage(model, test_image, iterations=20):
    """
    Test model memory usage
    
    Args:
        model: YOLO model instance
        test_image: Input image as numpy array
        iterations: Number of test iterations
    
    Returns:
        dict: Memory test results
    """
    mem_before = get_memory_usage_mb()
    mem_peaks = []
    
    for _ in range(iterations):
        _ = model(test_image, verbose=False)
        mem_peak = get_memory_usage_mb()
        mem_peaks.append(mem_peak)
    
    mem_after = get_memory_usage_mb()
    
    return {
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "mem_peak_mb": max(mem_peaks),
        "mem_increase_mb": max(mem_peaks) - mem_before,
    }


def test_batch_inference(model, batch_sizes=BATCH_SIZES):
    """
    Test batch inference performance
    
    Args:
        model: YOLO model instance
        batch_sizes: List of batch sizes to test
    
    Returns:
        dict: Batch inference results
    """
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch of random images
        batch_images = [
            np.random.randint(0, 255, (*TEST_IMAGE_SIZE, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        
        # Warmup
        _ = model(batch_images, verbose=False)
        
        # Test
        start = time.time()
        _ = model(batch_images, verbose=False)
        end = time.time()
        
        total_ms = (end - start) * 1000
        per_image_ms = total_ms / batch_size
        
        results[batch_size] = {
            "total_ms": total_ms,
            "per_image_ms": per_image_ms,
            "fps": 1000 / per_image_ms,
        }
    
    return results


def test_video_processing(video_path, model, max_frames=200, frame_skip=1):
    """
    Test video processing performance
    
    Args:
        video_path: Path to video file
        model: YOLO model instance
        max_frames: Maximum number of frames to process
        frame_skip: Process every Nth frame
    
    Returns:
        dict: Video processing results
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    frame_count = 0
    process_times = []
    mem_samples = []
    
    start_time = time.time()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            mem_before = get_memory_usage_mb()
            
            frame_start = time.time()
            results = model(frame, verbose=False)
            frame_end = time.time()
            
            mem_after = get_memory_usage_mb()
            
            process_times.append((frame_end - frame_start) * 1000)
            mem_samples.append(mem_after)
        
        frame_count += 1
    
    cap.release()
    total_time = time.time() - start_time
    
    if not process_times:
        return None
    
    return {
        "frames_processed": frame_count,
        "total_time_s": total_time,
        "avg_frame_ms": sum(process_times) / len(process_times),
        "fps": frame_count / total_time,
        "peak_memory_mb": max(mem_samples) if mem_samples else 0,
    }


# ==================== MAIN FUNCTION ====================

def main():
    """Main test runner"""
    print(f"\n{COLOR_GREEN}🚀 YOLOv8 Performance Testing Tool{COLOR_RESET}")
    print(f"Test iterations: {TEST_ITERATIONS} (averaged)")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    
    # Prepare test image
    test_image = np.random.randint(0, 255, (*TEST_IMAGE_SIZE, 3), dtype=np.uint8)
    
    # Store all results
    all_speed_results = {}
    all_memory_results = {}
    all_batch_results = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\n{COLOR_YELLOW}⚠️ Model not found: {model_path}{COLOR_RESET}")
            continue
        
        print(f"\n{COLOR_GREEN}📦 Loading model: {model_name}{COLOR_RESET}")
        
        try:
            model = YOLO(model_path)
            
            # Test 1: Inference Speed
            print_section(f"{model_name} - Inference Speed")
            speed_results = test_inference_speed(model, test_image)
            print_result("Average inference time", speed_results["avg_ms"], "ms")
            print_result("Fastest inference", speed_results["min_ms"], "ms")
            print_result("Slowest inference", speed_results["max_ms"], "ms")
            print_result("FPS", speed_results["fps"], "fps")
            all_speed_results[model_name] = speed_results
            
            # Test 2: Memory Usage
            print_section(f"{model_name} - Memory Usage")
            memory_results = test_memory_usage(model, test_image)
            print_result("Memory before test", memory_results["mem_before_mb"], "MB")
            print_result("Memory after test", memory_results["mem_after_mb"], "MB")
            print_result("Peak memory", memory_results["mem_peak_mb"], "MB")
            print_result("Memory increase", memory_results["mem_increase_mb"], "MB")
            all_memory_results[model_name] = memory_results
            
            # Test 3: Batch Inference
            print_section(f"{model_name} - Batch Inference")
            batch_results = test_batch_inference(model)
            for batch_size, res in batch_results.items():
                print(f"  Batch size {batch_size:2d}: {res['per_image_ms']:.2f} ms/image, {res['fps']:.1f} FPS")
            all_batch_results[model_name] = batch_results
            
        except Exception as e:
            print(f"  {COLOR_YELLOW}❌ Test failed: {e}{COLOR_RESET}")
    
    # Print summary report
    print_section("PERFORMANCE SUMMARY REPORT")
    
    for model_name in all_speed_results.keys():
        speed = all_speed_results[model_name]
        memory = all_memory_results.get(model_name, {})
        
        print(f"\n{COLOR_GREEN}🔹 {model_name}{COLOR_RESET}")
        print(f"   Inference: {speed['avg_ms']:.2f} ms ({speed['fps']:.1f} FPS)")
        if memory:
            print(f"   Memory: +{memory['mem_increase_mb']:.1f} MB (peak: {memory['mem_peak_mb']:.1f} MB)")
    
    print_section("OPTIMIZATION RECOMMENDATIONS")
    print("   1. Use yolov8n.pt for fastest inference speed")
    print("   2. Enable half precision (FP16) for GPU acceleration")
    print("   3. Use batch inference for video processing")
    print("   4. Limit trajectory queue length to reduce memory")
    print("   5. Add periodic garbage collection")
    
    print(f"\n{COLOR_GREEN}✅ Performance testing completed!{COLOR_RESET}\n")


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()