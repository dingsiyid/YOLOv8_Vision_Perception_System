"""
测试修复后的 view_result 函数逻辑
模拟包含 None object_id 的检测结果数据
"""
import numpy as np
import cv2
from collections import deque

# 模拟 COLORS
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255)]

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


def view_result_fixed(result_list_json, centers=None):
    """修复后的可视化函数"""
    # 创建测试图像
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    for res in result_list_json:
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        
        # 绘制边界框
        cv2.rectangle(image, (res['bbox']['x_min'], res['bbox']['y_min']), 
                      (res['bbox']['x_max'], res['bbox']['y_max']), class_color, 2)
        
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
    
    return image


def test_case_1():
    """Test case 1: Mixed None and valid object_id"""
    print("=" * 60)
    print("Test Case 1: Mixed None and valid object_id")
    print("=" * 60)
    
    # Simulate detection results with None object_id
    result_list_json = [
        {
            'class_id': 0,
            'class': 'person',
            'confidence': 0.95,
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 200, 'y_max': 250},
            'object_id': 1  # Valid ID
        },
        {
            'class_id': 0,
            'class': 'person',
            'confidence': 0.87,
            'bbox': {'x_min': 300, 'y_min': 100, 'x_max': 400, 'y_max': 300},
            'object_id': None  # None ID (tracking failed)
        },
        {
            'class_id': 2,
            'class': 'car',
            'confidence': 0.92,
            'bbox': {'x_min': 450, 'y_min': 150, 'x_max': 600, 'y_max': 350},
            'object_id': 2  # Valid ID
        },
        {
            'class_id': 1,
            'class': 'dog',
            'confidence': 0.78,
            'bbox': {'x_min': 50, 'y_min': 200, 'x_max': 150, 'y_max': 350},
            # No object_id key
        }
    ]
    
    # Initialize tracking centers
    centers = [deque(maxlen=30) for _ in range(10000)]
    
    try:
        result_image = view_result_fixed(result_list_json, centers=centers)
        print("[PASS] Test passed! Image generated successfully")
        print(f"   - Detected {len(result_list_json)} objects")
        print(f"   - Valid tracking IDs: {[r.get('object_id') for r in result_list_json if r.get('object_id') is not None]}")
        print(f"   - None/Missing IDs: {[r.get('object_id') for r in result_list_json if r.get('object_id') is None or 'object_id' not in r]}")
        
        # Save test image
        cv2.imwrite('test_output_1.png', result_image)
        print(f"   - Test image saved: test_output_1.png")
        return True
    except Exception as e:
        print(f"[FAIL] Test failed: {str(e)}")
        return False


def test_case_2():
    """Test case 2: All object_id are None"""
    print("\n" + "=" * 60)
    print("Test Case 2: All object_id are None")
    print("=" * 60)
    
    result_list_json = [
        {
            'class_id': 0,
            'class': 'person',
            'confidence': 0.88,
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 200, 'y_max': 250},
            'object_id': None
        },
        {
            'class_id': 2,
            'class': 'car',
            'confidence': 0.91,
            'bbox': {'x_min': 300, 'y_min': 100, 'x_max': 450, 'y_max': 300},
            'object_id': None
        }
    ]
    
    centers = [deque(maxlen=30) for _ in range(10000)]
    
    try:
        result_image = view_result_fixed(result_list_json, centers=centers)
        print("[PASS] Test passed! Image generated successfully")
        print(f"   - All object_ids are None, tracking disabled")
        cv2.imwrite('test_output_2.png', result_image)
        print(f"   - Test image saved: test_output_2.png")
        return True
    except Exception as e:
        print(f"[FAIL] Test failed: {str(e)}")
        return False


def test_case_3():
    """Test case 3: No object_id key (image mode)"""
    print("\n" + "=" * 60)
    print("Test Case 3: No object_id key (image mode)")
    print("=" * 60)
    
    result_list_json = [
        {
            'class_id': 0,
            'class': 'person',
            'confidence': 0.95,
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 200, 'y_max': 250}
        },
        {
            'class_id': 1,
            'class': 'dog',
            'confidence': 0.82,
            'bbox': {'x_min': 300, 'y_min': 100, 'x_max': 400, 'y_max': 300}
        }
    ]
    
    # Image mode doesn't use centers
    try:
        result_image = view_result_fixed(result_list_json, centers=None)
        print("[PASS] Test passed! Image generated successfully")
        print(f"   - Image mode, no object_id required")
        cv2.imwrite('test_output_3.png', result_image)
        print(f"   - Test image saved: test_output_3.png")
        return True
    except Exception as e:
        print(f"[FAIL] Test failed: {str(e)}")
        return False


def test_case_4():
    """Test case 4: Edge case - object_id out of centers range"""
    print("\n" + "=" * 60)
    print("Test Case 4: Edge case - object_id out of centers range")
    print("=" * 60)
    
    result_list_json = [
        {
            'class_id': 0,
            'class': 'person',
            'confidence': 0.95,
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 200, 'y_max': 250},
            'object_id': 99999  # Out of range
        }
    ]
    
    # Create limited centers
    centers = [deque(maxlen=30) for _ in range(100)]
    
    try:
        result_image = view_result_fixed(result_list_json, centers=centers)
        print("[PASS] Test passed! Image generated successfully")
        print(f"   - object_id=99999 out of centers range (100), auto-skipped tracking")
        cv2.imwrite('test_output_4.png', result_image)
        print(f"   - Test image saved: test_output_4.png")
        return True
    except Exception as e:
        print(f"[FAIL] Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n[Test] Starting tests for fixed view_result function\n")
    
    results = []
    results.append(test_case_1())
    results.append(test_case_2())
    results.append(test_case_3())
    results.append(test_case_4())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[Success] All tests passed! Fix logic is correct.")
    else:
        print(f"\n[Failed] {total - passed} tests failed.")
