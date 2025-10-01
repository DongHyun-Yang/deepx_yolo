"""
YOLOv11 Pose Estimation ONNX Inference with Ultralytics Postprocessing

This implementation removes the dependency on Ultralytics YOLO class and provides
custom implementations for preprocessing and ONNX Runtime inference.
Only the postprocessing components (Results, ops, NMS) use Ultralytics utilities.

Implementation details:
- Preprocessing: Custom letterbox and image preprocessing
- Inference: Direct ONNX Runtime session execution
- Postprocessing: Ultralytics utilities (Results, ops.scale_boxes, ops.scale_coords, non_max_suppression)

Dependencies: cv2, numpy, torch, onnxruntime, ultralytics (postprocessing only)
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# Add ultralytics path
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

# Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'onnx'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'boats.jpg'
SOURCE_PATH = PROJECT_ROOT / 'assets'
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_postprocess"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

# Detection parameters (Ultralytics defaults)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# COCO Pose keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO Pose skeleton connections (matching ultralytics)
POSE_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9), (8, 10),
    (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)
]

# COCO Pose colors (matching ultralytics)
POSE_PALETTE = [
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
]

# Keypoint and limb colors (matching ultralytics)
KPT_COLOR = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
LIMB_COLOR = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=64):
    """
    Resize and pad image while meeting stride-multiple constraints.
    Custom implementation based on Ultralytics letterbox function.
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)

def preprocess_image(image_path, imgsz=640):
    """
    Read image and preprocess it for model input.
    Custom preprocessing implementation without Ultralytics dependencies.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    # Apply letterbox preprocessing
    processed_image, ratio, (dw, dh) = letterbox(image, new_shape=(imgsz, imgsz))
    
    # Convert to RGB and normalize
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_image = processed_image.transpose(2, 0, 1)  # HWC to CHW
    processed_image = np.ascontiguousarray(processed_image)
    processed_image = processed_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    print(f"Debug info:")
    print(f"  Original size: {original_width}x{original_height}")
    print(f"  Ratio: {ratio}")
    print(f"  Padding (dw, dh): {(dw, dh)}")
    print(f"  Input tensor shape: {processed_image.shape}")
    print(f"  Input tensor range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    return processed_image, image, ratio, (dw, dh), (original_width, original_height)

def postprocess_pose(preds, orig_img):
    """
    Postprocess pose model output using Ultralytics utilities.
    Uses Ultralytics non_max_suppression, ops.scale_boxes, ops.scale_coords, and Results class.
    
    Args:
        preds: Raw Model output tensor [1, 56, 8400]
            Format: 56 = 4(bbox) + 1(conf) + 51(17 keypoints * 3 [x, y, conf])
        orig_img: Original image (BGR) - used for shape extraction and Results object creation
    
    Returns:
        Results: Results object containing box detections and keypoints
    """
    # Convert to torch tensor if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    
    print(f"\n[Postprocess] Input shape: {preds.shape}")
    print(f"[Postprocess] Input range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    # Apply Non-Maximum Suppression using Ultralytics
    # Ultralytics NMS expects [batch, num_values, num_detections]
    # Current format: [1, 56, 8400] where 56 = 4(bbox) + 1(conf) + 51(17 kpts * 3)
    # Uses Ultralytics NMS function
    results = non_max_suppression(
        preds,
        conf_thres=CONFIDENCE_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=1,  # 1 class (person) for pose estimation
    )
    
    print(f"[Postprocess] NMS output: {len(results)} image(s)")
    if len(results) > 0 and len(results[0]) > 0:
        print(f"[Postprocess] First result shape: {results[0].shape}")
        print(f"[Postprocess] Total detections: {len(results[0])}")
    
    # Process first image result
    if len(results) == 0 or len(results[0]) == 0:
        # No detections
        return Results(
            orig_img=orig_img,
            path=None,
            names={0: 'person'},
            boxes=None,
            keypoints=None
        )
    
    pred = results[0]  # [num_detections, 6 + 51] = [x1, y1, x2, y2, conf, cls, kpt_x1, kpt_y1, kpt_conf1, ...]
    
    print(f"[Postprocess] Pred shape before scale: {pred.shape}")
    print(f"[Postprocess] Pred sample (first detection): {pred[0, :6] if len(pred) > 0 else 'none'}")
    
    # Split predictions into boxes and keypoints
    pred_boxes = pred[:, :6]  # [x1, y1, x2, y2, conf, cls]
    pred_kpts = pred[:, 6:].reshape(-1, 17, 3)  # [num_detections, 17, 3] where 3 = [x, y, conf]
    
    # Scale boxes from letterbox size to original image size
    preproc_shape = (INPUT_SIZE, INPUT_SIZE)
    pred_boxes[:, :4] = ops.scale_boxes(preproc_shape, pred_boxes[:, :4], orig_img.shape)
    
    # Scale keypoints from letterbox size to original image size
    # ops.scale_coords expects keypoints in format [num_kpts, num_points, 2]
    # We have [num_detections, 17, 3] so we need to scale x,y coordinates
    pred_kpts_xy = pred_kpts[..., :2]  # Extract x, y coordinates
    pred_kpts_conf = pred_kpts[..., 2:3]  # Extract confidence
    
    # Reshape for scale_coords: [num_detections * 17, 2]
    num_detections, num_kpts, _ = pred_kpts.shape
    pred_kpts_xy_flat = pred_kpts_xy.reshape(-1, 2)
    
    # Scale coordinates
    pred_kpts_xy_flat = ops.scale_coords(preproc_shape, pred_kpts_xy_flat, orig_img.shape)
    
    # Reshape back: [num_detections, 17, 2]
    pred_kpts_xy_scaled = pred_kpts_xy_flat.reshape(num_detections, num_kpts, 2)
    
    # Combine scaled x,y with original confidence
    pred_kpts_scaled = torch.cat([pred_kpts_xy_scaled, pred_kpts_conf], dim=-1)
    
    print(f"[Postprocess] Box data shape after scale: {pred_boxes.shape}")
    print(f"[Postprocess] Keypoints shape after scale: {pred_kpts_scaled.shape}")
    print(f"[Postprocess] Confidence range: {pred_boxes[:, 4].min():.3f} ~ {pred_boxes[:, 4].max():.3f}")
    
    # Create Results object
    # Keypoints format for Results: [num_detections, 17, 3] where 3 = [x, y, conf]
    result = Results(
        orig_img=orig_img,
        path=None,
        names={0: 'person'},
        boxes=pred_boxes,  # [n, 6] tensor: [x1, y1, x2, y2, conf, cls]
        keypoints=pred_kpts_scaled  # [n, 17, 3] tensor: [x, y, conf]
    )
    
    return result

def draw_pose_detections(source_path, result, output_path, save=True, show=True):
    """
    Draw pose estimation results on image using Results object.
    Custom visualization implementation using OpenCV.
    """
    image = cv2.imread(source_path)
    
    if result.keypoints is not None:
        # Get pose keypoints data
        keypoints = result.keypoints.xy.cpu().numpy()  # Shape: [num_persons, 17, 2]
        keypoints_conf = result.keypoints.conf.cpu().numpy()  # Shape: [num_persons, 17]
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf.cpu().numpy()  # confidence score of each box
        
        # Line width calculation
        line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
        
        # Draw pose for each person
        for i in range(len(keypoints)):
            box = boxes[i].astype(int)
            conf = confs[i]
            name = names[i]
            person_keypoints = keypoints[i]  # Shape: [17, 2]
            person_conf = keypoints_conf[i]  # Shape: [17]
            
            # Draw bounding box
            box_color = (255, 144, 30)  # Ultralytics default box color (BGR)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, line_width)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, max(line_width - 1, 1))[0]
            cv2.rectangle(image, (box[0], box[1] - label_size[1] - 10), 
                         (box[0] + label_size[0], box[1]), box_color, -1)
            cv2.putText(image, label, (box[0], box[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), max(line_width - 1, 1))
            
            # Draw keypoints
            radius = max(line_width - 1, 1)
            conf_thres = 0.25
            
            for j in range(17):  # 17 keypoints
                if person_conf[j] > conf_thres:
                    kx, ky = person_keypoints[j]
                    if kx > 0 and ky > 0:  # Valid keypoint
                        # Get keypoint color
                        color_idx = KPT_COLOR[j]
                        color = POSE_PALETTE[color_idx]
                        # Convert RGB to BGR for OpenCV
                        color_bgr = (color[2], color[1], color[0])
                        cv2.circle(image, (int(kx), int(ky)), radius, color_bgr, -1, lineType=cv2.LINE_AA)
            
            # Draw skeleton connections
            for j, (start_idx, end_idx) in enumerate(POSE_SKELETON):
                # Adjust indices (ultralytics uses 1-based, we use 0-based)
                start_idx_adj = start_idx - 1
                end_idx_adj = end_idx - 1
                
                if 0 <= start_idx_adj < 17 and 0 <= end_idx_adj < 17:
                    start_conf = person_conf[start_idx_adj]
                    end_conf = person_conf[end_idx_adj]
                    start_kpt = person_keypoints[start_idx_adj]
                    end_kpt = person_keypoints[end_idx_adj]
                    
                    # Check confidence and validity
                    if (start_conf > conf_thres and end_conf > conf_thres and
                        start_kpt[0] > 0 and start_kpt[1] > 0 and
                        end_kpt[0] > 0 and end_kpt[1] > 0):
                        
                        # Get limb color
                        limb_color_idx = LIMB_COLOR[j]
                        limb_color_rgb = POSE_PALETTE[limb_color_idx]
                        # Convert RGB to BGR for OpenCV
                        limb_color_bgr = (limb_color_rgb[2], limb_color_rgb[1], limb_color_rgb[0])
                        
                        start_point = (int(start_kpt[0]), int(start_kpt[1]))
                        end_point = (int(end_kpt[0]), int(end_kpt[1]))
                        
                        cv2.line(image, start_point, end_point, limb_color_bgr, 
                               max(int(line_width / 2), 1), lineType=cv2.LINE_AA)
    
    elif result.boxes is not None:
        # Fallback to bounding boxes only if no keypoints
        boxes = result.boxes.xyxy.cpu().numpy()
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        confs = result.boxes.conf.cpu().numpy()
        
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            conf = confs[i]
            name = names[i]
            # Draw rectangle
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
            # Draw label
            label = f"{name}: {conf:.2f}"
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, image)
        print(f"Pose estimation result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("Pose Estimation Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_results(result, filename):
    """
    Analyze and print pose estimation results from Results object.
    Provides detailed statistics on detected objects.

    Args:
        result (Results): Results object containing boxes and keypoints
        filename (str): Name of the processed file
    """
    
    if result.keypoints is not None and len(result.keypoints) > 0:
        print("="*50)
        print(f"Total pose instances: {len(result.keypoints)}")
        print(f"Keypoints tensor shape: {result.keypoints.data.shape}")
        print(f"Boxes tensor shape: {result.boxes.data.shape}")
        
        # Get confidence values
        confidences = result.boxes.conf.cpu().numpy()
        print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
        print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
        
        # Check class distribution (should be all persons for pose)
        classes = result.boxes.cls.cpu().numpy()
        unique_classes, counts = np.unique(classes, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
        
        # More detailed conf analysis
        conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(len(conf_bins)-1):
            count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
            print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
        print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
        
        # Keypoint statistics
        keypoints_conf = result.keypoints.conf.cpu().numpy()  # Shape: [num_persons, 17]
        
        boxes = result.boxes.xyxy.cpu().numpy()
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        
        print(f"[{filename}] Total {len(result.keypoints)} pose instances detected.")
        for i, person_kpt_conf in enumerate(keypoints_conf):
            visible_keypoints = np.sum(person_kpt_conf > 0.5)
            box = boxes[i]
            x1, y1, x2, y2 = box
            print(f"  {i+1}. {names[i]}: {confidences[i]:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f}) - Keypoints: {visible_keypoints}/17 visible")
            
            # Show keypoint details for first person only to avoid too much output
            if i == 0:
                for j, kpt_name in enumerate(KEYPOINT_NAMES):
                    if person_kpt_conf[j] > 0.25:
                        print(f"    {kpt_name}: {person_kpt_conf[j]:.2f}")
        
        print(f"Average visible keypoints per person: {np.mean([np.sum(conf > 0.5) for conf in keypoints_conf]):.1f}/17")
        
    elif result.boxes is not None and len(result.boxes) > 0:
        print("="*50)
        print(f"Total object detections: {len(result.boxes)}")
        print(f"Boxes tensor shape: {result.boxes.data.shape}")
        
        # Get confidence values
        confidences = result.boxes.conf.cpu().numpy()
        print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
        print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
        
        # Check class distribution
        classes = result.boxes.cls.cpu().numpy()
        unique_classes, counts = np.unique(classes, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
        
        print(f"[{filename}] No pose keypoints detected, showing {len(result.boxes)} bounding boxes only.")
        
    else:
        print(f"[{filename}] No pose keypoints or object detections found.")

def run_inference_using_onnx(model_path, input_tensor):
    """
    Run inference with ONNX model using ONNX Runtime.
    Direct implementation without Ultralytics dependencies.
    """
    import onnxruntime as ort

    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"\nRunning ONNX inference...")
    outputs = session.run([output_name], {input_name: input_tensor})

    return outputs

def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False):
    """
    Run complete inference using specified backend.
    
    Args:
        model_path: Path to model file
        image_path: Path to input image
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output image with pose estimation
        show: Display output image
    
    Returns:
        str: Path to output image if successful, None otherwise
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # Process image
        print(f"\nProcessing: {Path(image_path).name}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Preprocess
        input_tensor, orig_img, ratio, pad, orig_shape = preprocess_image(image_path, INPUT_SIZE)

        # Debug: Visualize preprocessed input tensor
        if debug:
            preprocess_image_dir = Path(DEBUG_OUTPUT_DIR) / 'input'
            preprocess_image_dir.mkdir(parents=True, exist_ok=True)
            preprocess_image_path = Path(preprocess_image_dir) / f'preprocessed_input_{Path(image_path).stem}_{timestamp}.jpg'
            debug_visualize_tensor(input_tensor, title="Preprocessed Input Tensor", save_path=preprocess_image_path, show=False)
        
        # 2. Inference
        outputs = run_inference_using_onnx(model_path, input_tensor)
        preds = outputs[0]  # Get first (and only) output

        for idx, output in enumerate(outputs):
            print(f"Raw output[{idx}] shape: {output.shape}")
            print(f"Raw output[{idx}] range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Debug: Save raw output for debugging
            if debug:
                raw_output_dir = Path(DEBUG_OUTPUT_DIR) / 'raw_output'
                raw_output_dir.mkdir(parents=True, exist_ok=True)
                raw_output_path = Path(raw_output_dir) / f'raw_output{idx}_{Path(image_path).stem}_{timestamp}.npy'
                np.save(str(raw_output_path), output)
                print(f"Inference Raw output saved to: {raw_output_path}")

        # 3. Post-processing using Ultralytics utilities
        result = postprocess_pose(preds, orig_img)

        # 4. Visualization and analysis
        filename = Path(image_path).stem
        output_filename = Path(image_path).stem + f'_detected_{timestamp}.jpg'
        output_path = str(Path(output_dir) / output_filename)

        # Draw pose estimation
        draw_pose_detections(image_path, result, output_path, save=save, show=show)

        # Print analysis result
        analyze_results(result, filename)

        return output_path

    except Exception as e:
        print(f"[{Path(image_path).name}] Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def debug_visualize_tensor(tensor, title="Input Tensor", save_path=None, show=False):
    """
    Debug utility to visualize tensors as images.
    Converts NCHW/CHW format tensors to displayable BGR images.
    """
    try:
        # Convert tensor to numpy if needed
        if isinstance(tensor, torch.Tensor):
            im_vis = tensor.cpu().numpy()
        else:
            im_vis = tensor
        
        # Handle different tensor formats
        if len(im_vis.shape) == 4:  # NCHW format
            im_vis = im_vis.squeeze(0).transpose(1, 2, 0)  # (H, W, C)
        elif len(im_vis.shape) == 3:  # CHW format
            im_vis = im_vis.transpose(1, 2, 0)  # (H, W, C)
        
        # Convert to displayable format
        if im_vis.dtype == np.float32 or im_vis.dtype == np.float64:
            # Convert from [0, 1] range to [0, 255]
            im_vis = (im_vis * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if im_vis.shape[-1] == 3:
            im_vis = im_vis[:, :, ::-1]
        
        # Show image
        if show:
            cv2.imshow(title, im_vis)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, im_vis)
            print(f"Tensor visualization saved to: {save_path}")
        
        print(f"[Debug] {title} - Shape: {tensor.shape if isinstance(tensor, torch.Tensor) else im_vis.shape}")
        return im_vis
        
    except Exception as e:
        print(f"[Debug] Failed to visualize tensor: {e}")
        print(f"[Debug] Tensor info - Type: {type(tensor)}, Shape: {tensor.shape if hasattr(tensor, 'shape') else 'unknown'}")

def main():
    """
    Main function to process single image or directory of images.
    Supports batch processing and provides detailed summary.
    """
    saved_files = []
    
    source_path = Path(SOURCE_PATH)

    # Check if source is file or directory
    if source_path.is_file():
        print("Processing single image file.")
        print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
        print("-" * 50)

        result_path = run_inference(MODEL_PATH, str(source_path), OUTPUT_DIR, debug=True)
        if result_path:
            saved_files.append(result_path)

    elif source_path.is_dir():
        print("Processing directory of images.")
        print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
        print("-" * 50)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(ext))
        
        if not image_files:
            print(f"No image files found in {source_path}")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
            print("-" * 50)
            result_path = run_inference(MODEL_PATH, str(image_path), OUTPUT_DIR, debug=True)
            if result_path:
                saved_files.append(result_path)
    else:
        print(f"Error: {source_path} is neither a file nor a directory")
        return

    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(saved_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nSaved files:")
    for idx, file_path in enumerate(saved_files, 1):
        print(f"  {idx}. {Path(file_path)}")
    print(f"{'='*70}")
    print("Processing completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()