"""
YOLOv11 Segmentation ONNX Inference with Ultralytics Postprocessing

This implementation removes the dependency on Ultralytics YOLO class and provides
custom implementations for preprocessing and ONNX Runtime inference.
Only the postprocessing components (Results, ops, process_mask, NMS) use Ultralytics utilities.

Implementation details:
- Preprocessing: Custom letterbox and image preprocessing
- Inference: Direct ONNX Runtime session execution
- Postprocessing: Ultralytics utilities (Results, ops.scale_boxes, ops.process_mask, non_max_suppression)

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
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 640

# COCO class names (based on dataset that YOLOv11 was trained on)
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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

def postprocess_segmentation(outputs, orig_img):
    """
    Postprocess segmentation model output using Ultralytics utilities.
    Uses Ultralytics non_max_suppression, ops.scale_boxes, ops.process_mask, and Results class.
    
    Args:
        outputs: List of model outputs [predictions, proto_masks]
            - predictions: [1, 116, 8400] (4 box + 80 classes + 32 mask coeffs)
            - proto_masks: [1, 32, 160, 160] (prototype masks)
        orig_img: Original image (BGR) - used for shape extraction and Results object creation
    
    Returns:
        Results: Ultralytics Results object containing box detections and segmentation masks
    """
    if len(outputs) < 2:
        return Results(
            orig_img=orig_img,
            path=None,
            names={i: name for i, name in enumerate(CLASSES)},
            boxes=None,
            masks=None
        )
    
    preds = outputs[0]  # [1, 116, 8400]
    protos = outputs[1]  # [1, 32, 160, 160]
    
    # Convert to torch tensor if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    if isinstance(protos, np.ndarray):
        protos = torch.from_numpy(protos).float()
    
    print(f"\n[Postprocess] Predictions shape: {preds.shape}")
    print(f"[Postprocess] Predictions range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"[Postprocess] Proto masks shape: {protos.shape}")
    
    # Apply Non-Maximum Suppression using Ultralytics
    # Ultralytics NMS expects [batch, num_classes + 4 + num_masks, num_detections]
    # Current format: [1, 116, 8400] where 116 = 4(bbox) + 80(classes) + 32(mask_coeffs)
    # This is already in the correct format - NO transpose needed!
    results = non_max_suppression(
        preds,
        conf_thres=CONFIDENCE_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=80,  # 80 classes for COCO dataset
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
            names={i: name for i, name in enumerate(CLASSES)},
            boxes=None,
            masks=None
        )
    
    pred = results[0]  # [num_detections, 6 + 32] = [x1, y1, x2, y2, conf, cls, mask_coeffs...]
    
    print(f"[Postprocess] Pred shape before scale: {pred.shape}")
    print(f"[Postprocess] Pred sample (first detection): {pred[0, :6] if len(pred) > 0 else 'none'}")
    
    # Split predictions into boxes and mask coefficients
    pred_boxes = pred[:, :6]  # [x1, y1, x2, y2, conf, cls]
    pred_masks_coeffs = pred[:, 6:]  # [32 mask coefficients]
    
    # IMPORTANT: Process masks BEFORE scaling boxes
    # Following SegmentationPredictor.construct_result() pattern
    preproc_shape = (INPUT_SIZE, INPUT_SIZE)
    
    # Process masks with letterbox coordinates
    # ops.process_mask(protos, masks_in, bboxes, shape, upsample)
    # - protos: [32, 160, 160] mask prototypes
    # - masks_in: [N, 32] mask coefficients after NMS
    # - bboxes: [N, 4] boxes in letterbox coordinates (NOT scaled yet)
    # - shape: (height, width) of input image (letterbox size)
    # - upsample: whether to upsample masks to letterbox image size
    masks = ops.process_mask(protos[0], pred_masks_coeffs, pred_boxes[:, :4], preproc_shape, upsample=True)
    
    print(f"[Postprocess] Masks shape after process_mask: {masks.shape}")
    print(f"[Postprocess] Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
    
    # THEN scale boxes to original image size
    pred_boxes[:, :4] = ops.scale_boxes(preproc_shape, pred_boxes[:, :4], orig_img.shape)
    
    # Scale masks from letterbox size to original image size
    # Using ops.scale_masks which handles letterbox padding correctly
    masks = ops.scale_masks(masks[None], orig_img.shape[:2], padding=True)[0]
    
    print(f"[Postprocess] Masks shape after scale: {masks.shape}")
    print(f"[Postprocess] Box data shape after scale: {pred_boxes.shape}")
    print(f"[Postprocess] Confidence range: {pred_boxes[:, 4].min():.3f} ~ {pred_boxes[:, 4].max():.3f}")
    
    # Create Results object
    result = Results(
        orig_img=orig_img,
        path=None,
        names={i: name for i, name in enumerate(CLASSES)},
        boxes=pred_boxes,  # [n, 6] tensor: [x1, y1, x2, y2, conf, cls]
        masks=masks  # [n, h, w] tensor: binary masks
    )
    
    return result

def draw_segmentation(source_path, result, output_path, save=True, show=True):
    """
    Draw segmentation masks and bounding boxes on image using Results object.
    
    Args:
        source_path: Path to source image
        result: Results object containing boxes and masks
        output_path: Path to save output image
        save: Whether to save the output
        show: Whether to display the output
    """
    image = cv2.imread(source_path)
    
    # Check if there are any detections
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections to visualize.")
        if save:
            cv2.imwrite(output_path, image)
        return
    
    # Create overlay for masks
    overlay = image.copy()
    
    # Color palette (different colors for each class)
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    # Get detection data
    xyxy = result.boxes.xyxy  # x1, y1, x2, y2 format
    confs = result.boxes.conf  # confidence scores
    cls_ids = result.boxes.cls.int()  # class IDs
    
    # Get masks data if available (use contours for drawing)
    masks_xy = None
    if hasattr(result, 'masks') and result.masks is not None:
        # Use result.masks.xy which contains mask contours in original image coordinates
        masks_xy = result.masks.xy
    
    for i in range(len(result.boxes)):
        box = xyxy[i].cpu().numpy().astype(int)
        conf = confs[i].item()
        class_id = cls_ids[i].item()
        
        x1, y1, x2, y2 = box
        class_name = result.names[class_id]
        label = f"{class_name}: {conf:.2f}"
        
        # Get class-specific color
        color = colors[class_id].tolist()
        
        # Draw segmentation mask if available
        if masks_xy is not None and i < len(masks_xy):
            # Get mask contour (already in original image coordinates)
            mask_contour = masks_xy[i]
            
            if len(mask_contour) > 0:
                # Create binary mask from contour
                mask_binary = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_binary, [mask_contour.astype(np.int32)], 1)
                
                # Create colored mask
                colored_mask = np.zeros_like(image)
                colored_mask[mask_binary > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
                
                # Draw mask contours
                cv2.drawContours(overlay, [mask_contour.astype(np.int32)], -1, color, 2)
        
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Label background rectangle
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Label text (in white)
        cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, overlay)
        print(f"Segmentation result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("Segmentation Results", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_results(result, filename):
    """
    Analyze and print segmentation results from Results object.
    Provides detailed statistics on detected objects and masks.

    Args:
        result (Results): Results object containing boxes and masks
        filename (str): Name of the processed file
    """
    if result.boxes is None or len(result.boxes) == 0:
        print(f"[{filename}] No objects detected.")
        return

    print("="*50)
    print(f"Total object detections: {len(result.boxes)}")
    print(f"Boxes tensor shape: {result.boxes.data.shape}")
    
    # Get confidence values
    confidences = result.boxes.conf.cpu().numpy()
    print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
    print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
    
    # Calculate mask statistics if available
    if hasattr(result, 'masks') and result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        total_mask_pixels = sum(np.sum(mask > 0.5) for mask in masks_data)
        avg_mask_pixels = total_mask_pixels / len(masks_data) if len(masks_data) > 0 else 0
        print(f"Total masks: {len(masks_data)}")
        print(f"Average mask area: {avg_mask_pixels:.0f} pixels")
    
    # Check class distribution
    classes = result.boxes.cls.cpu().numpy()
    unique_classes, counts = np.unique(classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
    
    # Detected classes with names
    print(f"Detected classes: {len(unique_classes)}")
    for cls_id, count in zip(unique_classes.astype(int), counts):
        class_name = result.names[cls_id] if cls_id < len(result.names) else f"Unknown_{cls_id}"
        print(f"  - {class_name}: {count} instance(s)")
    
    # More detailed conf analysis
    conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(conf_bins)-1):
        count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
        print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
    print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
    
    # Detailed detection info
    xyxy = result.boxes.xyxy
    cls_ids = result.boxes.cls.int()
    confs = result.boxes.conf
    
    print(f"[{filename}] Total {len(result.boxes)} objects segmented.")
    for i in range(len(result.boxes)):
        class_name = result.names[cls_ids[i].item()]
        score = confs[i].item()
        box = xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = box
        
        mask_info = ""
        if hasattr(result, 'masks') and result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()
            if i < len(masks_data):
                mask_area = np.sum(masks_data[i] > 0.5)
                mask_info = f" - Mask area: {mask_area:.0f} pixels"
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f}){mask_info}")

def run_inference_using_onnx(model_path, input_tensor):
    """
    Run inference with ONNX model using ONNX Runtime.
    Direct implementation without Ultralytics dependencies.
    """
    import onnxruntime as ort

    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    return outputs

def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False):
    """
    Run inference using specified backend, and Ultralytics postprocessing.
    
    Args:
        model_path: Path to ONNX model file
        image_path: Path to input image
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output image with segmentation
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

        # 3. Post-processing
        result = postprocess_segmentation(outputs, orig_img)

        # 4. Visualization and analysis
        filename = Path(image_path).stem
        output_filename = Path(image_path).stem + f'_detected_{timestamp}.jpg'
        output_path = str(Path(output_dir) / output_filename)

        # Draw segmentation
        draw_segmentation(image_path, result, output_path, save=save, show=show)

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