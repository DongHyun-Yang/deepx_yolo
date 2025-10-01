# TODO:
# ultralytics deepx 라이브러리의 AutoBackend.forward() 함수에서 처리되는 경우 
# bboxes 가 오동작함. 확인필요

"""
YOLOv11 Segmentation DXNN Inference using Custom Ultralytics DEEPX Library

This implementation uses the custom Ultralytics DEEPX library for end-to-end inference.
All preprocessing, inference, and postprocessing are handled internally by Ultralytics.

Implementation details:
- Preprocessing: Ultralytics internal letterbox and normalization
- Inference: DXNN Runtime execution via Ultralytics AutoBackend
- Postprocessing: Ultralytics internal NMS, mask generation, and Results generation

Dependencies: cv2, numpy, ultralytics (customized by DEEPX)
"""

# IMPORTANT: Import this BEFORE any ultralytics imports
import ultralytics_deepx_lib_setup
"""
'ultralytics_deepx_lib_setup' imports the custom Ultralytics DEEPX library from 'lib/ultralytics'
(defined in '.gitmodules').

The custom Ultralytics DEEPX library includes modifications to 'lib/ultralytics/ultralytics/nn/autobackend.py'
that enable the following debugging and DXNN features:

1. DXNN model (.dxnn) inference support
   - Enables DEEPX runtime for DXNN model inference
   - See 'predict_dxnn_ultralytics_deepx.py' run_inference() for usage example

2. Debug: model input tensor visualization and saving
   - Saves preprocessed input tensors to:
     'runs/predict/[MODEL_EXTENSION]/ultralytics_deepx/debug/input/preprocessed_input_[TIMESTAMP].jpg'

3. Debug: model raw output tensor saving and comparison
   - Saves raw output tensors to:
     'runs/predict/[MODEL_EXTENSION]/ultralytics_deepx/debug/raw_output/raw_output_[TIMESTAMP].npy'
   - Use 'util/compare_raw_outputs.py' to compare outputs with other implementations:
     * predict_dxnn_standalone.py: Direct DXNN Runtime without Ultralytics
       → 'runs/predict/dxnn/standalone/debug/raw_output/raw_output_[TIMESTAMP].npy'
     * predict_dxnn_ultralytics_deepx.py: DXNN model inference with custom library
       → 'runs/predict/dxnn/ultralytics_deepx/debug/raw_output/raw_output_[TIMESTAMP].npy'
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

from ultralytics import YOLO

# Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'dxnn'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'boats.jpg'
SOURCE_PATH = PROJECT_ROOT / 'assets'
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_deepx"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
DEBUG_ORIGIN_OUTPUT_DIR = DEBUG_OUTPUT_DIR / 'origin_output'
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

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


def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False):
    """
    Run complete inference using specified backend.
    
    Args:
        model_path: Path to model file
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

        if not debug:
            print("[INFO] Even if debug=False, Ultralytics DEEPX always saves debug data(preprocessed image, raw output).")

        # Load the DXNN model (use task='segment' for instance segmentation)
        model = YOLO(model=model_path, task='segment')

        # Debug: Verify model class names
        print("DXNN Model names:", model.names)

        # Run inference using Ultralytics YOLO class
        # The YOLO class internally handles:
        # 1. Preprocessing: letterbox, normalization, channel conversion
        # 2. Inference: DXNN Runtime execution via AutoBackend
        # 3. Postprocessing: NMS, mask generation, coordinate scaling, Results object creation
        results = model(source=image_path, save=save, project=CURRENT_DIR, name=DEBUG_ORIGIN_OUTPUT_DIR)
        result = results[0]  # Get first (and only) result

        # 4. Visualization and analysis
        filename = Path(image_path).stem
        output_filename = Path(image_path).stem + f'_detected_{timestamp}.jpg'
        output_path = str(Path(output_dir) / output_filename)

        # Draw segmentation using Results object
        draw_segmentation(image_path, result, output_path, save=save, show=show)

        # Print analysis result
        analyze_results(result, filename)

        return output_path

    except Exception as e:
        print(f"[{Path(image_path).name}] Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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