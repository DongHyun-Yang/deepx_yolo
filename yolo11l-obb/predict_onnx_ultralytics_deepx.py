"""
YOLOv11 OBB ONNX Inference using Custom Ultralytics DEEPX Library

This implementation uses the custom Ultralytics DEEPX library for end-to-end inference.
All preprocessing, inference, and postprocessing are handled internally by Ultralytics.

Implementation details:
- Preprocessing: Ultralytics internal letterbox and normalization
- Inference: ONNX Runtime execution via Ultralytics AutoBackend
- Postprocessing: Ultralytics internal NMS and Results generation

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
     * predict_onnx_standalone.py: Direct ONNX Runtime without Ultralytics
       → 'runs/predict/onnx/standalone/debug/raw_output/raw_output_[TIMESTAMP].npy'
     * predict_onnx_ultralytics_deepx.py: DXNN model inference with custom library
       → 'runs/predict/onnx/ultralytics_deepx/debug/raw_output/raw_output_[TIMESTAMP].npy'
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'onnx'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'boats.jpg'
SOURCE_PATH = PROJECT_ROOT / 'assets'
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_deepx"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
DEBUG_ORIGIN_OUTPUT_DIR = DEBUG_OUTPUT_DIR / 'origin_output'
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

# DOTAv1.0 class names (dataset that YOLOv11-obb was trained on)
CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 
           'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 
           'roundabout', 'soccer-ball-field', 'swimming-pool']

def draw_detections(source_path, result, output_path, save=True, show=True):
    """
    Draw oriented bounding boxes on image using Ultralytics Results object.
    Custom visualization implementation using OpenCV.
    """
    image = cv2.imread(source_path)
    
    if result.obb is not None:
        xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
        xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
        confs = result.obb.conf  # confidence score of each box

        # Color palette (different colors for each class)
        colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        for i in range(len(xywhr)):
            poly = xyxyxyxy[i].cpu().numpy().astype(int)
            conf = confs[i].item()
            name = names[i]
            class_id = result.obb.cls[i].item()
            
            # Get class-specific color
            color = colors[int(class_id)].tolist()
            
            # Draw oriented bounding box (polygon)
            cv2.drawContours(image, [poly.reshape(-1, 2)], 0, color, 3)
            
            # Get center point for drawing
            center_x, center_y = int(xywhr[i][0].item()), int(xywhr[i][1].item())
            cv2.circle(image, (center_x, center_y), 3, color, -1)
            
            # Calculate label position (use min x,y of polygon)
            min_x, min_y = np.min(poly.reshape(-1, 2), axis=0)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (min_x, min_y - label_size[1] - 10), 
                         (min_x + label_size[0], min_y), color, -1)
            cv2.putText(image, label, (min_x, min_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, image)
        print(f"OBB detection result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("OBB Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_results(result, filename):
    """
    Analyze and print detection results from Ultralytics Results object.
    Provides detailed statistics on detected objects.

    Args:
        result (Results): Results object containing boxes and masks
        filename (str): Name of the processed file
    """
    if result.obb is None or len(result.obb) == 0:
        print(f"[{filename}] No oriented objects detected.")
        return

    print("="*50)
    print(f"Total OBB detections: {len(result.obb)}")
    print(f"OBB tensor shape: {result.obb.data.shape}")
    
    # Get confidence values
    confidences = result.obb.conf.cpu().numpy()
    print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
    print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
    
    # Check class distribution
    classes = result.obb.cls.cpu().numpy()
    unique_classes, counts = np.unique(classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
    
    # More detailed conf analysis
    conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(conf_bins)-1):
        count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
        print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
    print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
    
    # Detailed detection info
    xywhr = result.obb.xywhr
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]
    confs = result.obb.conf
    
    print(f"[{filename}] Total {len(result.obb)} oriented objects detected.")
    for i in range(len(result.obb)):
        class_name = names[i]
        score = confs[i].item()
        xywhr_data = xywhr[i].cpu().numpy()
        cx, cy, w, h, angle = xywhr_data
        
        # Convert to corner coordinates for display
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f}) - OBB: ({cx:.1f}, {cy:.1f}, {w:.1f}x{h:.1f}, {np.degrees(angle):.1f}°)")

def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False):
    """
    Run complete inference using specified backend.
    
    Args:
        model_path: Path to model file
        image_path: Path to input image
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output image with detections
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

        # Load the ONNX model (use task='obb' for object detection)
        model = YOLO(model=model_path, task='obb')

        # Debug: Verify model class names
        print("ONNX Model names:", model.names)

        # Run inference using Ultralytics YOLO class
        # The YOLO class internally handles:
        # 1. Preprocessing: letterbox, normalization, channel conversion
        # 2. Inference: ONNX Runtime execution via AutoBackend
        # 3. Postprocessing: NMS, coordinate scaling, Results object creation
        results = model(source=image_path, save=save, project=CURRENT_DIR, name=DEBUG_ORIGIN_OUTPUT_DIR)
        result = results[0]  # Get first (and only) result

        # 4. Visualization and analysis
        filename = Path(image_path).stem
        output_filename = Path(image_path).stem + f'_detected_{timestamp}.jpg'
        output_path = str(Path(output_dir) / output_filename)

        # Draw detections
        draw_detections(image_path, result, output_path, save=save, show=show)

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