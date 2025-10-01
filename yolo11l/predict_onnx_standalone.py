"""
YOLOv11 Standalone ONNX Inference

A completely self-contained implementation with zero Ultralytics dependencies.
All necessary functions and classes are ported directly into this single file.

Ported classes from Ultralytics:
- Boxes: Complete container for detection boxes with all attributes
  * xyxy, xywh, xyxyn, xywhn: Various box formats
  * conf, cls, id: Confidence, class labels, tracking IDs
  * shape, is_track: Box properties
  * cpu(), numpy(), cuda(), to(): Device management methods
- Results: Container for inference results

Ported functions from Ultralytics:
- empty_like: Create empty tensor/array with same shape
- xywh2xyxy: Convert box format from center to corner
- xyxy2xywh: Convert box format from corner to center
- clip_boxes: Clip boxes to image boundaries
- scale_boxes: Rescale boxes from letterbox to original image size
- box_iou_torch: Calculate IoU between boxes for NMS
- nms_torch: Custom NMS implementation
- non_max_suppression: Main NMS function with confidence filtering

External dependencies: cv2, numpy, torch, onnxruntime
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import time
import sys

# No external dependencies from Ultralytics!
# All necessary classes are ported below

# Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'onnx'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'boats.jpg'
SOURCE_PATH = PROJECT_ROOT / 'assets'
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "standalone"
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

# ============================================================================
# Ported classes and functions from Ultralytics (to avoid external dependencies)
# ============================================================================

class Boxes:
    """
    A class for managing and manipulating detection boxes.
    Ported from ultralytics.engine.results.Boxes
    
    Supports various box formats and provides methods for easy manipulation and conversion
    between different coordinate systems.
    
    Attributes:
        data (torch.Tensor | np.ndarray): Raw tensor containing detection boxes and associated data
        orig_shape (tuple[int, int]): Original image shape (height, width)
        is_track (bool): Whether tracking IDs are included in the box data
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format
        conf (torch.Tensor | np.ndarray): Confidence scores for each box
        cls (torch.Tensor | np.ndarray): Class labels for each box
        id (torch.Tensor | None): Tracking IDs for each box (if available)
        xywh (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height] format
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape
        xywhn (torch.Tensor | np.ndarray): Normalized [x_center, y_center, w, h] boxes relative to orig_shape
    """
    
    def __init__(self, boxes, orig_shape):
        """
        Initialize Boxes with detection data.
        
        Args:
            boxes (torch.Tensor | np.ndarray): Tensor of shape (n, 6) or (n, 7)
                Format (6 values): [x1, y1, x2, y2, confidence, class]
                Format (7 values): [x1, y1, x2, y2, confidence, class, track_id]
            orig_shape (tuple[int, int]): Original image shape (height, width)
        """
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"
        
        self.data = boxes
        self.orig_shape = orig_shape
        self.is_track = n == 7
    
    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape
    
    def __len__(self):
        """Return the number of boxes."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a subset of boxes."""
        return Boxes(self.data[idx], self.orig_shape)
    
    @property
    def xyxy(self):
        """
        Return boxes in [x1, y1, x2, y2] format.
        
        Returns:
            (torch.Tensor | np.ndarray): Boxes with shape (n, 4)
        """
        return self.data[:, :4]
    
    @property
    def conf(self):
        """
        Return confidence scores for each box.
        
        Returns:
            (torch.Tensor | np.ndarray): Confidence scores with shape (n,)
        """
        return self.data[:, -2]
    
    @property
    def cls(self):
        """
        Return class labels for each box.
        
        Returns:
            (torch.Tensor | np.ndarray): Class IDs with shape (n,)
        """
        return self.data[:, -1]
    
    @property
    def id(self):
        """
        Return tracking IDs for each box (if available).
        
        Returns:
            (torch.Tensor | None): Tracking IDs with shape (n,) or None
        """
        return self.data[:, -3] if self.is_track else None
    
    @property
    def xywh(self):
        """
        Return boxes in [x_center, y_center, width, height] format.
        
        Returns:
            (torch.Tensor | np.ndarray): Boxes with shape (n, 4)
        """
        return xyxy2xywh(self.xyxy)
    
    @property
    def xyxyn(self):
        """
        Return normalized boxes in [x1, y1, x2, y2] format.
        Values are normalized to [0, 1] relative to original image size.
        
        Returns:
            (torch.Tensor | np.ndarray): Normalized boxes with shape (n, 4)
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]  # width
        xyxy[..., [1, 3]] /= self.orig_shape[0]  # height
        return xyxy
    
    @property
    def xywhn(self):
        """
        Return normalized boxes in [x_center, y_center, width, height] format.
        Values are normalized to [0, 1] relative to original image size.
        
        Returns:
            (torch.Tensor | np.ndarray): Normalized boxes with shape (n, 4)
        """
        xywh = xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]  # width
        xywh[..., [1, 3]] /= self.orig_shape[0]  # height
        return xywh
    
    def cpu(self):
        """Return a copy with tensors on CPU memory."""
        return Boxes(self.data.cpu() if isinstance(self.data, torch.Tensor) else self.data, self.orig_shape)
    
    def numpy(self):
        """Return a copy with numpy arrays."""
        data = self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data
        return Boxes(data, self.orig_shape)
    
    def cuda(self):
        """Return a copy with tensors on GPU memory."""
        return Boxes(self.data.cuda() if isinstance(self.data, torch.Tensor) else torch.from_numpy(self.data).cuda(), self.orig_shape)
    
    def to(self, device):
        """
        Move boxes to specified device.
        
        Args:
            device: Target device (e.g., 'cpu', 'cuda')
        
        Returns:
            (Boxes): New Boxes object on specified device
        """
        if isinstance(self.data, torch.Tensor):
            return Boxes(self.data.to(device), self.orig_shape)
        else:
            return Boxes(torch.from_numpy(self.data).to(device), self.orig_shape)


class Results:
    """
    A simplified class for storing inference results.
    Ported from ultralytics.engine.results.Results
    
    This class stores detection results and provides access to boxes, class names, etc.
    Only includes attributes and methods used in this script.
    
    Attributes:
        orig_img: Original image as numpy array
        orig_shape: Original image shape (height, width)
        boxes: Boxes object containing detections
        names: Dictionary mapping class IDs to names
        path: Path to the image file
    """
    
    def __init__(self, orig_img, path, names, boxes=None):
        """
        Initialize Results object.
        
        Args:
            orig_img: Original image as numpy array
            path: Path to image file (can be None)
            names: Dictionary mapping class IDs to class names
            boxes: Tensor of shape (n, 6) with [x1, y1, x2, y2, conf, cls] or None
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.names = names
        self.path = path
    
    def __len__(self):
        """Return the number of detections."""
        return len(self.boxes) if self.boxes is not None else 0
    
    def __getitem__(self, idx):
        """Return a subset of results."""
        if self.boxes is None:
            return Results(self.orig_img, self.path, self.names, boxes=None)
        return Results(self.orig_img, self.path, self.names, boxes=self.boxes[idx].data)


def empty_like(x):
    """
    Create empty tensor or array with same shape as input.
    Ported from ultralytics.utils.ops.empty_like
    """
    if isinstance(x, torch.Tensor):
        return torch.empty_like(x, dtype=torch.float32)
    else:
        return np.empty_like(x, dtype=np.float32)


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2) format.
    Ported from ultralytics.utils.ops.xywh2xyxy
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) to (x, y, width, height) format.
    Ported from ultralytics.utils.ops.xyxy2xywh

    Args:
        x: Bounding boxes in (x1, y1, x2, y2) format
    
    Returns:
        Bounding boxes in (x_center, y_center, width, height) format
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries.
    Ported from ultralytics.utils.ops.clip_boxes
    """
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, w)  # x1
        boxes[..., 1].clamp_(0, h)  # y1
        boxes[..., 2].clamp_(0, w)  # x2
        boxes[..., 3].clamp_(0, h)  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescale bounding boxes from one image shape to another.
    Ported from ultralytics.utils.ops.scale_boxes
    
    Args:
        img1_shape: Shape of the source image (height, width)
        boxes: Bounding boxes to rescale
        img0_shape: Shape of the target image (height, width)
        ratio_pad: Tuple of (ratio, pad) for scaling
        padding: Whether boxes are based on YOLO-style augmented images with padding
        xywh: Whether box format is xywh (True) or xyxy (False)
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        if not xywh:
            boxes[..., 2] -= pad_x  # x padding
            boxes[..., 3] -= pad_y  # y padding
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def box_iou_torch(box1, box2, eps=1e-7):
    """
    Calculate IoU between two sets of boxes.
    Simple implementation for NMS.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2.T) - torch.max(b1_x1, b2_x1.T)).clamp(0) * \
            (torch.min(b1_y2, b2_y2.T) - torch.max(b1_y1, b2_y1.T)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + (w2 * h2).T - inter + eps
    
    return inter / union


def nms_torch(boxes, scores, iou_threshold):
    """
    Simple NMS implementation using torch operations.
    Ported from ultralytics TorchNMS.nms
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while sorted_indices.numel() > 0:
        # Pick the box with highest score
        idx = sorted_indices[0]
        keep.append(idx)
        
        if sorted_indices.numel() == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = box_iou_torch(boxes[idx:idx+1], boxes[sorted_indices[1:]])[0]
        
        # Keep boxes with IoU less than threshold
        sorted_indices = sorted_indices[1:][ious <= iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on prediction results.
    Simplified version ported from ultralytics.utils.nms.non_max_suppression
    
    Args:
        prediction: Predictions with shape (batch_size, num_classes + 4, num_boxes)
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: List of class indices to consider
        agnostic: Whether to perform class-agnostic NMS
        multi_label: Whether each box can have multiple labels
        labels: A priori labels for each image
        max_det: Maximum number of detections to keep
        nc: Number of classes
        max_nms: Maximum number of boxes for NMS
        max_wh: Maximum box width and height
    
    Returns:
        List of detections per image with shape (num_boxes, 6)
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"
    
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)
    
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    
    # Settings
    max_time_img = 0.05
    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1
    
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, preds
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls = x.split((4, nc), 1)
        
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        # NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes = x[:, :4] + c  # boxes (offset by class)
        scores = x[:, 4]  # scores
        
        # Use torchvision.ops.nms if available, otherwise use custom implementation
        if "torchvision" in sys.modules:
            import torchvision
            i = torchvision.ops.nms(boxes, scores, iou_thres)
        else:
            i = nms_torch(boxes, scores, iou_thres)
        
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        
        if (time.time() - t) > time_limit:
            print(f"NMS time limit {time_limit:.3f}s exceeded")
            break
    
    return output

# ============================================================================
# End of ported functions
# ============================================================================

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

def postprocess_output(preds, orig_img):
    """
    Postprocess Model output using ported utilities from Ultralytics.
    Uses ported non_max_suppression and coordinate scaling functions.
    
    Args:
        preds: Raw Model output tensor [1, 84, 8400] for YOLO11
        orig_img: Original image (BGR) - used for shape extraction

    Returns:
        Results: Results object with box detections
    """
    # Convert to torch tensor if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    
    print(f"\n[Postprocess] Input shape: {preds.shape}")
    print(f"[Postprocess] Input range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    # Apply Non-Maximum Suppression using Ultralytics
    # Ultralytics NMS expects [batch, num_classes + 4, num_detections]
    # Current format: [1, 84, 8400] where 84 = 4(bbox) + 80(classes)
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
            boxes=None
        )
    
    pred = results[0]  # [num_detections, 6] = [x1, y1, x2, y2, conf, cls]
    
    print(f"[Postprocess] Pred shape before scale: {pred.shape}")
    print(f"[Postprocess] Pred sample (first detection): {pred[0] if len(pred) > 0 else 'none'}")
    
    # Scale boxes from letterbox size to original image size
    # Input size is 640x640
    preproc_shape = (INPUT_SIZE, INPUT_SIZE)
    pred[:, :4] = scale_boxes(preproc_shape, pred[:, :4], orig_img.shape)
    
    print(f"[Postprocess] Box data shape after scale: {pred.shape}")
    print(f"[Postprocess] Confidence range: {pred[:, 4].min():.3f} ~ {pred[:, 4].max():.3f}")
    
    # Create Results object
    # Results expects boxes in format: [n, 6] where each row is [x1, y1, x2, y2, conf, cls]
    result = Results(
        orig_img=orig_img,
        path=None,
        names={i: name for i, name in enumerate(CLASSES)},
        boxes=pred  # [n, 6] tensor: [x1, y1, x2, y2, conf, cls]
    )
    
    return result

def draw_detections(source_path, result, output_path, save=True, show=True):
    """
    Draw bounding boxes on image using Ultralytics Results object.
    Custom visualization implementation using OpenCV.
    """
    image = cv2.imread(source_path)
    
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy  # x1, y1, x2, y2 format
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

        # Color palette (different colors for each class)
        colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        for i in range(len(xyxy)):
            box = xyxy[i].cpu().numpy().astype(int)
            conf = confs[i].item()
            name = names[i]
            class_id = result.boxes.cls[i].item()
            
            x1, y1, x2, y2 = box
            
            # Get class-specific color
            color = colors[int(class_id)].tolist()
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, image)
        print(f"Detection result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("Object Detections", image)
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
    
    # Check class distribution
    classes = result.boxes.cls.cpu().numpy()
    unique_classes, counts = np.unique(classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
    
    # More detailed conf analysis
    conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(conf_bins)-1):
        count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
        print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
    print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
    
    # Detailed detection info
    xyxy = result.boxes.xyxy
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    confs = result.boxes.conf
    
    print(f"[{filename}] Total {len(result.boxes)} objects detected.")
    for i in range(len(result.boxes)):
        class_name = names[i]
        score = confs[i].item()
        box = xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = box
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f})")

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
        preds = outputs[0]  # Get first (and only) result

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
        result = postprocess_output(preds, orig_img)

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