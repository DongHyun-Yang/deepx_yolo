"""
YOLOv11 OBB Standalone DXNN Inference

A completely self-contained implementation with zero Ultralytics dependencies.
All necessary functions and classes are ported directly into this single file.

Ported classes from Ultralytics:
- OBB: Complete container for oriented bounding boxes with all attributes
  * xywhr, xyxyxyxy, xyxy: Various box formats
  * conf, cls: Confidence, class labels
  * shape: Box properties
  * cpu(), numpy(), cuda(), to(): Device management methods
- Results: Container for inference results

Ported functions from Ultralytics:
- empty_like: Create empty tensor/array with same shape
- xywh2xyxy: Convert box format from center to corner
- xyxy2xywh: Convert box format from corner to center
- xywhr2xyxyxyxy: Convert rotated boxes to corner points
- regularize_rboxes: Regularize rotated boxes to [0, pi/2]
- clip_boxes: Clip boxes to image boundaries
- scale_boxes: Rescale boxes from letterbox to original image size
- batch_probiou: Calculate probabilistic IoU between rotated boxes
- nms_rotated: NMS implementation for rotated boxes
- non_max_suppression: Main NMS function with confidence filtering

External dependencies: cv2, numpy, torch, dx_engine
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import time
import sys
import math

# No external dependencies from Ultralytics!
# All necessary classes are ported below

# Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'dxnn'
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
INPUT_SIZE = 1024

# DOTAv1.0 class names (dataset that YOLOv11-obb was trained on)
CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 
           'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 
           'roundabout', 'soccer-ball-field', 'swimming-pool']

# ============================================================================
# Ported classes and functions from Ultralytics (to avoid external dependencies)
# ============================================================================

class OBB:
    """
    A class for managing and manipulating Oriented Bounding Boxes (OBB).
    Ported from ultralytics.engine.results.OBB
    
    Supports various OBB formats and provides methods for easy manipulation and conversion.
    
    Attributes:
        data (torch.Tensor | np.ndarray): Raw tensor containing OBB data
        orig_shape (tuple[int, int]): Original image shape (height, width)
        xywhr (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height, rotation] format
        conf (torch.Tensor | np.ndarray): Confidence scores for each box
        cls (torch.Tensor | np.ndarray): Class labels for each box
        xyxyxyxy (torch.Tensor | np.ndarray): Rotated boxes as 4 corner points
        xyxy (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in [x1, y1, x2, y2] format
    """
    
    def __init__(self, boxes, orig_shape):
        """
        Initialize OBB with detection data.
        
        Args:
            boxes (torch.Tensor | np.ndarray): Tensor of shape (n, 7)
                Format: [x, y, w, h, rotation, confidence, class]
            orig_shape (tuple[int, int]): Original image shape (height, width)
        """
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        
        assert boxes.shape[-1] == 7, f"expected 7 values but got {boxes.shape[-1]}"
        
        self.data = boxes
        self.orig_shape = orig_shape
    
    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape
    
    def __len__(self):
        """Return the number of boxes."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a subset of boxes."""
        return OBB(self.data[idx], self.orig_shape)
    
    @property
    def xywhr(self):
        """
        Return oriented boxes in [x_center, y_center, width, height, rotation] format.
        
        Returns:
            (torch.Tensor | np.ndarray): Boxes with shape (n, 5)
        """
        return self.data[:, :5]
    
    @property
    def conf(self):
        """
        Return confidence scores for each box.
        
        Returns:
            (torch.Tensor | np.ndarray): Confidence scores with shape (n,)
        """
        return self.data[:, 5]
    
    @property
    def cls(self):
        """
        Return class labels for each box.
        
        Returns:
            (torch.Tensor | np.ndarray): Class IDs with shape (n,)
        """
        return self.data[:, 6]
    
    @property
    def xyxyxyxy(self):
        """
        Return rotated boxes as 4 corner points.
        
        Returns:
            (torch.Tensor | np.ndarray): Corner points with shape (n, 4, 2) -> reshaped to (n, 8)
        """
        return xywhr2xyxyxyxy(self.xywhr).view(len(self.xywhr), -1)
    
    @property
    def xyxy(self):
        """
        Return axis-aligned bounding boxes in [x1, y1, x2, y2] format.
        
        Returns:
            (torch.Tensor | np.ndarray): Axis-aligned boxes with shape (n, 4)
        """
        corners = self.xyxyxyxy.view(-1, 4, 2)
        x_coords = corners[..., 0]
        y_coords = corners[..., 1]
        x1 = x_coords.min(dim=1)[0]
        y1 = y_coords.min(dim=1)[0]
        x2 = x_coords.max(dim=1)[0]
        y2 = y_coords.max(dim=1)[0]
        
        if isinstance(self.data, torch.Tensor):
            return torch.stack([x1, y1, x2, y2], dim=-1)
        else:
            return np.stack([x1, y1, x2, y2], axis=-1)
    
    def cpu(self):
        """Return a copy with tensors on CPU memory."""
        return OBB(self.data.cpu() if isinstance(self.data, torch.Tensor) else self.data, self.orig_shape)
    
    def numpy(self):
        """Return a copy with numpy arrays."""
        data = self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data
        return OBB(data, self.orig_shape)
    
    def cuda(self):
        """Return a copy with tensors on GPU memory."""
        return OBB(self.data.cuda() if isinstance(self.data, torch.Tensor) else torch.from_numpy(self.data).cuda(), self.orig_shape)
    
    def to(self, device):
        """
        Move boxes to specified device.
        
        Args:
            device: Target device (e.g., 'cpu', 'cuda')
        
        Returns:
            (OBB): New OBB object on specified device
        """
        if isinstance(self.data, torch.Tensor):
            return OBB(self.data.to(device), self.orig_shape)
        else:
            return OBB(torch.from_numpy(self.data).to(device), self.orig_shape)

class Results:
    """
    A simplified class for storing OBB inference results.
    Ported from ultralytics.engine.results.Results
    
    This class stores OBB detection results and provides access to oriented boxes, class names, etc.
    Only includes attributes and methods used in this script.
    
    Attributes:
        orig_img: Original image as numpy array
        orig_shape: Original image shape (height, width)
        obb: OBB object containing oriented box detections
        names: Dictionary mapping class IDs to names
        path: Path to the image file
    """
    
    def __init__(self, orig_img, path, names, obb=None):
        """
        Initialize Results object.
        
        Args:
            orig_img: Original image as numpy array
            path: Path to image file (can be None)
            names: Dictionary mapping class IDs to class names
            obb: Tensor of shape (n, 7) with [x, y, w, h, rotation, conf, cls] or None
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.names = names
        self.path = path
    
    def __len__(self):
        """Return the number of detections."""
        return len(self.obb) if self.obb is not None else 0
    
    def __getitem__(self, idx):
        """Return a subset of results."""
        if self.obb is None:
            return Results(self.orig_img, self.path, self.names, obb=None)
        return Results(self.orig_img, self.path, self.names, obb=self.obb[idx].data)

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


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4] format.
    Ported from ultralytics.utils.ops.xywhr2xyxyxyxy

    Args:
        x (np.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5).
            Rotation values should be in radians from 0 to pi/2.

    Returns:
        (np.ndarray | torch.Tensor): Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def regularize_rboxes(rboxes):
    """
    Regularize rotated bounding boxes to range [0, pi/2].
    Ported from ultralytics.utils.ops.regularize_rboxes

    Args:
        rboxes (torch.Tensor): Input rotated boxes with shape (N, 5) in xywhr format.

    Returns:
        (torch.Tensor): Regularized rotated boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries.
    Ported from ultralytics.utils.ops.clip_boxes
    """
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, w)  # x
        boxes[..., 1].clamp_(0, h)  # y
        boxes[..., 2].clamp_(0, w)  # w
        boxes[..., 3].clamp_(0, h)  # h
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x, w
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y, h
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescale bounding boxes from one image shape to another.
    For OBB: handles xywh format (center-based)
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


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple:
    """
    Generate covariance matrix from oriented bounding boxes.
    Ported from ultralytics.utils.metrics._get_covariance_matrix

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (tuple): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.
    Ported from ultralytics.utils.metrics.batch_probiou

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def nms_rotated(boxes, scores, iou_threshold):
    """
    NMS for rotated boxes using batch_probiou.
    Ported from Ultralytics TorchNMS for rotated boxes.
    
    Args:
        boxes: Rotated boxes in xywhr format with shape (N, 5)
        scores: Confidence scores with shape (N,)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
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
        
        # Calculate IoU with remaining boxes using probiou
        ious = batch_probiou(boxes[idx:idx+1], boxes[sorted_indices[1:]])[0]
        
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
    rotated=True,
):
    """
    Perform non-maximum suppression (NMS) on OBB prediction results.
    Ported from ultralytics.utils.nms.non_max_suppression with OBB support
    
    Args:
        prediction: Predictions with shape (batch_size, num_classes + 4 + 1, num_boxes)
            For OBB: (1, 20, 21504) where 20 = 4(xywh) + 15(classes) + 1(angle)
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
        rotated: Whether to use rotated box IoU (True for OBB)
    
    Returns:
        List of detections per image with shape (num_boxes, 7) = [x, y, w, h, conf, cls, angle]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"
    
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)
    
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4 - 1)  # number of classes (subtract 4 for bbox, 1 for angle)
    mi = 4 + nc  # mask/angle start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    
    # Settings
    max_time_img = 0.05
    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1
    
    prediction = prediction.transpose(-1, -2)  # shape(1,20,21504) to shape(1,21504,20)
    # Don't convert xywh to xyxy for rotated boxes, keep as xywh + angle
    
    t = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * bs  # 7 for xywh + conf + cls + angle
    
    for xi, x in enumerate(prediction):  # image index, preds
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # Detections matrix: box(xywh), angle, conf, cls
        box, cls, angle = x.split((4, nc, 1), 1)
        
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), angle[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), angle), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        # NMS for rotated boxes
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        rboxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, 6:7]), dim=-1)  # xywhr for IoU calculation
        scores = x[:, 4]  # scores
        
        # Use rotated NMS
        i = nms_rotated(rboxes, scores, iou_thres)
        
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        
        if (time.time() - t) > time_limit:
            print(f"NMS time limit {time_limit:.3f}s exceeded")
            break
    
    return output

# ============================================================================
# End of ported functions
# ============================================================================

def letterbox(image, new_shape=(1024, 1024), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=64):
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

def preprocess_image(image_path, imgsz=1024):
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
    Postprocess Model output using ported utilities.
    Uses ported non_max_suppression and coordinate scaling functions for OBB.
    
    Args:
        preds: Raw Model output tensor [1, 20, 21504] for YOLO11-OBB
            where 20 = 4(xywh) + 15(classes) + 1(angle)
        orig_img: Original image (BGR)

    Returns:
        Results: Results object with OBB detections
    """
    # Convert to torch tensor if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    
    print(f"\n[Postprocess] Input shape: {preds.shape}")
    print(f"[Postprocess] Input range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    # Apply Non-Maximum Suppression for OBB
    # OBB NMS expects [batch, num_classes + 4 + 1, num_detections]
    # For OBB: [1, 20, 21504] where 20 = 4(bbox) + 15(classes) + 1(angle)
    results = non_max_suppression(
        preds,
        conf_thres=CONFIDENCE_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=15,  # 15 classes for DOTA dataset
        rotated=True,  # Critical for OBB
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
            obb=None
        )

    pred = results[0]  # [num_detections, 7] = [x, y, w, h, conf, cls, angle]

    # Create pseudo preprocessed image shape for scale_boxes
    preproc_shape = (INPUT_SIZE, INPUT_SIZE)

    # Extract components
    # OBB format from NMS: [x, y, w, h, conf, cls, angle]
    rboxes = pred[:, :4]  # [x, y, w, h]
    confs = pred[:, 4]    # confidence
    clss = pred[:, 5]     # class
    angles = pred[:, 6:7] # angle

    # Scale boxes from letterbox size to original image size
    # For OBB, we only scale the center point and dimensions (xywh)
    rboxes = scale_boxes(preproc_shape, rboxes, orig_img.shape, xywh=True)

    # Regularize rotated boxes (normalize angle to [0, pi/2])
    rboxes_with_angle = torch.cat([rboxes, angles], dim=-1)
    rboxes_regularized = regularize_rboxes(rboxes_with_angle)

    # Combine with confidence and class for Results object
    # Results expects: [x, y, w, h, angle, conf, cls]
    obb_data = torch.cat([rboxes_regularized, confs.unsqueeze(1), clss.unsqueeze(1)], dim=-1)

    print(f"[Postprocess] OBB data shape: {obb_data.shape}")
    print(f"[Postprocess] Confidence range: {confs.min():.3f} ~ {confs.max():.3f}")

    # Create Results object
    result = Results(
        orig_img=orig_img,
        path=None,
        names={i: name for i, name in enumerate(CLASSES)},
        obb=obb_data  # [n, 7] tensor: [x, y, w, h, angle, conf, cls]
    )

    return result

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

def run_inference_using_dxnn(model_path, input_tensor):
    """Run inference with DXNN model using dx_engine."""

    if not isinstance(model_path, str):
        model_path = str(model_path)

    # Initialize InferenceEngine
    from dx_engine import InferenceEngine
    engine = InferenceEngine(model_path)
    
    # Convert input tensor to uint8 format as expected by DEEPX
    # input_tensor is in range [0, 1], convert to [0, 255]
    im_np = (input_tensor * 255).astype("uint8")
    
    # Convert from NCHW (Batch, Channel, Height, Width) to HWC (Height, Width, Channel)
    if len(im_np.shape) == 4:  # NCHW format
        im_np = np.squeeze(im_np, axis=0)  # Remove batch dimension (N=1)
        im_np = np.transpose(im_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    
    # Prepare input data as list
    input_data = [im_np]
    
    # Run inference using DEEPX InferenceEngine
    outputs = engine.run(input_data)
    
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
        outputs = run_inference_using_dxnn(model_path, input_tensor)
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