"""
YOLOv11 Pose Estimation Standalone ONNX Inference

A completely self-contained implementation with zero Ultralytics dependencies.
All necessary functions and classes are ported directly into this single file.

Ported classes from Ultralytics:
- Boxes: Complete container for detection boxes with all attributes
  * xyxy, xywh, xyxyn, xywhn: Various box formats
  * conf, cls, id: Confidence, class labels, tracking IDs
  * shape, is_track: Box properties
  * cpu(), numpy(), cuda(), to(): Device management methods
- Keypoints: Container for pose keypoints with 17 COCO keypoint format
  * xy, xyn, conf: Keypoint coordinates and confidence
- Results: Container for inference results with keypoints support

Ported functions from Ultralytics:
- clip_boxes: Clip bounding boxes to image boundaries
- scale_boxes: Rescale boxes from letterbox to original image coordinates
- scale_coords: Rescale keypoint coordinates from letterbox to original image
- empty_like: Create empty tensor/array with same shape
- xywh2xyxy: Convert box format from center (cx, cy, w, h) to corner (x1, y1, x2, y2)
- xyxy2xywh: Convert box format from corner (x1, y1, x2, y2) to center (cx, cy, w, h)
- box_iou_torch: Calculate IoU between two sets of boxes
- nms_torch: Perform Non-Maximum Suppression on boxes
- non_max_suppression: Full NMS implementation with multi-class support and pose keypoints

External dependencies: cv2, numpy, torch, onnxruntime
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# No external dependencies from Ultralytics
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

# ============================================================================
# Ported classes and functions from Ultralytics (to avoid external dependencies)
# ============================================================================

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
        img1_shape: Shape of the source image (height, width) or full shape
        boxes: Bounding boxes to rescale
        img0_shape: Shape of the target image (height, width) or full shape
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

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale coordinates from one image shape to another.
    Ported from ultralytics.utils.ops.scale_coords
    
    Used for scaling keypoint coordinates from letterbox to original image size.
    
    Args:
        img1_shape: Shape of the source image (height, width)
        coords: Coordinates to rescale, shape (n, 2)
        img0_shape: Shape of the target image (height, width)
        ratio_pad: Tuple of (ratio, pad) for scaling
        normalize: Whether to normalize coordinates
        padding: Whether coords are based on YOLO-style augmented images with padding
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad_x  # x padding
        coords[..., 1] -= pad_y  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    
    return coords

def empty_like(x):
    """
    Create empty tensor or array with same shape and type as input.
    Ported from ultralytics.utils.ops.empty_like
    """
    if isinstance(x, torch.Tensor):
        return torch.empty_like(x, dtype=torch.float32)
    else:
        return np.empty_like(x, dtype=np.float32)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from center format to corner format.
    Ported from ultralytics.utils.ops.xywh2xyxy
    
    Args:
        x (torch.Tensor | np.ndarray): Boxes in xywh format, shape (..., 4)
    
    Returns:
        (torch.Tensor | np.ndarray): Boxes in xyxy format, shape (..., 4)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    xy = x[..., :2]  # centers (cx, cy)
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from corner format to center format.
    Ported from ultralytics.utils.ops.xyxy2xywh
    
    Args:
        x (torch.Tensor | np.ndarray): Boxes in xyxy format, shape (..., 4)
    
    Returns:
        (torch.Tensor | np.ndarray): Boxes in xywh format, shape (..., 4)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y

def box_iou_torch(box1, box2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    Ported from ultralytics for NMS implementation.
    
    Args:
        box1 (torch.Tensor): First set of boxes, shape (N, 4) in xyxy format
        box2 (torch.Tensor): Second set of boxes, shape (M, 4) in xyxy format
        eps (float): Small epsilon value to prevent division by zero
    
    Returns:
        (torch.Tensor): IoU matrix of shape (N, M)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Intersection area
    inter = (torch.min(b1_x2[:, None], b2_x2) - torch.max(b1_x1[:, None], b2_x1)).clamp(0) * \
            (torch.min(b1_y2[:, None], b2_y2) - torch.max(b1_y1[:, None], b2_y1)).clamp(0)
    
    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1[:, None] * h1[:, None] + w2 * h2 - inter + eps
    
    return inter / union

def nms_torch(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on a single set of boxes.
    Ported from ultralytics NMS implementation.
    
    Args:
        boxes (torch.Tensor): Boxes in xyxy format, shape (N, 4)
        scores (torch.Tensor): Confidence scores for each box, shape (N,)
        iou_threshold (float): IoU threshold for suppression
    
    Returns:
        (torch.Tensor): Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        ious = box_iou_torch(boxes[current:current+1], boxes[indices[1:]])[0]
        indices = indices[1:][ious < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long)

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes with pose keypoints.
    Ported from ultralytics.utils.nms.non_max_suppression
    
    Args:
        prediction (torch.Tensor): Tensor of shape (batch_size, num_boxes, num_classes + 4 + num_keypoints)
            For pose: shape is (1, 56, 8400) where 56 = 4(bbox_xywh) + 1(conf) + 51(17 kpts * 3)
        conf_thres (float): Minimum confidence threshold
        iou_thres (float): IoU threshold for NMS
        classes (List[int]): List of class indices to consider
        agnostic (bool): If True, class-agnostic NMS
        multi_label (bool): If True, one box can have multiple labels
        labels (List): External labels
        max_det (int): Maximum number of detections to keep
        nc (int): Number of classes output by the model
        max_nms (int): Maximum number of boxes to process in NMS
        max_wh (int): Maximum box width and height in pixels
    
    Returns:
        (List[torch.Tensor]): List of tensors, one per image, shape (num_detections, 6 + 51)
            Format: [x1, y1, x2, y2, conf, cls, kpt_x1, kpt_y1, kpt_conf1, ...]
    """
    import time
    
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of additional values (keypoints: 51)
    mi = 4 + nc  # mask/keypoints start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    
    # Settings
    max_wh = 7680
    max_nms = 30000
    time_limit = 2.0 + 0.05 * bs
    multi_label &= nc > 1
    
    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence filtering
        
        if not x.shape[0]:
            continue
        
        # Split detections: box is in xywh format
        box, cls, kpts = x.split((4, nc, nm), 1)
        
        # Convert box coordinates from xywh to xyxy format
        box = xywh2xyxy(box)
        
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), kpts[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        # Check shape
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms_torch(boxes, scores, iou_thres)
        i = i[:max_det]
        
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break
    
    return output

class Boxes:
    """
    A class for managing and manipulating detection boxes.
    Ported from ultralytics.engine.results.Boxes
    
    Attributes:
        data (torch.Tensor | np.ndarray): Raw tensor containing detection boxes
        orig_shape (tuple[int, int]): Original image shape (height, width)
        is_track (bool): Whether tracking IDs are included
        xyxy, conf, cls, id, xywh, xyxyn, xywhn: Various box representations
    """
    
    def __init__(self, boxes, orig_shape):
        """Initialize Boxes with detection data."""
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
        return self.data.shape
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return Boxes(self.data[idx], self.orig_shape)
    
    @property
    def xyxy(self):
        """Return boxes in [x1, y1, x2, y2] format."""
        return self.data[:, :4]
    
    @property
    def conf(self):
        """Return confidence scores."""
        return self.data[:, -2]
    
    @property
    def cls(self):
        """Return class labels."""
        return self.data[:, -1]
    
    @property
    def id(self):
        """Return tracking IDs (if available)."""
        return self.data[:, -3] if self.is_track else None
    
    @property
    def xywh(self):
        """Return boxes in [x_center, y_center, width, height] format."""
        return xyxy2xywh(self.xyxy)
    
    @property
    def xyxyn(self):
        """Return normalized boxes in [x1, y1, x2, y2] format."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]  # width
        xyxy[..., [1, 3]] /= self.orig_shape[0]  # height
        return xyxy
    
    @property
    def xywhn(self):
        """Return normalized boxes in [x_center, y_center, width, height] format."""
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
        """Move boxes to specified device."""
        if isinstance(self.data, torch.Tensor):
            return Boxes(self.data.to(device), self.orig_shape)
        else:
            return Boxes(torch.from_numpy(self.data).to(device), self.orig_shape)

class Keypoints:
    """
    A class for managing pose keypoints (17 COCO keypoints).
    Ported from ultralytics.engine.results.Keypoints
    
    Attributes:
        data (torch.Tensor): Raw keypoint tensor, shape (num_persons, 17, 3) where 3 = [x, y, conf]
        orig_shape (tuple): Original image shape (height, width)
        xy (torch.Tensor): Keypoint coordinates in pixels, shape (num_persons, 17, 2)
        xyn (torch.Tensor): Normalized keypoint coordinates, shape (num_persons, 17, 2)
        conf (torch.Tensor): Keypoint confidence scores, shape (num_persons, 17)
    """
    
    def __init__(self, keypoints, orig_shape):
        """
        Initialize Keypoints with pose data.
        
        Args:
            keypoints (torch.Tensor): Keypoint tensor, shape (num_persons, 17, 3)
            orig_shape (tuple): Original image shape (height, width)
        """
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints)
        
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :, :]
        
        self.data = keypoints
        self.orig_shape = orig_shape
    
    @property
    def shape(self):
        """Return the shape of keypoints data."""
        return self.data.shape
    
    def __len__(self):
        """Return number of pose instances."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return keypoints at index."""
        return Keypoints(self.data[idx], self.orig_shape)
    
    @property
    def xy(self):
        """Return keypoint coordinates in pixels, shape (num_persons, 17, 2)."""
        return self.data[..., :2]
    
    @property
    def xyn(self):
        """Return normalized keypoint coordinates, shape (num_persons, 17, 2)."""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]  # normalize x
        xy[..., 1] /= self.orig_shape[0]  # normalize y
        return xy
    
    @property
    def conf(self):
        """Return keypoint confidence scores, shape (num_persons, 17)."""
        return self.data[..., 2]
    
    def cpu(self):
        """Return a copy with tensors on CPU memory."""
        return Keypoints(self.data.cpu() if isinstance(self.data, torch.Tensor) else self.data, self.orig_shape)
    
    def numpy(self):
        """Return a copy with numpy arrays."""
        data = self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data
        return Keypoints(data, self.orig_shape)
    
    def cuda(self):
        """Return a copy with tensors on GPU memory."""
        return Keypoints(self.data.cuda() if isinstance(self.data, torch.Tensor) else torch.from_numpy(self.data).cuda(), self.orig_shape)
    
    def to(self, device):
        """Move keypoints to specified device."""
        if isinstance(self.data, torch.Tensor):
            return Keypoints(self.data.to(device), self.orig_shape)
        else:
            return Keypoints(torch.from_numpy(self.data).to(device), self.orig_shape)

class Results:
    """
    A simplified class for storing pose estimation results.
    Ported from ultralytics.engine.results.Results
    
    Attributes:
        orig_img (np.ndarray): Original image as BGR numpy array
        orig_shape (tuple): Original image shape (height, width)
        boxes (Boxes): Boxes object containing detection boxes
        keypoints (Keypoints): Keypoints object containing pose keypoints
        names (dict): Dictionary mapping class IDs to class names
        path (str): Path to the source image file
    """
    
    def __init__(self, orig_img, path, names, boxes=None, keypoints=None):
        """
        Initialize Results object with pose detection outputs.
        
        Args:
            orig_img (np.ndarray): Original image as BGR numpy array
            path (str): Path to image file
            names (dict): Dictionary mapping class IDs to class names
            boxes: Tensor of shape (n, 6) with [x1, y1, x2, y2, conf, cls] or None
            keypoints: Tensor of shape (n, 17, 3) with keypoint data or None
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.names = names
        self.path = path
    
    def __len__(self):
        """Return the number of detections."""
        return len(self.boxes) if self.boxes is not None else 0
    
    def __getitem__(self, idx):
        """Return a subset of results."""
        if self.boxes is None:
            return Results(self.orig_img, self.path, self.names, boxes=None, keypoints=None)
        
        new_keypoints = None
        if self.keypoints is not None:
            kpt_data = self.keypoints.data[idx]
            new_keypoints = kpt_data.cpu().numpy() if isinstance(kpt_data, torch.Tensor) else kpt_data
            
        return Results(self.orig_img, self.path, self.names, boxes=self.boxes[idx].data, keypoints=new_keypoints)

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

def postprocess_pose(preds, orig_img):
    """
    Postprocess pose model output using ported utilities from Ultralytics.
    Produces identical results to ultralytics_postprocess version.
    
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
    
    # Apply Non-Maximum Suppression using ported function
    # Input format: [1, 56, 8400] where 56 = 4(bbox_xywh) + 1(conf) + 51(17 kpts * 3)
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
    pred_boxes[:, :4] = scale_boxes(preproc_shape, pred_boxes[:, :4], orig_img.shape)
    
    # Scale keypoints from letterbox size to original image size
    # ops.scale_coords expects keypoints in format [num_kpts, num_points, 2]
    # We have [num_detections, 17, 3] so we need to scale x,y coordinates
    pred_kpts_xy = pred_kpts[..., :2]  # Extract x, y coordinates
    pred_kpts_conf = pred_kpts[..., 2:3]  # Extract confidence
    
    # Reshape for scale_coords: [num_detections * 17, 2]
    num_detections, num_kpts, _ = pred_kpts.shape
    pred_kpts_xy_flat = pred_kpts_xy.reshape(-1, 2)
    
    # Scale coordinates
    pred_kpts_xy_flat = scale_coords(preproc_shape, pred_kpts_xy_flat, orig_img.shape)
    
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

        # 3. Post-processing using ported utilities
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