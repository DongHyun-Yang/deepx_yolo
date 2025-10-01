"""
YOLOv11 Segmentation Standalone ONNX Inference

A completely self-contained implementation with zero Ultralytics dependencies.
All necessary functions and classes are ported directly into this single file.

Ported classes from Ultralytics:
- Boxes: Complete container for detection boxes with all attributes
  * xyxy, xywh, xyxyn, xywhn: Various box formats
  * conf, cls, id: Confidence, class labels, tracking IDs
  * shape, is_track: Box properties
  * cpu(), numpy(), cuda(), to(): Device management methods
- Results: Container for inference results with masks support

Ported functions from Ultralytics:
- clip_boxes: Clip bounding boxes to image boundaries
- scale_boxes: Rescale boxes from letterbox to original image coordinates
- crop_mask: Crop masks to bounding box regions
- process_mask: Apply mask coefficients to prototype masks
- scale_masks: Rescale masks to target shape with padding
- empty_like: Create empty tensor/array with same shape
- xywh2xyxy: Convert box format from center (cx, cy, w, h) to corner (x1, y1, x2, y2)
- xyxy2xywh: Convert box format from corner (x1, y1, x2, y2) to center (cx, cy, w, h)
- box_iou_torch: Calculate IoU between two sets of boxes
- nms_torch: Perform Non-Maximum Suppression on boxes
- non_max_suppression: Full NMS implementation with multi-class support and mask handling

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

import torch.nn.functional as F

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

def crop_mask(masks, boxes):
    """
    Crop masks to bounding box regions.
    Ported from ultralytics.utils.ops.crop_mask
    
    Args:
        masks (torch.Tensor): Masks with shape (N, H, W).
        boxes (torch.Tensor): Bounding box coordinates with shape (N, 4) in relative point form.
    
    Returns:
        (torch.Tensor): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """
    Apply masks to bounding boxes using mask head output.
    Ported from ultralytics.utils.ops.process_mask
    
    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).
        upsample (bool): Whether to upsample masks to original image size.
    
    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)

def scale_masks(masks, shape, padding: bool = True):
    """
    Rescale segment masks to target shape.
    Ported from ultralytics.utils.ops.scale_masks
    
    Args:
        masks (torch.Tensor): Masks with shape (N, C, H, W).
        shape (tuple): Target height and width as (height, width).
        padding (bool): Whether masks are based on YOLO-style augmented images with padding.
    
    Returns:
        (torch.Tensor): Rescaled masks.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad_w = mw - shape[1] * gain
    pad_h = mh - shape[0] * gain
    if padding:
        pad_w /= 2
        pad_h /= 2
    top, left = (int(round(pad_h - 0.1)), int(round(pad_w - 0.1))) if padding else (0, 0)
    bottom = mh - int(round(pad_h + 0.1))
    right = mw - int(round(pad_w + 0.1))
    return F.interpolate(masks[..., top:bottom, left:right], shape, mode="bilinear", align_corners=False)  # NCHW masks

def empty_like(x):
    """
    Create empty tensor or array with same shape and type as input.
    Ported from ultralytics.utils.ops.empty_like
    
    Args:
        x (torch.Tensor | np.ndarray): Input tensor or array
        
    Returns:
        (torch.Tensor | np.ndarray): Empty tensor/array with same shape as input
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
            - x[..., 0]: center x coordinate
            - x[..., 1]: center y coordinate
            - x[..., 2]: box width
            - x[..., 3]: box height
    
    Returns:
        (torch.Tensor | np.ndarray): Boxes in xyxy format, shape (..., 4)
            - y[..., 0]: top-left x (x1)
            - y[..., 1]: top-left y (y1)
            - y[..., 2]: bottom-right x (x2)
            - y[..., 3]: bottom-right y (y2)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    xy = x[..., :2]  # centers (cx, cy)
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy (x1 = cx - w/2, y1 = cy - h/2)
    y[..., 2:] = xy + wh  # bottom right xy (x2 = cx + w/2, y2 = cy + h/2)
    return y


def box_iou_torch(box1, box2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    Ported from ultralytics for NMS implementation.
    
    This function computes the IoU matrix between two sets of bounding boxes in xyxy format.
    Used by NMS to determine which boxes overlap significantly.
    
    Args:
        box1 (torch.Tensor): First set of boxes, shape (N, 4) in xyxy format
        box2 (torch.Tensor): Second set of boxes, shape (M, 4) in xyxy format
        eps (float): Small epsilon value to prevent division by zero
    
    Returns:
        (torch.Tensor): IoU matrix of shape (N, M) where element [i, j] is the IoU 
                       between box1[i] and box2[j]
    """
    # box1: (N, 4), box2: (M, 4)
    # Get corner coordinates
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
    
    NMS eliminates redundant overlapping boxes by keeping only the highest-scoring
    box among boxes that have IoU greater than the threshold with each other.
    
    Algorithm:
    1. Sort boxes by confidence score (descending)
    2. Keep the highest scoring box
    3. Remove all boxes with IoU > threshold with the kept box
    4. Repeat until no boxes remain
    
    Args:
        boxes (torch.Tensor): Boxes in xyxy format, shape (N, 4)
        scores (torch.Tensor): Confidence scores for each box, shape (N,)
        iou_threshold (float): IoU threshold for suppression (typically 0.45)
    
    Returns:
        (torch.Tensor): Indices of boxes to keep, shape (K,) where K <= N
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Sort by score (descending order)
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(indices) > 0:
        # Keep highest scoring box
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU between current box and all remaining boxes
        ious = box_iou_torch(boxes[current:current+1], boxes[indices[1:]])[0]
        
        # Keep only boxes with IoU < threshold (non-overlapping boxes)
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
    Perform non-maximum suppression (NMS) on a set of boxes with associated scores and optional masks.
    Ported from ultralytics.utils.nms.non_max_suppression
    
    Key feature: Converts box coordinates from xywh (center format) to xyxy (corner format) 
    before applying NMS, which is crucial for correct IoU calculation.
    
    Args:
        prediction (torch.Tensor): Tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes in xywh format, scores, and optionally masks for each image.
            The boxes will be automatically converted to xyxy format internally.
        conf_thres (float): Minimum confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        classes (List[int]): List of class indices to consider.
        agnostic (bool): If True, class-agnostic NMS.
        multi_label (bool): If True, one box can have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): External labels.
        max_det (int): Maximum number of detections to keep.
        nc (int): Number of classes output by the model.
        max_nms (int): Maximum number of boxes to process in NMS.
        max_wh (int): Maximum box width and height in pixels.
    
    Returns:
        (List[torch.Tensor]): List of tensors, one per image in batch, where each tensor has shape
            (num_detections, 6 + num_masks) and contains (x1, y1, x2, y2, conf, class_id, ...mask_coeffs).
    """
    import time
    
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    
    # Settings
    max_wh = 7680
    max_nms = 30000
    time_limit = 2.0 + 0.05 * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    
    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence filtering
        
        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # Split detections: box is in xywh format (center_x, center_y, width, height) from model output
        box, cls, mask = x.split((4, nc, nm), 1)
        
        # Convert box coordinates from xywh to xyxy format for proper NMS IoU calculation
        box = xywh2xyxy(box)
        
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
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


class Masks:
    """
    A class for managing segmentation masks, compatible with Ultralytics Masks interface.
    Supports both torch.Tensor (from process_mask) and numpy array inputs.
    
    This class provides:
    - .data attribute: Raw mask data as torch.Tensor
    - .xy attribute: List of mask contours (polygon format) for visualization
    
    Attributes:
        data (torch.Tensor): Mask data tensor of shape (N, H, W) where N is number of masks
        xy (list): List of mask contours, each as numpy array of shape (num_points, 2)
        orig_shape (tuple): Original image shape (height, width)
    """
    
    def __init__(self, masks, orig_shape):
        """
        Initialize Masks with mask data.
        
        Handles two input formats:
        1. torch.Tensor from process_mask() - shape (N, H, W), values in [0, 1]
        2. List of numpy arrays - each array is a binary mask
        
        Args:
            masks (torch.Tensor | list): Mask data as tensor or list of numpy arrays
            orig_shape (tuple): Original image shape (height, width)
        """
        self.orig_shape = orig_shape
        
        # Store masks - handle both torch.Tensor (from process_mask) and list of numpy arrays
        if masks is not None:
            # If it's already a torch tensor (from process_mask), keep it directly
            if isinstance(masks, torch.Tensor):
                self.data = masks
            elif len(masks) > 0:
                # If it's a list of numpy arrays, stack and convert to tensor
                self.data = np.stack(masks) if isinstance(masks[0], np.ndarray) else masks
                if not isinstance(self.data, torch.Tensor):
                    self.data = torch.from_numpy(self.data)
            else:
                self.data = None
        else:
            self.data = None
        
        # Generate contours (.xy attribute) for each mask
        self.xy = []
        if self.data is not None:
            # Convert to numpy for contour extraction
            masks_np = self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data
            
            for mask in masks_np:
                # Convert boolean/float mask to uint8
                mask_uint8 = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask
                
                # Find contours
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Reshape to (N, 2) format
                    contour_points = largest_contour.squeeze()
                    if contour_points.ndim == 1:
                        contour_points = contour_points.reshape(-1, 2)
                    self.xy.append(contour_points.astype(np.float32))
                else:
                    # Empty contour if no contours found
                    self.xy.append(np.empty((0, 2), dtype=np.float32))
    
    def __len__(self):
        """Return number of masks."""
        return len(self.xy) if self.xy is not None else 0
    
    def __getitem__(self, idx):
        """Return mask at index."""
        return self.xy[idx] if self.xy is not None else None


class Results:
    """
    A simplified class for storing inference results.
    Ported from ultralytics.engine.results.Results
    
    This class provides a unified interface for detection results including:
    - Bounding boxes with confidence scores and class labels
    - Instance segmentation masks
    - Class name mapping
    - Original image and shape information
    
    Attributes:
        orig_img (np.ndarray): Original image as BGR numpy array
        orig_shape (tuple): Original image shape (height, width)
        boxes (Boxes): Boxes object containing all detection boxes in xyxy format
        masks (Masks): Masks object containing segmentation masks and contours
        names (dict): Dictionary mapping class IDs to class names (COCO dataset)
        path (str): Path to the source image file
    """
    
    def __init__(self, orig_img, path, names, boxes=None, masks=None):
        """
        Initialize Results object with detection outputs.
        
        Args:
            orig_img (np.ndarray): Original image as BGR numpy array
            path (str): Path to image file (can be None)
            names: Dictionary mapping class IDs to class names
            boxes: Tensor of shape (n, 6) with [x1, y1, x2, y2, conf, cls] or None
            masks: List of binary masks (numpy arrays) or Masks object or None
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        
        # Wrap masks in Masks object if it's a list
        if masks is not None and not isinstance(masks, Masks):
            self.masks = Masks(masks, self.orig_shape)
        else:
            self.masks = masks
        
        self.names = names
        self.path = path
    
    def __len__(self):
        """Return the number of detections."""
        return len(self.boxes) if self.boxes is not None else 0
    
    def __getitem__(self, idx):
        """Return a subset of results."""
        if self.boxes is None:
            return Results(self.orig_img, self.path, self.names, boxes=None, masks=None)
        
        # Extract mask data at index if Masks object exists
        new_masks = None
        if self.masks is not None:
            # Get the raw mask data (not Masks object)
            if hasattr(self.masks, 'data') and self.masks.data is not None:
                mask_data = self.masks.data[idx].cpu().numpy() if isinstance(self.masks.data, torch.Tensor) else self.masks.data[idx]
                new_masks = [mask_data]
            
        return Results(self.orig_img, self.path, self.names, boxes=self.boxes[idx].data, masks=new_masks)


def empty_like(x):
    """
    Create empty tensor or array with same shape as input.
    Ported from ultralytics.utils.ops.empty_like
    """
    if isinstance(x, torch.Tensor):
        return torch.empty_like(x, dtype=torch.float32)
    else:
        return np.empty_like(x, dtype=np.float32)

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from corner format to center format.
    Ported from ultralytics.utils.ops.xyxy2xywh
    
    This is the inverse operation of xywh2xyxy. Used primarily in the Boxes class
    for converting between different box representations.

    Args:
        x (torch.Tensor | np.ndarray): Boxes in xyxy format, shape (..., 4)
            - x[..., 0]: top-left x (x1)
            - x[..., 1]: top-left y (y1)
            - x[..., 2]: bottom-right x (x2)
            - x[..., 3]: bottom-right y (y2)
    
    Returns:
        (torch.Tensor | np.ndarray): Boxes in xywh format, shape (..., 4)
            - y[..., 0]: center x coordinate
            - y[..., 1]: center y coordinate
            - y[..., 2]: box width
            - y[..., 3]: box height
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center = (x1 + x2) / 2
    y[..., 1] = (y1 + y2) / 2  # y center = (y1 + y2) / 2
    y[..., 2] = x2 - x1  # width = x2 - x1
    y[..., 3] = y2 - y1  # height = y2 - y1
    return y


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

def postprocess_segmentation(outputs, orig_img):
    """
    Postprocess Model output using ported utilities from Ultralytics.
    Produces identical results to ultralytics_postprocess version.
    
    Processing pipeline:
    1. Apply NMS with xywh→xyxy conversion (crucial for correct detection filtering)
    2. Process masks using coefficients and prototypes (in letterbox coordinate space)
    3. Scale boxes from letterbox (640x640) to original image coordinates
    4. Scale masks from letterbox (640x640) to original image size with proper padding handling
    
    Args:
        outputs: List of model outputs [predictions, proto_masks]
            - predictions: [1, 116, 8400] where 116 = 4(bbox_xywh) + 80(classes) + 32(mask_coeffs)
            - proto_masks: [1, 32, 160, 160] (prototype masks for mask generation)
        orig_img: Original image (BGR) - used for shape extraction and Results object creation
    
    Returns:
        Results: Results object containing box detections and segmentation masks in original image space
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
    
    # Apply Non-Maximum Suppression using ported Ultralytics function
    # Input format: [1, 116, 8400] where 116 = 4(bbox_xywh) + 80(classes) + 32(mask_coeffs)
    # The NMS function will automatically convert bbox from xywh to xyxy format internally
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
    
    # Process masks with letterbox coordinates using ported function
    # process_mask(protos, masks_in, bboxes, shape, upsample)
    # - protos: [32, 160, 160] mask prototypes
    # - masks_in: [N, 32] mask coefficients after NMS
    # - bboxes: [N, 4] boxes in letterbox coordinates (NOT scaled yet)
    # - shape: (height, width) of input image (letterbox size)
    # - upsample: whether to upsample masks to letterbox image size
    masks = process_mask(protos[0], pred_masks_coeffs, pred_boxes[:, :4], preproc_shape, upsample=True)
    
    print(f"[Postprocess] Masks shape after process_mask: {masks.shape}")
    print(f"[Postprocess] Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
    
    # THEN scale boxes to original image size using ported function
    pred_boxes[:, :4] = scale_boxes(preproc_shape, pred_boxes[:, :4], orig_img.shape)
    
    # Scale masks from letterbox size to original image size using ported function
    # Using scale_masks which handles letterbox padding correctly
    masks = scale_masks(masks[None], orig_img.shape[:2], padding=True)[0]
    
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


def generate_mask_from_proto_simple(mask_coeff, proto_masks, box, img_width, img_height):
    """
    Generate final mask from mask coefficients and prototype masks.
    Simplified version matching the simple implementation.
    
    Args:
        mask_coeff: Mask coefficients (32,)
        proto_masks: Prototype masks (32, 160, 160)
        box: Bounding box in original image coordinates [x1, y1, x2, y2]
        img_width: Original image width
        img_height: Original image height
    
    Returns:
        Binary mask array matching original image size
    """
    c, mh, mw = proto_masks.shape  # (32, 160, 160)
    
    # Matrix multiplication: mask_coeff @ proto_masks
    mask = np.dot(mask_coeff, proto_masks.reshape(c, -1))  # (32,) @ (32, 25600) -> (25600,)
    mask = mask.reshape(mh, mw)  # Reshape to (160, 160)
    
    # Calculate ratios for downsampled bounding box
    width_ratio = mw / img_width   # 160 / original_width  
    height_ratio = mh / img_height  # 160 / original_height
    
    # Downsample bounding box coordinates to match proto mask size
    x1, y1, x2, y2 = box
    downsampled_bbox = [
        x1 * width_ratio,
        y1 * height_ratio,
        x2 * width_ratio,
        y2 * height_ratio
    ]
    
    # Crop mask to bounding box
    mask = crop_mask_np(mask, downsampled_bbox)
    
    # Scale mask to original image size
    mask = scale_mask_to_original(mask, (img_height, img_width), (mh, mw))
    
    # Apply threshold to get binary mask
    mask = (mask > 0.0).astype(np.uint8)
    
    return mask


def crop_mask_np(mask, bbox):
    """Crop mask to bounding box region (NumPy version of ultralytics crop_mask)."""
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within bounds
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    
    # Create coordinate grids
    r = np.arange(w)[None, :]  # Shape: (1, w)
    c = np.arange(h)[:, None]  # Shape: (h, 1)
    
    # Create crop mask
    crop_mask = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
    
    return mask * crop_mask

def scale_mask_to_original(mask, target_shape, mask_shape):
    """Scale mask to original image size (following ultralytics scale_masks logic)."""
    ih, iw = target_shape  # Original image dimensions
    mh, mw = mask_shape    # Proto mask dimensions
    
    # Calculate gain and padding (following letterbox logic)
    gain = min(mh / ih, mw / iw)
    pad_w = mw - iw * gain
    pad_h = mh - ih * gain
    pad_w /= 2
    pad_h /= 2
    
    top = int(round(pad_h - 0.1))
    left = int(round(pad_w - 0.1))
    bottom = mh - int(round(pad_h + 0.1))
    right = mw - int(round(pad_w + 0.1))
    
    # Crop the padded regions
    mask_cropped = mask[top:bottom, left:right]
    
    # Resize to original image size
    if mask_cropped.size > 0:
        mask_resized = cv2.resize(mask_cropped, (iw, ih), interpolation=cv2.INTER_LINEAR)
    else:
        mask_resized = np.zeros((ih, iw), dtype=np.float32)
    
    return mask_resized

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