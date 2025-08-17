# Core dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.ops import box_iou
import numpy as np
import cv2
from PIL import Image
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import wandb
from tqdm import tqdm
from itertools import combinations

# Dataset Class
class COCOObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None, max_objects=100):
        self.images_dir = Path(images_dir)
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.max_objects = max_objects
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_id_to_label = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)
    
    def __len__(self):
        return len(self.image_ids)
    
#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
        
#         # Load image
#         image_info = self.coco.imgs[image_id]
#         image_path = self.images_dir / image_info['file_name']
#         image = cv2.imread(str(image_path))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w = image.shape[:2]
        
#         # Load annotations
#         ann_ids = self.coco.getAnnIds(imgIds=image_id)
#         annotations = self.coco.loadAnns(ann_ids)
        
#         # Extract bounding boxes and labels
#         boxes = []
#         labels = []
#         areas = []
        
#         for ann in annotations:
#             #x, y, w, h = ann['bbox'] # Convert to x1, y1, x2, y2 format
#             x, y, bw, bh = ann['bbox'] # Using bw and bh to avoid conflict with w and h of image
#             #boxes.append([x, y, x + w, y + h])
#             boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h]) # Normalized coordinates
#             labels.append(self.category_id_to_label[ann['category_id']])
#             areas.append(ann['area'])
        
#         # Convert to tensors
#         if len(boxes) == 0:
#             boxes = torch.zeros((0, 4), dtype=torch.float32)
#             labels = torch.zeros((0,), dtype=torch.int64)
#             areas = torch.zeros((0,), dtype=torch.float32)
#         else:
#             boxes = torch.tensor(boxes, dtype=torch.float32)
#             labels = torch.tensor(labels, dtype=torch.int64)
#             areas = torch.tensor(areas, dtype=torch.float32)
        
#         target = {
#             'boxes': boxes,
#             'labels': labels,
#             'image_id': torch.tensor([image_id]),
#             'area': areas,
#             'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
#         }
        
#         if self.transforms:
#             # Apply albumentations transforms
#             transformed = self.transforms(image=image, bboxes=boxes.numpy(), 
#                                        class_labels=labels.numpy())
#             image = transformed['image']
            
#             if len(transformed['bboxes']) > 0:
#                 target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
#                 target['labels'] = torch.tensor(transformed['class_labels'], dtype=torch.int64)
        
#         return image, target

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco.imgs[image_id]
        image_path = self.images_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2] # Get image height and width
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Extract and validate bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        
        for ann in annotations:
            x, y, bw, bh = ann['bbox'] # raw bbox values
            
            # --- Initial Validation: Skip boxes with zero or negative width/height ---
            if bw <= 0 or bh <= 0:
                continue
            
            # Convert to x1, y1, x2, y2 format and normalize
            # Ensure coordinates are within [0, 1] range after normalization
            x1 = np.clip(x / w, 0.0, 1.0)
            y1 = np.clip(y / h, 0.0, 1.0)
            x2 = np.clip((x + bw) / w, 0.0, 1.0)
            y2 = np.clip((y + bh) / h, 0.0, 1.0)
            
            # Ensure x1 < x2 and y1 < y2 even after potential clipping/floating point issues
            # Swap if x1 > x2 or y1 > y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            # Discard very small or degenerate boxes
            # A small threshold (e.g., 1e-6 or 0.001) for width/height after normalization
            # to filter out boxes that become tiny or invalid due to floating point arithmetic
            if (x2 - x1) < 1e-6 or (y2 - y1) < 1e-6: 
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[ann['category_id']])
            areas.append(ann['area']) # COCO area is usually pixel area, not normalized
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            # Ensure iscrowd is also an empty tensor for empty boxes case
            iscrowd = torch.zeros((0,), dtype=torch.int64) 
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64) # All boxes are not crowd here
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        if self.transforms:
            # Apply albumentations transforms
            # Pass numpy arrays of boxes and labels to albumentations
            transformed = self.transforms(image=image, bboxes=target['boxes'].numpy(), 
                                       class_labels=target['labels'].numpy())
            image = transformed['image']
            
            # --- Post-Transformation Validation and Update ---
            if len(transformed['bboxes']) > 0:
                # Albumentations returns normalized coordinates, so directly convert back to tensor
                target_boxes_after_transform = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                target_labels_after_transform = torch.tensor(transformed['class_labels'], dtype=torch.int64)

                # Filter out any degenerate boxes that might have resulted from transformations
                valid_mask_after_transform = (target_boxes_after_transform[:, 2] > target_boxes_after_transform[:, 0]) & \
                                             (target_boxes_after_transform[:, 3] > target_boxes_after_transform[:, 1])

                target['boxes'] = target_boxes_after_transform[valid_mask_after_transform]
                target['labels'] = target_labels_after_transform[valid_mask_after_transform]
                
                # Update area and iscrowd based on valid boxes after transform
                if len(target['boxes']) > 0:
                    # Recalculate area for transformed boxes if needed, or use dummy values
                    # Since boxes are normalized [0,1], area is (x2-x1)*(y2-y1)
                    target['area'] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                                     (target['boxes'][:, 3] - target['boxes'][:, 1])
                    target['iscrowd'] = torch.zeros((len(target['boxes']),), dtype=torch.int64)
                else:
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
                    target['area'] = torch.zeros((0,), dtype=torch.float32)
                    target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
            else:
                # If no bounding boxes remain after transformation, set target fields to empty tensors
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
        
        return image, target

# Data Augmentation
def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        #A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-10, 10), p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    #], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    #], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels'], min_area=1.0, min_visibility=0.0))
# def get_train_transforms(image_size=224):
#     return A.Compose([
#         A.LongestMaxSize(max_size=image_size),
#         A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),  # preserve aspect ratio, pad
#         A.OneOf([
#             A.HorizontalFlip(p=1.0),
#             A.VerticalFlip(p=1.0),
#             A.RandomRotate90(p=1.0),
#         ], p=0.7),
#         A.ShiftScaleRotate(
#             shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=0, p=0.7
#         ),
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
#         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
#         A.OneOf([
#             A.GaussNoise(p=1.0),
#             A.ISONoise(p=1.0),
#         ], p=0.3),
#         A.CoarseDropout(
#             max_holes=4,
#             max_height=32,
#             max_width=32,
#             fill_value=0,
#             p=0.3
#         ),
#         A.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ], bbox_params=A.BboxParams(format='albumentations',
#                                  label_fields=['class_labels'],
#                                  min_area=1.0,
#                                  min_visibility=0.0))


def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    #], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    #], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels'], min_area=1.0, min_visibility=0.0))

# Collate
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

# Patch Embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

# Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.projection(context)
        output = self.projection_dropout(output)
        
        return output

# MLP Block
class MLPBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

# Encoder
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(self.layer_norm1(x))
        x = x + attn_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.layer_norm2(x))
        x = x + mlp_output
        
        return x

# ViT
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_layers=12, num_heads=12, mlp_dim=3072, dropout=0.1, num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.patch_embedding.n_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.position_embedding
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        
        return x  # Return all tokens for detection

# Object Detection Head
class ObjectDetectionHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=80, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Object queries
        self.object_queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Prediction heads
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(embed_dim, 4)  # x, y, w, h
        
    def forward(self, encoder_features):
        batch_size = encoder_features.shape[0]
        
        # Expand object queries for batch
        object_queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Decode
        decoder_output = self.decoder(object_queries, encoder_features)
        
        # Predictions
        class_logits = self.class_head(decoder_output)  # (batch_size, num_queries, num_classes+1)
        bbox_coords = self.bbox_head(decoder_output)    # (batch_size, num_queries, 4)
        bbox_coords = torch.sigmoid(bbox_coords)        # Normalize to [0, 1]
        
        return {
            'class_logits': class_logits,
            'bbox_coords': bbox_coords
        }

# ViT Object Detection
class ViTObjectDetector(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_layers=12, num_heads=12, mlp_dim=3072, dropout=0.1, 
                 num_classes=80, num_queries=100):
        super().__init__()
        
        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        self.detection_head = ObjectDetectionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_queries=num_queries
        )
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Add projection layer for triplet loss features
        self.triplet_projection = nn.Linear(embed_dim, 256)

    # def forward(self, images):
    #     # Extract features using ViT backbone
    #     features = self.backbone(images)  # (batch_size, n_patches+1, embed_dim)
        
    #     # Remove CLS token for detection
    #     features = features[:, 1:, :]  # (batch_size, n_patches, embed_dim)
        
    #     # Object detection
    #     predictions = self.detection_head(features)
        
    #     return predictions

    def forward(self, images, return_features=False):
        # Extract features using ViT backbone
        features = self.backbone(images)  # (batch_size, n_patches+1, embed_dim)
        
        # Extract triplet features from CLS token
        triplet_features = None
        if return_features or self.training:
            cls_features = features[:, 0, :]  # Extract CLS token
            triplet_features = self.triplet_projection(cls_features)
            triplet_features = F.normalize(triplet_features, p=2, dim=1)
        
        # Remove CLS token for detection
        features = features[:, 1:, :]  # (batch_size, n_patches, embed_dim)
        
        # Object detection
        predictions = self.detection_head(features)
        
        if return_features or self.training:
            return predictions, triplet_features
        else:
            return predictions

# Hungarian Matching - Loss function
from scipy.optimize import linear_sum_assignment

# class HungarianMatcher(nn.Module):
#     def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_bbox = cost_bbox
#         self.cost_giou = cost_giou
        
#     def forward(self, outputs, targets):
#         batch_size, num_queries = outputs['class_logits'].shape[:2]
        
#         # Flatten to compute cost matrix
#         out_prob = outputs['class_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
#         out_bbox = outputs['bbox_coords'].flatten(0, 1)  # [batch_size * num_queries, 4]
        
#         # Also concat the target labels and boxes
#         tgt_ids = torch.cat([v["labels"] for v in targets])
#         tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
#         # Compute the classification cost
#         cost_class = -out_prob[:, tgt_ids]
        
#         # Compute the L1 cost between boxes
#         cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
#         # Compute the GIoU cost between boxes
#         cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        
#         # Final cost matrix
#         C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
#         C = C.view(batch_size, num_queries, -1).cpu()
        
#         sizes = [len(v["boxes"]) for v in targets]
#         indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @staticmethod
    def filter_valid_boxes(boxes):
        """Filter out invalid bounding boxes where x2 < x1 or y2 < y1"""
        if len(boxes) == 0:
            return boxes, torch.tensor([], dtype=torch.bool)
        mask = (boxes[:, 2:] >= boxes[:, :2]).all(dim=1)
        return boxes[mask], mask
        
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs['class_logits'].shape[:2]
        
        # Handle empty targets case
        if all(len(v["boxes"]) == 0 for v in targets):
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(batch_size)]
        
        # Flatten to compute cost matrix
        out_prob = outputs['class_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['bbox_coords'].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Concat the target labels and boxes (only from non-empty targets)
        tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"]) > 0])
        tgt_bbox = torch.cat([v["boxes"] for v in targets if len(v["boxes"]) > 0])
        
        # Handle case where no valid targets exist
        if len(tgt_ids) == 0 or len(tgt_bbox) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(batch_size)]
        
        # Filter invalid boxes before computing costs
        out_bbox_filtered, valid_out_mask = self.filter_valid_boxes(out_bbox)
        tgt_bbox_filtered, valid_tgt_mask = self.filter_valid_boxes(tgt_bbox)
        
        # Handle case where filtering removes all boxes
        if len(out_bbox_filtered) == 0 or len(tgt_bbox_filtered) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(batch_size)]
        
        # Compute the classification cost (use original indices for class cost)
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the GIoU cost between filtered boxes
        try:
            cost_giou = -generalized_box_iou(out_bbox_filtered, tgt_bbox_filtered)
            
            # If filtering changed the sizes, we need to adjust the cost matrix
            if len(out_bbox_filtered) != len(out_bbox) or len(tgt_bbox_filtered) != len(tgt_bbox):
                # Create a full cost matrix with high costs for filtered boxes
                full_cost_giou = torch.ones(len(out_bbox), len(tgt_bbox), device=cost_giou.device) * 1000.0  # High cost
                
                # Map the filtered costs back to the full matrix
                out_indices = torch.nonzero(valid_out_mask).squeeze(1)
                tgt_indices = torch.nonzero(valid_tgt_mask).squeeze(1)
                
                for i, out_idx in enumerate(out_indices):
                    for j, tgt_idx in enumerate(tgt_indices):
                        full_cost_giou[out_idx, tgt_idx] = cost_giou[i, j]
                
                cost_giou = full_cost_giou
        except Exception as e:
            print(f"Error in GIoU calculation: {e}")
            # Fallback: use zero cost for GIoU
            cost_giou = torch.zeros_like(cost_bbox)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        
        start_idx = 0
        for i, size in enumerate(sizes):
            if size > 0:
                cost_matrix = C[i, :, start_idx:start_idx + size].detach()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))
                start_idx += size
            else:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
        
        return indices


# def generalized_box_iou(boxes1, boxes2):
#     # degenerate boxes gives inf / nan results
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
#     iou, union = box_iou(boxes1, boxes2)
    
#     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
#     wh = (rb - lt).clamp(min=0)
#     area = wh[:, :, 0] * wh[:, :, 1]
    
#     return iou - (area - union) / area
def generalized_box_iou(boxes1, boxes2):
    # degenerate boxes gives inf / nan results
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Calculate union
    union = area1[:, None] + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    # For GIoU, we need the smallest enclosing box
    lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]
    
    # GIoU = IoU - (area_enclosing - union) / area_enclosing
    return iou - (area_enclosing - union) / area_enclosing

# Triplet Loss
class TripletMiner(nn.Module):
    """Online triplet mining for object detection features"""
    
    def __init__(self, margin=0.2, mining_strategy='hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        
    def get_triplets(self, features, targets):
        batch_size = features.shape
        device = features.device
        
        # Create image-level labels using dominant class
        batch_labels = []
        for i, target in enumerate(targets):
            if len(target['labels']) > 0:
                labels = target['labels'].cpu().numpy()
                unique_labels, counts = np.unique(labels, return_counts=True)
                dominant_label = unique_labels[np.argmax(counts)]
                batch_labels.append(dominant_label)
            else:
                batch_labels.append(-1)  # No objects
        
        batch_labels = torch.tensor(batch_labels, device=device)
        valid_indices = torch.where(batch_labels >= 0)
        
        if len(valid_indices) < 3:
            return torch.empty(0, 3, dtype=torch.long, device=device), torch.empty(0, dtype=torch.bool, device=device)
        
        valid_labels = batch_labels[valid_indices]
        unique_labels = torch.unique(valid_labels)
        
        if len(unique_labels) < 2:
            return torch.empty(0, 3, dtype=torch.long, device=device), torch.empty(0, dtype=torch.bool, device=device)
        
        triplets = []
        
        for anchor_idx in valid_indices:
            anchor_label = batch_labels[anchor_idx]
            positive_indices = valid_indices[valid_labels == anchor_label]
            positive_indices = positive_indices[positive_indices != anchor_idx]
            negative_indices = valid_indices[valid_labels != anchor_label]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                if self.mining_strategy == 'hard':
                    anchor_feat = features[anchor_idx:anchor_idx+1]
                    pos_distances = torch.cdist(anchor_feat, features[positive_indices])
                    hardest_pos_idx = positive_indices[torch.argmax(pos_distances)]
                    neg_distances = torch.cdist(anchor_feat, features[negative_indices])
                    hardest_neg_idx = negative_indices[torch.argmin(neg_distances)]
                    triplets.append([anchor_idx.item(), hardest_pos_idx.item(), hardest_neg_idx.item()])
                else:  # random
                    pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
                    neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
                    triplets.append([anchor_idx.item(), pos_idx.item(), neg_idx.item()])
        
        if len(triplets) == 0:
            return torch.empty(0, 3, dtype=torch.long, device=device), torch.empty(0, dtype=torch.bool, device=device)
        
        triplets = torch.tensor(triplets, device=device)
        valid_mask = torch.ones(len(triplets), dtype=torch.bool, device=device)
        return triplets, valid_mask

class TripletLoss(nn.Module):
    
    def __init__(self, margin=0.2, mining_strategy='hard'):
        super().__init__()
        self.margin = margin
        self.miner = TripletMiner(margin=margin, mining_strategy=mining_strategy)
        
    def forward(self, features, targets):
        if features.shape[0] < 3:
            return torch.tensor(0.0, device=features.device, requires_grad=True), 0
        
        triplets, valid_mask = self.miner.get_triplets(features, targets)
        
        if len(triplets) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True), 0
        
        anchors = features[triplets[:, 0]]
        positives = features[triplets[:, 1]]
        negatives = features[triplets[:, 2]]
        
        pos_distances = torch.sum((anchors - positives) ** 2, dim=1)
        neg_distances = torch.sum((anchors - negatives) ** 2, dim=1)
        
        losses = F.relu(pos_distances - neg_distances + self.margin)
        losses = losses[valid_mask]
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True), 0
        
        return torch.mean(losses), len(losses)

# Object Detection Loss
# class ObjectDetectionLoss(nn.Module):
#     def __init__(self, num_classes, matcher, weight_dict, losses=['labels', 'boxes', 'cardinality']):
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
        
#     def loss_labels(self, outputs, targets, indices, num_boxes):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         src_logits = outputs['class_logits']
        
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                   dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
        
#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
#         losses = {'loss_ce': loss_ce}
        
#         return losses
    
#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss"""
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['bbox_coords'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
#         loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
        
#         return losses
    
#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx
    
#     def forward(self, outputs, targets):
#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs, targets)
        
#         # Compute the average number of target boxes across all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
#         return losses
    
#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'cardinality': self.loss_cardinality,
#             'boxes': self.loss_boxes,
#         }
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

# Object Detection Loss
class ObjectDetectionLoss(nn.Module):
    # def __init__(self, num_classes, matcher, weight_dict, losses=['labels', 'boxes']):  # Removed 'cardinality' for now
    #     super().__init__()
    #     self.num_classes = num_classes
    #     self.matcher = matcher
    #     self.weight_dict = weight_dict
    #     self.losses = losses
        
    #     # Define empty_weight for classification loss (background class gets lower weight)
    #     empty_weight = torch.ones(self.num_classes + 1)
    #     empty_weight[-1] = 0.1  # Lower weight for background class
    #     self.register_buffer('empty_weight', empty_weight)

    def __init__(self, num_classes, matcher, weight_dict, losses=['labels', 'boxes'], 
                use_triplet_loss=True, triplet_margin=0.2, triplet_mining='hard'):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        
        # Add triplet loss
        self.use_triplet_loss = use_triplet_loss
        if use_triplet_loss:
            self.triplet_loss_fn = TripletLoss(margin=triplet_margin, mining_strategy=triplet_mining)
        
        # Define empty_weight for classification loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)"""
        src_logits = outputs['class_logits']
        
        idx = self._get_src_permutation_idx(indices)
        
        # Handle empty indices case
        if len(idx[0]) == 0:
            target_classes_o = torch.tensor([], dtype=torch.long, device=src_logits.device)
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        if len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes"""
        # If num_boxes is zero, return zero loss
        if num_boxes == 0:
            return {
                'loss_bbox': torch.tensor(0.0, device=outputs['bbox_coords'].device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=outputs['bbox_coords'].device, requires_grad=True)
            }
            
        idx = self._get_src_permutation_idx(indices)
        
        # Handle empty indices case
        if len(idx) == 0:
            return {
                'loss_bbox': torch.tensor(0.0, device=outputs['bbox_coords'].device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=outputs['bbox_coords'].device, requires_grad=True)
            }
        
        src_boxes = outputs['bbox_coords'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # Calculate GIoU loss with additional safety check
        if len(src_boxes) > 0 and len(target_boxes) > 0:
            try:
                loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
                losses['loss_giou'] = loss_giou.sum() / num_boxes
            except AssertionError:
                # Fallback if boxes are still invalid
                losses['loss_giou'] = torch.tensor(0.0, device=src_boxes.device, requires_grad=True)
        else:
            losses['loss_giou'] = torch.tensor(0.0, device=outputs['bbox_coords'].device, requires_grad=True)
        
        return losses
    
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error (optional loss)"""
        pred_logits = outputs['class_logits']
        device = pred_logits.device
        
        # Count true number of target boxes across the batch
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], dtype=torch.float, device=device)
        
        # Count the number of predictions that are NOT "no-object" (background class)
        card_pred = (pred_logits.argmax(-1) != self.num_classes).sum(1).float()
        
        card_err = F.l1_loss(card_pred, tgt_lengths)
        losses = {'loss_cardinality': card_err}
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    # def forward(self, outputs, targets):
    #     # Retrieve the matching between the outputs of the last layer and the targets
    #     indices = self.matcher(outputs, targets)
        
    #     # Compute the average number of target boxes across all nodes, for normalization purposes
    #     num_boxes = sum(len(t["labels"]) for t in targets)
    #     num_boxes = torch.as_tensor([max(num_boxes, 1)], dtype=torch.float, device=next(iter(outputs.values())).device)
        
    #     # Compute all the requested losses
    #     losses = {}
    #     for loss in self.losses:
    #         losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
    #     return losses
    def forward(self, outputs, targets, triplet_features=None):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([max(num_boxes, 1)], dtype=torch.float, 
                                device=next(iter(outputs.values())).device)
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # Add triplet loss
        if self.use_triplet_loss and triplet_features is not None:
            triplet_loss, num_triplets = self.triplet_loss_fn(triplet_features, targets)
            losses['loss_triplet'] = triplet_loss
            losses['num_triplets'] = torch.tensor(float(num_triplets), device=triplet_loss.device)
        
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


# Training
class Config:
    # Model parameters
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    EMBED_DIM = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12
    MLP_DIM = 3072
    DROPOUT = 0.5
    NUM_QUERIES = 100
    API = "api_key"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 5e-5
    EPOCHS = 50
    WARMUP_EPOCHS = 10

    # Triplet loss parameters
    USE_TRIPLET_LOSS = True
    TRIPLET_MARGIN = 0.2
    TRIPLET_MINING_STRATEGY = 'hard'
    
    # Loss weights
    WEIGHT_DICT = {
       'loss_ce': 1,
       'loss_bbox': 5,
       'loss_giou': 2,
       'loss_triplet': 0.5,
    }

    
    # Data paths
    TRAIN_IMAGES_DIR = "E:/Projects/gc/dataset/train" # Path to images for training
    TRAIN_ANNOTATIONS = "E:/Projects/gc/dataset/train/_annotations.coco.json" # Path to annotations for training
    VAL_IMAGES_DIR = "E:/Projects/gc/dataset/valid" # Path to Images for validation
    VAL_ANNOTATIONS = "E:/Projects/gc/dataset/valid/_annotations.coco.json" # Path for annotations for validation

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WandbConfig:
    """
    Extract only the needed fields from Config for wandb logging.
    This helps avoid passing complex objects like DEVICE or API keys.
    """
    @staticmethod
    def from_config(config_class) -> dict:
        keys_to_include = [
            'IMAGE_SIZE',
            'PATCH_SIZE',
            'EMBED_DIM',
            'NUM_LAYERS',
            'NUM_HEADS',
            'MLP_DIM',
            'DROPOUT',
            'NUM_QUERIES',
            'BATCH_SIZE',
            'LEARNING_RATE',
            'WEIGHT_DECAY',
            'EPOCHS',
            'WARMUP_EPOCHS',
            'WEIGHT_DICT',
        ]
        return {k: getattr(config_class, k) for k in keys_to_include}

# Training Loop
def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    criterion.train()
    
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        # #with torch.amp.autocast(enabled=scaler is not None):
        # with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=(scaler is not None)):
        #     outputs = model(images)
        #     loss_dict = criterion(outputs, targets)
        with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=(scaler is not None)):
            if Config.USE_TRIPLET_LOSS:
                outputs, triplet_features = model(images, return_features=True)
                loss_dict = criterion(outputs, targets, triplet_features=triplet_features)
            else:
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
            
            # Compute total loss
            losses = sum(loss_dict[k] * Config.WEIGHT_DICT[k] for k in loss_dict.keys() if k in Config.WEIGHT_DICT)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        
        running_loss += losses.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb
        if batch_idx % 100 == 0:
            wandb.log({
                'train/batch_loss': losses.item(),
                #'train/learning_rate': optimizer.param_groups['lr'],
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                **{f'train/{k}': v.item() for k, v in loss_dict.items()}
            })
    
    return running_loss / len(data_loader)

# Validation Function
@torch.no_grad()
def validate(model, data_loader, criterion, device):
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc='Validation')
    
    #for images, targets in progress_bar:
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        
        losses = sum(loss_dict[k] * Config.WEIGHT_DICT[k] for k in loss_dict.keys() if k in Config.WEIGHT_DICT)
        running_loss += losses.item()
        
        #progress_bar.set_postfix({'val_loss': running_loss / (len(progress_bar.n) + 1)})
        progress_bar.set_postfix({'val_loss': running_loss / (batch_idx + 1)})
    
    return running_loss / len(data_loader)

## Training Script
def main():
    # Initialize wandb
    os.environ["WANDB_MODE"] = "offline"
    #wandb.init(project="vit-object-detection", config=Config.__dict__)
    wandb.init(project="vit-object-detection", config=WandbConfig.from_config(Config))

    
    # Create datasets
    train_dataset = COCOObjectDetectionDataset(
        Config.TRAIN_IMAGES_DIR,
        Config.TRAIN_ANNOTATIONS,
        transforms=get_train_transforms(Config.IMAGE_SIZE)
    )
    
    val_dataset = COCOObjectDetectionDataset(
        Config.VAL_IMAGES_DIR,
        Config.VAL_ANNOTATIONS,
        transforms=get_val_transforms(Config.IMAGE_SIZE)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = ViTObjectDetector(
        image_size=Config.IMAGE_SIZE,
        patch_size=Config.PATCH_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        mlp_dim=Config.MLP_DIM,
        dropout=Config.DROPOUT,
        num_classes=train_dataset.num_classes,
        num_queries=Config.NUM_QUERIES
    #).to(Config.DEVICE)
    ).to(Config.DEVICE).to(memory_format=torch.channels_last)
    
    #model = torch.compile(model, mode="max-autotune")

    # Create loss function
    matcher = HungarianMatcher()
    criterion = ObjectDetectionLoss(
        num_classes=train_dataset.num_classes,
        matcher=matcher,
        weight_dict=Config.WEIGHT_DICT,
        use_triplet_loss=Config.USE_TRIPLET_LOSS,
        triplet_margin=Config.TRIPLET_MARGIN,
        triplet_mining=Config.TRIPLET_MINING_STRATEGY
    ).to(Config.DEVICE)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            Config.DEVICE, epoch, scaler
        )
        
        # Validation
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                #'config': Config.__dict__
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
            }, 'best_vit_detector.pth')
            
            print(f'New best model saved with val_loss: {val_loss:.4f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                #'config': Config.__dict__
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()
