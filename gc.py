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
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco.imgs[image_id]
        image_path = self.images_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox'] # Convert to x1, y1, x2, y2 format
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[ann['category_id']])
            areas.append(ann['area'])
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms:
            # Apply albumentations transforms
            transformed = self.transforms(image=image, bboxes=boxes.numpy(), 
                                       class_labels=labels.numpy())
            image = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.tensor(transformed['class_labels'], dtype=torch.int64)
        
        return image, target

# Data Augmentation
def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

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
        
    def forward(self, images):
        # Extract features using ViT backbone
        features = self.backbone(images)  # (batch_size, n_patches+1, embed_dim)
        
        # Remove CLS token for detection
        features = features[:, 1:, :]  # (batch_size, n_patches, embed_dim)
        
        # Object detection
        predictions = self.detection_head(features)
        
        return predictions

# Hungarian Matching - Loss function
from scipy.optimize import linear_sum_assignment
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    """
    # degenerate boxes gives inf / nan results
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area

Object Detection Loss
class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses=['labels', 'boxes', 'cardinality']):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['class_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bbox_coords'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
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
    DROPOUT = 0.1
    NUM_QUERIES = 100
    API = "wandb_api_key"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    WARMUP_EPOCHS = 10
    
    # Loss weights
    WEIGHT_DICT = {
       'loss_ce': 1,
       'loss_bbox': 5,
       'loss_giou': 2,
    }

    
    # Data paths
    TRAIN_IMAGES_DIR = "dataset/train" # Path to images for training
    TRAIN_ANNOTATIONS = "dataset/annotations/train.json" # Path to annotations for training
    VAL_IMAGES_DIR = "dataset/val" # Path to Images for validation
    VAL_ANNOTATIONS = "dataset/annotations/instances_val.json" # Path for annotations for validation
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Loop
def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    criterion.train()
    
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(enabled=(scaler is not None)):
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
                'train/learning_rate': optimizer.param_groups['lr'],
                **{f'train/{k}': v.item() for k, v in loss_dict.items()}
            })
    
    return running_loss / len(data_loader)

## Training Script
def main():
    # Initialize wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="vit-object-detection", config=Config.__dict__)

    
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
        num_workers=4,
        collate_fn=collate_fn,
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
    ).to(Config.DEVICE)
    
    # Create loss function
    matcher = HungarianMatcher()
    criterion = ObjectDetectionLoss(
        num_classes=train_dataset.num_classes,
        matcher=matcher,
        weight_dict=Config.WEIGHT_DICT
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
                'config': Config.__dict__
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()
