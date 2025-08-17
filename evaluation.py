# ViT Object Detection - Evaluation Script
# Fixed version that properly loads the checkpoint from training script

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# COPY ALL MODEL CLASSES FROM TRAINING SCRIPT (EXACT SAME ARCHITECTURE)
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

# Transformer Block
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

# Vision Transformer
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
        bbox_coords = self.bbox_head(decoder_output)  # (batch_size, num_queries, 4)
        bbox_coords = torch.sigmoid(bbox_coords)  # Normalize to [0, 1]
        
        return {
            'class_logits': class_logits,
            'bbox_coords': bbox_coords
        }

# ViT Object Detector
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

# DATASET CLASS FOR EVALUATION
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
        self.label_to_category_id = {idx: cat['id'] for idx, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        
        # Create category name mapping
        self.category_names = {idx: cat['name'] for idx, cat in enumerate(self.categories)}

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
        
        # Extract and validate bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        
        for ann in annotations:
            x, y, bw, bh = ann['bbox']
            
            # Skip invalid boxes
            if bw <= 0 or bh <= 0:
                continue
                
            # Convert to x1, y1, x2, y2 format and normalize
            x1 = np.clip(x / w, 0.0, 1.0)
            y1 = np.clip(y / h, 0.0, 1.0)
            x2 = np.clip((x + bw) / w, 0.0, 1.0)
            y2 = np.clip((y + bh) / h, 0.0, 1.0)
            
            # Ensure valid boxes
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            if (x2 - x1) < 1e-6 or (y2 - y1) < 1e-6:
                continue
                
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[ann['category_id']])
            areas.append(ann['area'])
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=target['boxes'].numpy(),
                                        class_labels=target['labels'].numpy())
            image = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                target_boxes_after_transform = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                target_labels_after_transform = torch.tensor(transformed['class_labels'], dtype=torch.int64)
                
                # Filter out degenerate boxes
                valid_mask = (target_boxes_after_transform[:, 2] > target_boxes_after_transform[:, 0]) & \
                           (target_boxes_after_transform[:, 3] > target_boxes_after_transform[:, 1])
                
                target['boxes'] = target_boxes_after_transform[valid_mask]
                target['labels'] = target_labels_after_transform[valid_mask]
                
                if len(target['boxes']) > 0:
                    target['area'] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                                   (target['boxes'][:, 3] - target['boxes'][:, 1])
                    target['iscrowd'] = torch.zeros((len(target['boxes']),), dtype=torch.int64)
                else:
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
                    target['area'] = torch.zeros((0,), dtype=torch.float32)
                    target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
        
        return image, target

# DATA TRANSFORMS
def get_eval_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels'], 
                               min_area=1.0, min_visibility=0.0))

# Collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

# EVALUATION FUNCTIONS
def load_model_checkpoint(model, checkpoint_path, device):
    """Safely load model checkpoint"""
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Model loaded successfully (direct state dict)")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def post_process_predictions(outputs, confidence_threshold=0.5, nms_threshold=0.5):
    """Post-process model predictions"""
    batch_size = outputs['class_logits'].shape[0]
    batch_predictions = []
    
    for i in range(batch_size):
        class_logits = outputs['class_logits'][i]  # (num_queries, num_classes+1)
        bbox_coords = outputs['bbox_coords'][i]    # (num_queries, 4)
        
        # Get class probabilities and predicted classes
        class_probs = F.softmax(class_logits, dim=-1)
        max_probs, predicted_classes = torch.max(class_probs[:, :-1], dim=-1)  # Exclude background class
        
        # Filter by confidence threshold
        confident_mask = max_probs > confidence_threshold
        
        if confident_mask.sum() > 0:
            filtered_boxes = bbox_coords[confident_mask]
            filtered_classes = predicted_classes[confident_mask]
            filtered_scores = max_probs[confident_mask]
            
            batch_predictions.append({
                'boxes': filtered_boxes,
                'labels': filtered_classes,
                'scores': filtered_scores
            })
        else:
            batch_predictions.append({
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros((0,), dtype=torch.long),
                'scores': torch.zeros((0,))
            })
    
    return batch_predictions

def visualize_predictions(image, predictions, targets, category_names, save_path=None, show_gt=True):
    # Visualize predictions and ground truth
    fig, axes = plt.subplots(1, 2 if show_gt else 1, figsize=(15, 8))
    if not show_gt:
        axes = [axes]

    # Move mean/std to image device
    device = image.device if isinstance(image, torch.Tensor) else torch.device('cpu')
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    # Denormalize image for visualization
    if isinstance(image, torch.Tensor):
        image_vis = image * std + mean
        image_vis = torch.clamp(image_vis, 0, 1)
        image_vis = image_vis.permute(1, 2, 0).cpu().numpy()
    else:
        image_vis = image
    
    # Plot predictions
    axes[0].imshow(image_vis)
    axes[0].set_title(f'Predictions ({len(predictions["boxes"])} objects)')
    axes[0].axis('off')
    
    # Draw prediction boxes
    h, w = image_vis.shape[:2]
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        x1, y1, x2, y2 = box.cpu().numpy()
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        class_name = category_names.get(label.item(), f'Class {label.item()}')
        axes[0].text(x1, y1-5, f'{class_name}: {score:.2f}', 
                    color='red', fontweight='bold', fontsize=8)
    
    if show_gt and targets is not None:
        # Plot ground truth
        axes[1].imshow(image_vis)
        axes[1].set_title(f'Ground Truth ({len(targets["boxes"])} objects)')
        axes[1].axis('off')
        
        # Draw ground truth boxes
        for box, label in zip(targets['boxes'], targets['labels']):
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='green', facecolor='none')
            axes[1].add_patch(rect)
            
            class_name = category_names.get(label.item(), f'Class {label.item()}')
            axes[1].text(x1, y1-5, class_name, color='green', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

@torch.no_grad()
def evaluate_model(model, data_loader, device, confidence_threshold=0.5, num_visualize=5):
    """Evaluate model on dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Running evaluation...")
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Post-process predictions
        batch_predictions = post_process_predictions(outputs, confidence_threshold)
        
        # Store predictions and targets
        all_predictions.extend(batch_predictions)
        all_targets.extend(targets)
        
        # Visualize first few batches
        if batch_idx < num_visualize:
            for i in range(min(len(images), 2)):  # Show first 2 images per batch
                visualize_predictions(
                    images[i], 
                    batch_predictions[i], 
                    targets[i],
                    data_loader.dataset.category_names,
                    save_path=f'eval_batch_{batch_idx}_img_{i}.png'
                )
    
    return all_predictions, all_targets

# MAIN EVALUATION SCRIPT
def main():
    # Configuration
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    EMBED_DIM = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12
    MLP_DIM = 3072
    DROPOUT = 0.1
    NUM_QUERIES = 100
    BATCH_SIZE = 8
    CONFIDENCE_THRESHOLD = 0.5
    
    # Paths - MODIFY THESE TO MATCH YOUR SETUP
    MODEL_CHECKPOINT = "E:/Projects/gc/best_vit_detector.pth"  # Path to your trained model
    TEST_IMAGES_DIR = "E:/Projects/gc/tinydataset/test"  # Test images
    TEST_ANNOTATIONS = "E:/Projects/gc/tinydataset/test/_annotations.coco.json"  # Test annotations
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = COCOObjectDetectionDataset(
        TEST_IMAGES_DIR,
        TEST_ANNOTATIONS,
        transforms=get_eval_transforms(IMAGE_SIZE)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Number of classes: {test_dataset.num_classes}")
    print(f"Categories: {list(test_dataset.category_names.values())}")
    
    # Create model
    print("Creating model...")
    model = ViTObjectDetector(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        num_classes=test_dataset.num_classes,
        num_queries=NUM_QUERIES
    ).to(device)
    
    # Load trained model
    load_model_checkpoint(model, MODEL_CHECKPOINT, device)
    
    # Evaluate model
    print("Starting evaluation...")
    predictions, targets = evaluate_model(
        model, test_loader, device, 
        confidence_threshold=CONFIDENCE_THRESHOLD,
        num_visualize=3
    )
    
    # Calculate basic statistics
    total_predictions = sum(len(pred['boxes']) for pred in predictions)
    total_targets = sum(len(target['boxes']) for target in targets)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total predictions: {total_predictions}")
    print(f"Total ground truth objects: {total_targets}")
    print(f"Average predictions per image: {total_predictions / len(predictions):.2f}")
    print(f"Average ground truth per image: {total_targets / len(targets):.2f}")
    
    # Class-wise statistics
    class_counts = {}
    for pred in predictions:
        for label in pred['labels']:
            class_name = test_dataset.category_names[label.item()]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print(f"\n=== Class-wise Predictions ===")
        for class_name, count in sorted(class_counts.items()):
            print(f"{class_name}: {count}")
    
    print(f"\n✓ Evaluation completed! Check saved visualization images.")

if __name__ == "__main__":
    main()
