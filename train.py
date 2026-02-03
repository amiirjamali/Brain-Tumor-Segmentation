#!/usr/bin/env python3
"""
Brain Tumor Segmentation Training Script
Main training script based on MONAI framework
"""

import os
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import torch
import argparse
import yaml


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


def get_transforms():
    """Get training and validation transforms"""
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    return train_transform, val_transform


def create_datasets(data_dir, train_transform, val_transform):
    """Create training and validation datasets"""
    train_ds = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    
    val_ds = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    
    return train_ds, val_ds


def train(config):
    """Main training function"""
    # Set deterministic training
    set_determinism(seed=config['seed'])
    
    # Setup device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_ds, val_ds = create_datasets(
        data_dir=config['data_dir'],
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Create model
    model = SegResNet(
        blocks_down=config['model']['blocks_down'],
        blocks_up=config['model']['blocks_up'],
        init_filters=config['model']['init_filters'],
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        dropout_prob=config['model']['dropout_prob'],
    ).to(device)
    
    # Loss function and optimizer
    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['max_epochs']
    )
    
    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    total_start = time.time()
    
    for epoch in range(config['max_epochs']):
        epoch_start = time.time()
        print("-" * 50)
        print(f"Epoch {epoch + 1}/{config['max_epochs']}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 10 == 0:
                print(
                    f"Step {step}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Time: {time.time() - step_start:.2f}s"
                )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average loss
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # Validation phase
        if (epoch + 1) % config['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    
                    # Inference
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=(240, 240, 160),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                    )
                    
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    
                    # Calculate metrics
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                
                # Aggregate metrics
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                
                # Reset metrics
                dice_metric.reset()
                dice_metric_batch.reset()
                
                # Save best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    
                    # Save model
                    torch.save(
                        model.state_dict(),
                        os.path.join(config['output_dir'], "best_model.pth")
                    )
                    print(f"Saved new best model with Dice: {best_metric:.4f}")
                
                print(
                    f"Current epoch: {epoch + 1}\n"
                    f"Mean Dice: {metric:.4f}\n"
                    f"TC: {metric_tc:.4f}, WT: {metric_wt:.4f}, ET: {metric_et:.4f}\n"
                    f"Best Mean Dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
        
        print(f"Epoch time: {time.time() - epoch_start:.2f}s")
    
    total_time = time.time() - total_start
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    return {
        'epoch_loss_values': epoch_loss_values,
        'metric_values': metric_values,
        'best_metric': best_metric,
        'best_epoch': best_metric_epoch
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation Training")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run training
    results = train(config)