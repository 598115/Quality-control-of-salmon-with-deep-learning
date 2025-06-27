from tqdm import tqdm
from .UnetPlusPlus import UNetPlusPlus
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class UNetTrain:
    """
    A class for training the U-Net++ model for multi-task segmentation (zones and spots).\n
    This class handles the training process including gradient scaling, early stopping,\n
    learning rate scheduling, and model checkpointing.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader  
        DataLoader for validation data
    model_lr : float, optional (default=1e-3)
        Initial learning rate for the AdamW optimizer
    model_weight_decay : float, optional (default=0.01)
        Weight decay (L2 penalty) for the AdamW optimizer
    model_betas : tuple of float, optional (default=(0.9, 0.999))
        Coefficients for computing running averages of gradient and its square
    model_droput : float, optional (default=0.2)
        Dropout probability in the U-Net++ model
    scheduler_patience : int, optional (default=5)
        Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor : float, optional (default=0.5)
        Factor by which the learning rate will be reduced
    scheduler_min_lr : float, optional (default=1e-6)
        Minimum learning rate for the scheduler
    max_grad_norm : float, optional (default=5.0)
        Maximum norm for gradient clipping
    Methods
    -------
    train(num_epochs, patience=5, min_delta=0.001, window_size=5)
        Trains the model for the specified number of epochs
        Returns tuple of (train_losses, val_losses)
    _training_step(batch, model)
        Performs one training step including forward and backward passes
        Returns float loss value
    _validation_step(batch, model)
        Performs one validation step
        Returns float loss value
    _save_checkpoint(epoch, model, train_loss, val_loss)
        Saves model checkpoint to disk
    _should_stop_early(recent_losses, epochs_without_improve, patience)
        Determines if training should stop early based on loss trends
        Returns boolean
    Notes
    -----
    - Uses mixed precision training with gradient scaling
    - Implements deep supervision if enabled in the model
    - Spots segmentation task is weighted higher (2x) than zones
    - Saves best model based on validation loss
    - Implements early stopping with patience and loss trend analysis
    """

    def __init__(self, train_loader, val_loader, model_lr=1e-3, model_weight_decay=0.01, model_betas=(0.9, 0.999), model_droput=0.2, scheduler_patience=5,
         scheduler_factor=0.5, scheduler_min_lr=1e-6, max_grad_norm=5.0):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetPlusPlus(n_classes_zone=4, n_classes_spot=3, dropout_p=model_droput).to(self.device)
        self.criterion_zone = nn.CrossEntropyLoss()
        self.criterion_spot = nn.CrossEntropyLoss()
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=model_lr,
            weight_decay=model_weight_decay,
            betas=model_betas
        )
        self.scaler = GradScaler()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

    def _training_step(self, batch, model):
        images, zone_masks, spot_masks = [x.to(self.device) for x in batch]
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=self.device.type):
            # Forward pass
            if model.use_deep_supervision and model.training:
                (zone_out, spot_out), deep_outputs = model(images)
                
                # Main losses with spot task weighted higher
                zone_loss = self.criterion_zone(zone_out, zone_masks)
                spot_loss = 2.0 * self.criterion_spot(spot_out, spot_masks)  # Weight spot loss higher
                
                # Deep supervision losses if enabled
                if model.use_deep_supervision:
                      for i, (zone_deep, spot_deep) in enumerate(deep_outputs):
                        # Applying progressive weighting (lower weight for earlier layers)
                        ds_weight = 0.2 * (i + 1) / len(deep_outputs)
                        zone_loss += ds_weight * self.criterion_zone(zone_deep, zone_masks)
                        spot_loss += ds_weight * self.criterion_spot(spot_deep, spot_masks)
                
                # Combine all losses
                total_loss = zone_loss + spot_loss
            else:
                zone_out, spot_out = model(images)
                zone_loss = self.criterion_zone(zone_out, zone_masks)
                spot_loss = 2.0 * self.criterion_spot(spot_out, spot_masks)
                total_loss = zone_loss + spot_loss
        
        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()
        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Return scalar value for loss accumulation
        return total_loss.item()

    def _validation_step(self, batch, model):
        images, zone_masks, spot_masks = [x.to(self.device) for x in batch]
        with torch.no_grad():
            zone_out, spot_out = model(images)
            loss = self.criterion_zone(zone_out, zone_masks) + self.criterion_spot(spot_out, spot_masks)
        return loss.item()

    def _save_checkpoint(self, epoch, model, train_loss, val_loss):
        save_path = os.path.join('saved_models', 'saved_segm_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, save_path)
        print(f'Saved best model with validation loss: {val_loss:.4f}')

    def _should_stop_early(self, recent_losses, epochs_without_improve, patience):
        if epochs_without_improve < patience:
            return False
        return not all(x > y for x, y in zip(recent_losses[:-1], recent_losses[1:]))

    def train(self, num_epochs, patience=5, min_delta=0.001, window_size=5):
        model = self.model
        best_val_loss = float('inf')
        epochs_without_improve = 0
        best_model_state = None
        train_losses, val_losses = [], []
        recent_losses = []
        
        last_lr = self.optimizer.param_groups[0]['lr']

        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0  # Initialize as float

            with tqdm(total=len(self.train_loader), desc='Training', position=0, leave=True) as pbar:
                for batch in self.train_loader:
                    loss = self._training_step(batch, model)
                    epoch_loss += float(loss)
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
                
            avg_train_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0  # Initialize as float

            with tqdm(total=len(self.val_loader), desc='Validation', position=0, leave=True) as pbar:
                for batch in self.val_loader:
                    loss = self._validation_step(batch, model)
                    val_loss += float(loss)
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
                
            avg_val_loss = val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            # Update learning rate
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Reset patience if learning rate changed
            if current_lr != last_lr:
                print(f'\nLearning rate changed from {last_lr:.2e} to {current_lr:.2e}')
                print('Subtracting patience by 1 to give more time for convergence.')
                epochs_without_improve = epochs_without_improve - 1
                if epochs_without_improve < 0:
                    epochs_without_improve = 0
                recent_losses = []
                last_lr = current_lr
                
            # Check improvement
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improve = 0
                best_model_state = model.state_dict().copy()
                self._save_checkpoint(epoch, model, avg_train_loss, avg_val_loss)
            else:
                epochs_without_improve += 1
            
            # Update recent losses for trend analysis
            recent_losses.append(avg_val_loss)
            if len(recent_losses) > window_size:
                recent_losses.pop(0)
            
            # Print epoch statistics
            print(f'Epoch [{epoch+1}/{num_epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f} - '
                  f'Val Loss: {avg_val_loss:.4f} - '
                  f'LR: {current_lr:.2e} - '
                  f'Best: {best_val_loss:.4f} - '
                  f'No improve: {epochs_without_improve}/{patience}')
            
            # Early stopping check
            if self._should_stop_early(recent_losses, epochs_without_improve, patience):
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                print(f'Best validation loss: {best_val_loss:.4f}')
                model.load_state_dict(best_model_state)
                break
        
        print("\nTraining completed!")
        torch.cuda.empty_cache()
        return train_losses, val_losses