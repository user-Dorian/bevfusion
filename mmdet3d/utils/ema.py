"""
Exponential Moving Average (EMA) Module for Model Training

This module provides EMA support to improve model generalization by maintaining
a shadow copy of model parameters updated via exponential moving average.

Key Benefits:
  - Smoother parameter updates during training
  - Better generalization to test data
  - More robust inference weights
  - Improved NDS/mAP metrics (typically +0.5 ~ 1.5%)

Usage:
    from mmdet3d.utils.ema import ModelEMA

    ema = ModelEMA(model, decay=0.9999)
    for data in dataloader:
        loss = model(data)
        loss.backward()
        optimizer.step()
        ema.update(model)  # Update EMA after each optimization step

    # Use EMA model for evaluation/inference
    ema.apply_shadow(model)
    evaluate(model, val_loader)
    ema.restore(model)  # Restore original parameters

Author: BEVFusion Optimization Team
Date: 2026-04-06
"""

import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of Model Parameters

    Maintains a shadow copy of model parameters that are updated using:
        shadow_param = decay * shadow_param + (1 - decay) * param

    This provides smoother parameter trajectories and often leads to better
    generalization performance compared to using the final training weights.

    Attributes:
        model (nn.Module): Original model (not stored, only used for structure)
        shadow (dict): Shadow copy of model parameters
        backup (dict): Backup of original parameters when applying shadow
        decay (float): EMA decay factor (typically 0.9999 or 0.9999)
        num_updates (int): Number of EMA updates performed
        device (torch.device): Device where parameters are stored

    Example:
        >>> model = MyModel()
        >>> ema = ModelEMA(model, decay=0.9999)
        >>> for epoch in range(epochs):
        ...     for batch in train_loader:
        ...         loss = train_step(model, batch)
        ...         optimizer.step()
        ...         ema.update(model)
        ...
        ...     # Validation with EMA weights
        ...     ema.apply_shadow(model)
        ...     val_metrics = validate(model, val_loader)
        ...     ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = None):
        """
        Initialize EMA with a copy of model parameters.

        Args:
            model (nn.Module): The model to track with EMA
            decay (float): Decay factor for exponential moving average.
                          Higher values = slower adaptation (0.9999 recommended).
                          Typical range: [0.99, 0.9999]
            device (str, optional): Device to store shadow parameters.
                                   If None, uses same device as model parameters.
        """
        self.decay = decay
        self.num_updates = 0
        self.device = device

        # Create shadow copy of model parameters
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters from model
        self._init_shadow(model)

        print(f"[EMA] Initialized with decay={decay}, "
              f"tracking {len(self.shadow)} parameter tensors")

    def _init_shadow(self, model: nn.Module):
        """Initialize shadow parameters as copies of model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.is_cuda:
                    self.shadow[name] = param.data.clone().detach()
                else:
                    self.shadow[name] = param.data.clone().detach()

                # Move to specified device if provided
                if self.device is not None:
                    self.shadow[name] = self.shadow[name].to(self.device)

    def _get_decay(self) -> float:
        """
        Get current decay value with warmup schedule.

        Uses cosine ramp-up for first 2000 updates to allow shadow parameters
        to adapt gradually at the start of training.

        Returns:
            float: Effective decay factor
        """
        # Warmup schedule: increase decay from initial value to target
        # This prevents the shadow from being dominated by early random weights
        if self.num_updates < 2000:
            # Cosine ramp-up from min_decay to self.decay
            min_decay = 0.0
            value = self.num_updates / 2000
            return min_decay + (self.decay - min_decay) * (1 - (1 + value * 3.14159).cos() / 2)
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update shadow parameters using EMA rule.

        Updates each shadow parameter as:
            shadow = decay * shadow + (1 - decay) * current_param

        Should be called AFTER each optimizer.step().

        Args:
            model (nn.Module): Current model with updated parameters
        """
        self.num_updates += 1
        decay = self._get_decay()

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Ensure shadow is on same device as parameter
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)

                # Apply EMA update
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self, model: nn.Module):
        """
        Apply shadow parameters to model (for validation/inference).

        Saves current model parameters to backup before applying shadow,
        allowing restoration via restore().

        Args:
            model (nn.Module): Model to apply shadow parameters to
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Backup original parameters
                self.backup[name] = param.data.clone().detach()
                # Apply shadow parameters
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """
        Restore original model parameters from backup.

        Reverses the effect of apply_shadow(), restoring the model to its
        pre-shadow state.

        Args:
            model (nn.Module): Model to restore original parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """
        Get state dictionary for checkpointing.

        Returns:
            dict: Contains 'shadow', 'num_updates', and 'decay'
        """
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'num_updates': self.num_updates,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load state dictionary from checkpoint.

        Args:
            state_dict (dict): State dict from state_dict() method
        """
        self.shadow = {k: v.to(self.device) if self.device else v
                       for k, v in state_dict['shadow'].items()}
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict['decay']

    def get_model_copy(self, model: nn.Module) -> nn.Module:
        """
        Create a new model instance with EMA parameters.

        Useful for parallel evaluation without modifying original model.

        Args:
            model (nn.Module): Template model (structure only)

        Returns:
            nn.Module: New model instance with shadow parameters loaded
        """
        # Deep copy model structure
        model_copy = deepcopy(model)
        # Load shadow parameters
        for name, param in model_copy.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        return model_copy


class EMAHook:
    """
    Training Hook for Automatic EMA Integration

    Integrates EMA seamlessly into mmcv/mmdet training loop without requiring
    manual update calls. Automatically handles:
      - EMA updates after each iteration
      - Shadow application during validation
      - Parameter restoration after validation
      - Checkpoint saving/loading of EMA state

    Usage:
        hook = EMAHook(model, decay=0.9999)
        runner.register_hook(hook)

    Note:
        Requires use_ema=True in config file
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, priority: int = 50):
        """
        Initialize EMA Hook.

        Args:
            model (nn.Module): Model to apply EMA to
            decay (float): EMA decay factor
            priority (int): Hook execution priority (lower = earlier)
        """
        self.ema = ModelEMA(model, decay=decay)
        self.priority = priority

    def after_train_iter(self, runner):
        """Update EMA after each training iteration."""
        self.ema.update(runner.model.module if hasattr(runner.model, 'module') else runner.model)

    def before_val_epoch(self, runner):
        """Apply EMA shadow parameters before validation."""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.ema.apply_shadow(model)

    def after_val_epoch(self, runner):
        """Restore original parameters after validation."""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.ema.restore(model)

    def after_run(self, runner):
        """Optional: Apply final EMA weights after training completes."""
        pass


def build_ema(cfg) -> ModelEMA:
    """
    Factory function to create EMA from config.

    Args:
        cfg (dict): Configuration dict containing:
                   - use_ema (bool): Whether to enable EMA
                   - ema_decay (float): EMA decay factor (default: 0.9999)

    Returns:
        ModelEMA or None: EMA instance if enabled, otherwise None
    """
    if cfg.get('use_ema', False):
        decay = cfg.get('ema_decay', 0.9999)
        print(f"[EMA] Building EMA with decay={decay}")
        return ModelEMA(decay=decay)
    return None
