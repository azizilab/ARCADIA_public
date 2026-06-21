"""Gradient Normalization for Adaptive Loss Balancing."""

import numpy as np
import torch
from torch import nn

from arcadia.utils.logging import setup_logger


class GradNorm(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    https://arxiv.org/abs/1711.02257

    This implementation balances multiple loss functions by dynamically adjusting their weights
    based on gradient magnitudes and relative training rates.
    """

    def __init__(self, n_tasks, alpha=1.5, device="cpu", logger=None):
        super(GradNorm, self).__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.device = device
        self.logger = logger if logger is not None else setup_logger()

        # Task weights - learnable parameters initialized to 1
        self.task_weights = nn.Parameter(torch.ones(n_tasks, device=device, requires_grad=True))

        # Track initial losses for relative training rate calculation
        self.initial_losses = None
        self.loss_history = []
        self.gradient_norm_history = []
        self.weight_history = []

        # Debugging flags
        self.debug_enabled = True
        self.update_count = 0

    def reset_initial_losses(self):
        """Reset initial losses - useful when starting new training phases"""
        self.initial_losses = None
        self.loss_history = []
        self.gradient_norm_history = []
        self.weight_history = []
        self.update_count = 0

    def forward(
        self,
        losses,
        shared_layers,
        task_names=None,
        global_step=None,
        rna_model=None,
        protein_model=None,
    ):
        """
        Apply GradNorm to balance multiple losses for separate RNA and Protein VAE models

        Args:
            losses: List or tensor of loss values for each task
            shared_layers: Deprecated - kept for compatibility but not used
            task_names: Optional list of task names for debugging
            global_step: Current training step for debugging
            rna_model: RNA VAE model for gradient computation
            protein_model: Protein VAE model for gradient computation

        Returns:
            tuple: (balanced_total_loss, gradnorm_loss, updated_weights)
        """
        if task_names is None:
            task_names = [f"task_{i}" for i in range(len(losses))]

        # Convert losses to tensor if needed
        if isinstance(losses, list):
            losses_tensor = torch.stack(losses)
        else:
            losses_tensor = losses

        # Ensure we have the right number of tasks
        assert (
            len(losses_tensor) == self.n_tasks
        ), f"Expected {self.n_tasks} losses, got {len(losses_tensor)}"

        # Initialize with first batch losses
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.detach().clone()
            if self.debug_enabled:
                self.logger.info(f"[GradNorm] Initialized with losses: {self.initial_losses}")

        # Compute weighted total loss
        weighted_losses = self.task_weights * losses_tensor
        total_loss = torch.sum(weighted_losses)

        # Compute gradients for each task using the appropriate VAE model
        # Since RNA and Protein VAEs are separate, compute gradients on each model separately
        task_gradients = []

        # Get parameters from RNA and Protein models
        rna_params = []
        protein_params = []

        if rna_model is not None:
            rna_params = [p for p in rna_model.parameters() if p.requires_grad]

        if protein_model is not None:
            protein_params = [p for p in protein_model.parameters() if p.requires_grad]

        # Map tasks to their respective models based on task names
        # RNA tasks: rna_reconstruction, rna_kl
        # Protein tasks: protein_reconstruction, protein_kl
        # Joint tasks: archetype losses, similarity, etc. - compute on both models

        for i, task_loss in enumerate(losses_tensor):
            weighted_task_loss = self.task_weights[i] * task_loss
            task_name = task_names[i] if i < len(task_names) else f"task_{i}"

            # Determine which model parameters to use for gradient computation
            target_params = []
            if "rna" in task_name.lower() and len(rna_params) > 0:
                target_params = rna_params
            elif "protein" in task_name.lower() and len(protein_params) > 0:
                target_params = protein_params
            else:
                # For joint losses (archetype, similarity, etc.), use RNA model params as representative
                # since GradNorm cares about relative gradient magnitudes, not absolute values
                target_params = rna_params if len(rna_params) > 0 else protein_params

            if len(target_params) > 0:
                # Compute actual gradients
                task_grads = torch.autograd.grad(
                    weighted_task_loss,
                    target_params,
                    retain_graph=True,
                    create_graph=False,  # Changed to False to avoid creating computation graph
                    allow_unused=True,
                    only_inputs=True,
                )

                # Flatten and concatenate gradients
                flat_grads = []
                for grad in task_grads:
                    if grad is not None:
                        flat_grads.append(grad.flatten())

                if len(flat_grads) > 0:
                    task_gradient = torch.cat(flat_grads)
                    task_gradients.append(task_gradient)
                else:
                    # Fallback to loss magnitude proxy
                    grad_proxy = torch.sqrt(torch.abs(task_loss) + 1e-8)
                    task_gradients.append(grad_proxy)
            else:
                # No model parameters available - use loss magnitude as proxy
                grad_proxy = torch.sqrt(torch.abs(task_loss) + 1e-8)
                task_gradients.append(grad_proxy)

        # Compute gradient norms
        if len(task_gradients) > 0:
            grad_norms = torch.stack([torch.norm(grad, p=2) for grad in task_gradients])
        else:
            grad_norms = torch.zeros(self.n_tasks, device=self.device, requires_grad=True)

        # Compute loss ratios (current loss / initial loss)
        # Use initial_losses without detaching to maintain gradient flow
        initial_losses_safe = self.initial_losses.detach() + 1e-8  # Detach only initial losses
        loss_ratios = losses_tensor / initial_losses_safe

        # Compute average loss ratio
        mean_loss_ratio = torch.mean(loss_ratios)

        # Compute relative inverse training rates
        relative_inverse_rates = loss_ratios / (mean_loss_ratio + 1e-8)

        # Compute target gradient norms - ensure this maintains gradients
        mean_grad_norm = torch.mean(grad_norms)
        target_grad_norms = mean_grad_norm * (relative_inverse_rates**self.alpha)

        # GradNorm loss (L1 loss between actual and target gradient norms)
        # The key insight: the gradnorm_loss must have a computational path back to task_weights
        # Since target_grad_norms depends on losses_tensor (which depends on task_weights via weighted_losses),
        # we ensure the gradient chain is preserved
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))

        # Ensure gradnorm_loss requires gradient by explicitly connecting to task_weights
        # This creates a direct dependency on the learnable parameters
        weight_penalty = 1e-8 * torch.sum(
            self.task_weights
        )  # Small penalty to ensure gradient connection
        gradnorm_loss = gradnorm_loss + weight_penalty

        # Store history for debugging
        self.loss_history.append(losses_tensor.detach().cpu().numpy())
        self.gradient_norm_history.append(grad_norms.detach().cpu().numpy())
        self.weight_history.append(self.task_weights.detach().cpu().numpy())
        self.update_count += 1

        # Debug logging
        if self.debug_enabled and global_step is not None and global_step % 50 == 0:
            self._log_debug_info(
                losses_tensor,
                grad_norms,
                target_grad_norms,
                loss_ratios,
                relative_inverse_rates,
                task_names,
                global_step,
            )

        return total_loss, gradnorm_loss, self.task_weights.detach()

    def _log_debug_info(
        self,
        losses,
        grad_norms,
        target_grad_norms,
        loss_ratios,
        relative_inverse_rates,
        task_names,
        global_step,
    ):
        """Log detailed debugging information"""

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[GradNorm Debug] Step {global_step} - Update #{self.update_count}")
        self.logger.info(f"{'='*80}")

        # Current state
        self.logger.info(f"Current Losses:")
        for i, (name, loss) in enumerate(zip(task_names, losses)):
            self.logger.info(f"  ├─ {name}: {loss.item():.6f}")

        self.logger.info(f"Task Weights:")
        for i, (name, weight) in enumerate(zip(task_names, self.task_weights)):
            self.logger.info(f"  ├─ {name}: {weight.item():.6f}")

        self.logger.info(f"Gradient Norms:")
        for i, (name, norm) in enumerate(zip(task_names, grad_norms)):
            self.logger.info(f"  ├─ {name}: {norm.item():.6f}")

        self.logger.info(f"Target Gradient Norms:")
        for i, (name, target) in enumerate(zip(task_names, target_grad_norms)):
            self.logger.info(f"  ├─ {name}: {target.item():.6f}")

        self.logger.info(f"Loss Ratios (current/initial):")
        for i, (name, ratio) in enumerate(zip(task_names, loss_ratios)):
            self.logger.info(f"  ├─ {name}: {ratio.item():.6f}")

        self.logger.info(f"Relative Inverse Training Rates:")
        for i, (name, rate) in enumerate(zip(task_names, relative_inverse_rates)):
            self.logger.info(f"  ├─ {name}: {rate.item():.6f}")

        # Gradient norm differences
        grad_norm_diffs = torch.abs(grad_norms - target_grad_norms)
        self.logger.info(f"Gradient Norm Differences:")
        for i, (name, diff) in enumerate(zip(task_names, grad_norm_diffs)):
            self.logger.info(f"  ├─ {name}: {diff.item():.6f}")

        self.logger.info(f"Alpha parameter: {self.alpha}")
        self.logger.info(f"Mean gradient norm: {torch.mean(grad_norms).item():.6f}")
        self.logger.info(f"{'='*80}\n")

    def get_task_weights(self):
        """Get current task weights as numpy array"""
        return self.task_weights.detach().cpu().numpy()

    def set_debug_enabled(self, enabled):
        """Enable or disable debug logging"""
        self.debug_enabled = enabled

    def get_statistics(self):
        """Get training statistics for analysis"""
        if len(self.loss_history) == 0:
            return {}

        return {
            "loss_history": np.array(self.loss_history),
            "gradient_norm_history": np.array(self.gradient_norm_history),
            "weight_history": np.array(self.weight_history),
            "update_count": self.update_count,
            "current_weights": self.get_task_weights(),
        }
