"""whisper_til_model: A Flower / PyTorch Task-Incremental Learning app using OpenAI Whisper."""

from transformers import WhisperForConditionalGeneration
from typing import Optional
import torch
import torch.nn as nn


class WhisperTaskIncrementalClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_classes_per_task: int, num_tasks: int):
        """
        Task-incremental classifier using Conv1d + Flatten,
        same architecture as centralized setup, but with per-task heads.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.current_task = 0

        # Shared feature projector: Conv1d(384→hidden) + ReLU
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)   # flatten [B, hidden_dim, 1500] → [B, hidden_dim*1500]

        # Task-specific heads: each takes flattened [hidden_dim*1500]
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 1500, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoder output [B, 1500, 384]
            task_id: task id (default = self.current_task)

        Returns:
            logits: [B, num_classes_per_task]
        """
        if task_id is None:
            task_id = self.current_task

        # Transpose for Conv1d: [B, 1500, 384] → [B, 384, 1500]
        x = x.transpose(1, 2)
        x = self.conv1d(x)           # [B, hidden_dim, 1500]
        x = self.relu(x)
        x = self.flatten(x)          # [B, hidden_dim*1500]

        return self.task_heads[task_id](x)

    def set_task(self, task_id: int):
        """Set the default task head if not specified during forward."""
        if 0 <= task_id < len(self.task_heads):
            self.current_task = task_id
        else:
            raise ValueError(f"Invalid task_id {task_id}. Must be in [0, {len(self.task_heads) - 1}]")


def get_model(device, num_classes_per_task, hidden_dim,
              num_tasks, freeze_encoder: bool = True, compile_encoder: bool = True):
    """
    Create model: Whisper-tiny encoder + Conv1d-based task-incremental head.

    Args:
        device: Target torch device
        num_classes_per_task: Classes per task
        hidden_dim: Conv1d output channels
        num_tasks: Total number of tasks
        freeze_encoder: Whether to freeze Whisper encoder
        compile_encoder: Whether to torch.compile the encoder

    Returns:
        encoder: Whisper encoder (frozen if specified)
        classifier: Task-incremental classifier
    """
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    encoder = whisper_model.get_encoder().to(device)

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    if compile_encoder:
        encoder = torch.compile(encoder)

    classifier = WhisperTaskIncrementalClassifier(
        input_dim=384,               # Whisper-tiny hidden size
        hidden_dim=hidden_dim,
        num_classes_per_task=num_classes_per_task,
        num_tasks=num_tasks
    ).to(device)

    return encoder, classifier
