"""
Knowledge Distillation implementation with teacher-student training.
"""
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from core.device_manager import get_device_manager


class PromptDataset(Dataset):
    """Dataset for loading training prompts from a text file."""
    
    def __init__(self, tokenizer, file_path: str, max_length: int = 256):
        """
        Initialize dataset.
        
        Args:
            tokenizer: Tokenizer to use
            file_path: Path to text file with one prompt per line
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Load prompts from file
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.lines = [l.strip() for l in f.readlines() if l.strip()]
        else:
            # Default prompts if file doesn't exist
            self.lines = [
                "Explain quantum computing in simple terms.",
                "Write a short story about a dragon.",
                "Summarize the causes of World War I.",
            ]
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        # Flatten batch dimension
        return {k: v.squeeze(0) for k, v in enc.items()}


def load_config() -> dict:
    """Load settings from YAML config."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def load_paths_config() -> dict:
    """Load paths from YAML config."""
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {"train_prompts": "data/train_prompts.txt"}


def distill_teacher_student(
    student: torch.nn.Module,
    student_tokenizer,
    teacher_model_path: str,
    teacher_type: str,
    device_manager,
    log_fn=None,
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: int = 100,
    learning_rate: float = 5e-5,
    temperature: float = 1.0,
    alpha: float = 0.5
) -> torch.nn.Module:
    """
    Perform teacher-student knowledge distillation.
    
    Args:
        student: Student model to train
        student_tokenizer: Student tokenizer
        teacher_model_path: Path to teacher model
        teacher_type: Type of teacher model
        device_manager: Device manager instance
        log_fn: Optional logging function
        epochs: Number of training epochs
        batch_size: Batch size
        max_steps: Maximum training steps
        learning_rate: Learning rate
        temperature: Temperature for KL divergence
        alpha: Weight for KL loss (1-alpha for CE loss)
        
    Returns:
        Distilled student model
    """
    if log_fn:
        log_fn("=" * 50)
        log_fn("Starting Teacher-Student Knowledge Distillation")
        log_fn("=" * 50)
    
    device = device_manager.get_device()
    
    # Load teacher model
    if log_fn:
        log_fn(f"Loading teacher model from: {teacher_model_path}")
    
    if teacher_type == "HuggingFace folder":
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    else:
        teacher = torch.load(teacher_model_path, map_location="cpu")
        teacher_tokenizer = student_tokenizer  # Fallback
    
    if teacher_tokenizer is None:
        teacher_tokenizer = student_tokenizer
    
    # Move models to device
    teacher = device_manager.move_model_to_device(teacher)
    student = device_manager.move_model_to_device(student)
    
    teacher.eval()
    student.train()
    
    if log_fn:
        log_fn(f"Teacher model loaded on: {device_manager.get_device_name()}")
        log_fn(f"Student model on: {device_manager.get_device_name()}")
    
    # Load dataset
    paths_config = load_paths_config()
    train_prompts_path = paths_config.get("train_prompts", "data/train_prompts.txt")
    train_prompts_path = Path(__file__).parent.parent / train_prompts_path
    
    dataset = PromptDataset(student_tokenizer, str(train_prompts_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if log_fn:
        log_fn(f"Loaded {len(dataset)} training prompts")
        log_fn(f"Training for {epochs} epochs, max {max_steps} steps")
        log_fn(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        log_fn(f"Temperature: {temperature}, Alpha (KL weight): {alpha}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    
    step = 0
    total_loss = 0.0
    
    for epoch in range(epochs):
        if log_fn:
            log_fn(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        for batch_idx, batch in enumerate(loader):
            if step >= max_steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False
                )
                teacher_logits = teacher_outputs.logits
            
            # Get student predictions
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False
            )
            student_logits = student_outputs.logits
            
            # Align sequence lengths if needed
            seq_len = min(teacher_logits.size(1), student_logits.size(1))
            teacher_logits = teacher_logits[:, :seq_len, :]
            student_logits = student_logits[:, :seq_len, :]
            
            # KL divergence loss (teacher -> student)
            teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs.exp(),
                reduction="batchmean"
            ) * (temperature ** 2)
            
            # Cross-entropy loss (student predictions on labels)
            # Shift logits and labels for next-token prediction
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            )
            ce_loss = (ce_loss * shift_mask.view(-1)).sum() / shift_mask.sum()
            
            # Combined loss
            loss = alpha * kl_loss + (1 - alpha) * ce_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 10 == 0 and log_fn:
                avg_loss = total_loss / 10
                log_fn(f"Step {step}/{max_steps} - Loss: {loss.item():.4f} (KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f})")
                total_loss = 0.0
        
        if step >= max_steps:
            break
    
    # Move models back to CPU
    student = student.cpu()
    teacher = teacher.cpu()
    device_manager.clear_cache()
    
    if log_fn:
        log_fn("=" * 50)
        log_fn("Knowledge Distillation Complete!")
        log_fn("=" * 50)
    
    return student


def distill_model(
    model: torch.nn.Module,
    tokenizer,
    device_manager=None,
    log_fn=None,
    distillation_config: dict = None
) -> torch.nn.Module:
    """
    Distill a model using either placeholder or teacher-student method.
    
    Args:
        model: The student model to distill
        tokenizer: Model tokenizer
        device_manager: Device manager instance
        log_fn: Optional logging function
        distillation_config: Distillation configuration dict
        
    Returns:
        Distilled model
    """
    if device_manager is None:
        device_manager = get_device_manager(log_fn)
    
    # Load config if not provided
    if distillation_config is None:
        config = load_config()
        distillation_config = config.get("distillation", {})
    
    if not distillation_config.get("enabled", True):
        if log_fn:
            log_fn("Distillation disabled, skipping...")
        return model
    
    mode = distillation_config.get("mode", "placeholder")
    
    if mode == "placeholder":
        if log_fn:
            log_fn("Running placeholder distillation (no training)...")
            log_fn(f"Distillation running on: {device_manager.get_device_name()}")
        
        # Ensure model is on the right device
        current_device = next(model.parameters()).device
        if current_device != device_manager.get_device():
            if log_fn:
                log_fn(f"Moving model to {device_manager.get_device_name()} for distillation...")
            try:
                model = device_manager.move_model_to_device(model)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if log_fn:
                        log_fn("⚠️  GPU OOM during distillation. Using CPU...")
                    device_manager.switch_to_cpu()
                    model = model.to(torch.device("cpu"))
        
        device_manager.clear_cache()
        return model
    
    elif mode == "teacher_student":
        teacher_path = distillation_config.get("teacher_model_path", "")
        if not teacher_path:
            if log_fn:
                log_fn("⚠️  Teacher model path not specified, using placeholder mode")
            return distill_model(model, tokenizer, device_manager, log_fn, {"mode": "placeholder", "enabled": True})
        
        return distill_teacher_student(
            student=model,
            student_tokenizer=tokenizer,
            teacher_model_path=teacher_path,
            teacher_type=distillation_config.get("teacher_type", "HuggingFace folder"),
            device_manager=device_manager,
            log_fn=log_fn,
            epochs=distillation_config.get("epochs", 1),
            batch_size=distillation_config.get("batch_size", 4),
            max_steps=distillation_config.get("max_steps", 100),
            learning_rate=distillation_config.get("learning_rate", 5e-5),
            temperature=distillation_config.get("temperature", 1.0),
            alpha=distillation_config.get("alpha", 0.5)
        )
    
    else:
        raise ValueError(f"Unknown distillation mode: {mode}")
