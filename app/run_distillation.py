import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from model_loader import ModelLoader
from registry import get_strategy, get_profile
from distillation import PatientKD
from compression_toolkit import save_model_robust # Import the robust save function

from torchao.quantization import (
    quantize_,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig as Int8DynamicConfig,
)

# A simple dummy dataloader for demonstration
def create_dummy_dataloader(tokenizer, batch_size=2):
    dummy_input = "This is a dummy sentence for distillation."
    inputs = tokenizer(dummy_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    return DataLoader(dataset, batch_size=batch_size)

def run_distillation_pipeline(
    teacher_model_path,
    student_model_path,
    custom_model_path=None,
    strategy_name=None,
    output_dir="distilled_models",
    quantization_type=None, # "INT4", "INT8", or None
    log_fn=print,
    **kwargs
):
    """
    Main pipeline for model distillation and optional quantization.
    """
    # 1. Load models
    loader = ModelLoader(
        teacher_path=teacher_model_path,
        student_path=student_model_path,
        custom_path=custom_model_path
    )
    loader.load_all()
    models = loader.get_models()

    teacher_model, teacher_tokenizer = models["teacher"]
    student_model, student_tokenizer = models["student"]
    custom_model, _ = models["custom"] if custom_model_path else (None, None)
    
    # 2. Get distillation strategy
    if not strategy_name:
        profile = get_profile(teacher_model_path)
        strategy_name = profile['strategy']
        log_fn(f"No strategy specified, using profile-based strategy: {strategy_name}")

    strategy_params = {}
    if strategy_name == 'patient_kd':
        profile = get_profile(teacher_model_path) # Get layers from profile
        strategy_params = profile.get('patient_kd_layers', {})

    strategy = get_strategy(strategy_name, teacher_model, student_model, custom_model, strategy_params)

    # 3. Sanity Check / Dry Run
    log_fn("Running sanity check (dry run)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    if custom_model:
        custom_model.to(device)
    # Ensure strategy (and its projections) are on the correct device
    if isinstance(strategy, torch.nn.Module):
        strategy.to(device)

    dummy_batch = create_dummy_dataloader(student_tokenizer, batch_size=1)
    for batch in dummy_batch:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            custom_outputs = None
            if custom_model:
                custom_outputs = custom_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            student_outputs = student_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Print shapes
            log_fn(f"Teacher hidden states: {[h.shape for h in teacher_outputs.hidden_states]}")
            log_fn(f"Student hidden states: {[h.shape for h in student_outputs.hidden_states]}")
            if custom_model:
                log_fn(f"Custom hidden states: {[h.shape for h in custom_outputs.hidden_states]}")
            
            # Check loss computation
            loss = strategy.compute_loss(student_outputs, teacher_outputs, custom_outputs)
            log_fn(f"Sanity check loss: {loss.item()}")
        break
    log_fn("Sanity check passed.")

    # 4. Simple training loop
    optimizer = AdamW(student_model.parameters(), lr=5e-5)
    dataloader = create_dummy_dataloader(student_tokenizer)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Already defined above
    # teacher_model.to(device) # Already moved
    # student_model.to(device) # Already moved
    # if custom_model:
    #     custom_model.to(device) # Already moved

    student_model.train()
    
    num_epochs = 1 # Keep it short for a demo
    log_fn("Starting distillation training...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids, attention_mask = [b.to(device) for b in batch]
            
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                custom_outputs = None
                if custom_model:
                    custom_outputs = custom_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            student_outputs = student_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            loss = strategy.compute_loss(student_outputs, teacher_outputs, custom_outputs)
            loss.backward()
            optimizer.step()
            
            log_fn(f"Epoch {epoch+1}, Loss: {loss.item()}")

    log_fn("Distillation training finished.")
    
    # 4. Save distilled model
    distilled_model_dir = os.path.join(output_dir, "distilled")
    save_model_robust(student_model, distilled_model_dir, tokenizer=student_tokenizer)
    log_fn(f"Distilled model saved to {distilled_model_dir}")

    # 5. Apply quantization if specified
    if quantization_type:
        log_fn(f"Applying {quantization_type} quantization...")
        try:
            from configs.torchao_configs import get_quantization_config
            config = get_quantization_config(quantization_type)
            
            if hasattr(config, "__call__") and not isinstance(config, type):
                # Update config parameters from kwargs if possible
                if hasattr(config, "alpha") and "smoothquant_alpha" in kwargs:
                    config.alpha = kwargs["smoothquant_alpha"]
                if hasattr(config, "kwargs") and "gptq_group_size" in kwargs:
                    config.kwargs["group_size"] = kwargs["gptq_group_size"]
                    
                student_model = config(student_model)
            else:
                quantize_(student_model, config)
        except Exception as e:
            log_fn(f"Quantization failed: {e}")
            return
        
        quantized_model_dir = os.path.join(output_dir, "quantized")
        save_model_robust(student_model, quantized_model_dir, tokenizer=student_tokenizer)
        log_fn(f"Quantized model saved to {quantized_model_dir}")

if __name__ == '__main__':
    # Example usage of the pipeline
    run_distillation_pipeline(
        teacher_model_path='bert-base-uncased',
        student_model_path='prajjwal1/bert-tiny',
        strategy_name='patient_kd',
        quantization_type="INT8"
    )
