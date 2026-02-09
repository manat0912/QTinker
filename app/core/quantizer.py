import os
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import traceback
from compression_toolkit import save_model_robust

# --- Configuration for output paths ---
OUTPUT_DIR = Path("quantized_models_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def quantize_model(model_id_or_path: str, quantization_level: str, log_fn=print):
    """
    Loads an AI model, quantizes it, and saves the quantized model as a zip.
    """
    if not model_id_or_path:
        raise ValueError("Please provide a Hugging Face Model ID or a path to a local model directory.")

    log_fn(f"[{model_id_or_path}] Attempting to quantize model.")
    log_fn(f"[{model_id_or_path}] Desired quantization level: {quantization_level}")

    # Create a unique name for the saved quantized model directory
    safe_model_name = model_id_or_path.replace('/', '__').replace('\\', '__').replace('.', '_')
    quantized_model_base_name = f"quantized_{safe_model_name}_{quantization_level.replace(' ', '_').replace('(', '').replace(')', '')}"
    quantized_model_save_path = OUTPUT_DIR / quantized_model_base_name

    try:
        # Determine quantization configuration based on selection
        bnb_config = None
        if "8-bit" in quantization_level:
            log_fn(f"[{model_id_or_path}] Configuring for 8-bit quantization (NF8).")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8", 
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            )
        elif "4-bit" in quantization_level:
            log_fn(f"[{model_id_or_path}] Configuring for 4-bit quantization (NF4).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            )
        elif "FP16" in quantization_level:
            log_fn(f"[{model_id_or_path}] Configuring for FP16 (Half-Precision).")
            pass 
        else:
            raise ValueError(f"Unsupported quantization level: {quantization_level}")

        # --- Load Model and Tokenizer ---
        log_fn(f"[{model_id_or_path}] Loading model and tokenizer from: {model_id_or_path}...")
        
        load_torch_dtype = torch.float32 # Default
        if torch.cuda.is_available():
            if "FP16" in quantization_level:
                load_torch_dtype = torch.float16
            elif bnb_config and bnb_config.bnb_4bit_compute_dtype:
                load_torch_dtype = bnb_config.bnb_4bit_compute_dtype

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=load_torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        log_fn(f"[{model_id_or_path}] Model and Tokenizer loaded successfully.")

        # --- Save the Quantized Model ---
        if quantized_model_save_path.exists():
            log_fn(f"[{model_id_or_path}] Cleaning up previous output directory: {quantized_model_save_path}")
            shutil.rmtree(quantized_model_save_path)

        save_model_robust(model, quantized_model_save_path, tokenizer=tokenizer)
        log_fn(f"[{model_id_or_path}] Quantized model and tokenizer saved to: {quantized_model_save_path}")

        # Zip the directory
        zip_file_path = shutil.make_archive(
            base_name=str(quantized_model_save_path),
            format='zip',
            root_dir=str(quantized_model_save_path)
        )
        log_fn(f"[{model_id_or_path}] Quantized model zipped to: {zip_file_path}")

        return zip_file_path

    except Exception as e:
        log_fn(f"[{model_id_or_path}] An error occurred during quantization: {e}")
        traceback.print_exc()
        raise e
