import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from torchao.quantization import quantize_
from torchao.quantization.configs import (
    Int4WeightOnlyConfig,
    Int8DynamicConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# FIXED OUTPUT DIRECTORIES
# -----------------------------
DISTILLED_DIR = r"C:\Users\manat\Desktop\Programs & AI\AI Programs and creations\Project Files & Workflows\AI Projects\AI app projects\distilled and quantizations\distilled"
QUANTIZED_DIR = r"C:\Users\manat\Desktop\Programs & AI\AI Programs and creations\Project Files & Workflows\AI Projects\AI app projects\distilled and quantizations\quantized"

os.makedirs(DISTILLED_DIR, exist_ok=True)
os.makedirs(QUANTIZED_DIR, exist_ok=True)


# -----------------------------
# CORE LOGIC
# -----------------------------
def log(msg, text_widget):
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)
    text_widget.update_idletasks()


def load_model(model_path, model_type, log_fn):
    log_fn(f"Loading model from: {model_path}")

    if model_type == "HuggingFace folder":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    elif model_type == "PyTorch .pt/.bin file":
        # Simple generic loader; you can customize this for your own architectures
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, torch.nn.Module):
            model = state
        else:
            raise ValueError("Loaded object is not a torch.nn.Module. Customize loader for your model.")
        tokenizer = None
        return model, tokenizer

    else:
        raise ValueError("Unsupported model type.")


def distill_model(model, log_fn):
    # Placeholder distillation logic
    log_fn("Running distillation placeholder (no real training)...")
    # Here you could:
    # - load a teacher model
    # - run a few KD steps
    # - return the student
    return model


def save_model(model, tokenizer, out_dir, log_fn, label):
    os.makedirs(out_dir, exist_ok=True)
    log_fn(f"Saving {label} model to: {out_dir}")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)
    else:
        torch.save(model, os.path.join(out_dir, "model.pt"))

    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log_fn(f"Tokenizer save failed (ok if not HF): {e}")


def apply_quantization(model, quant_type, log_fn):
    if quant_type == "INT4 (weight-only)":
        config = Int4WeightOnlyConfig(group_size=128)
    elif quant_type == "INT8 (dynamic)":
        config = Int8DynamicConfig()
    else:
        raise ValueError("Unsupported quantization type.")

    log_fn(f"Applying TorchAO quantization: {quant_type}")
    quantize_(model, config)
    return model


def run_pipeline(model_path, model_type, quant_type, text_log):
    def _log(msg):
        log(msg, text_log)

    try:
        _log("=== Starting distill + quantize pipeline ===")

        # 1. Load model
        model, tokenizer = load_model(model_path, model_type, _log)

        # 2. Distill
        distilled_model = distill_model(model, _log)

        # 3. Save distilled
        distilled_out = os.path.join(DISTILLED_DIR, "distilled_model")
        save_model(distilled_model, tokenizer, distilled_out, _log, "distilled")

        # 4. Quantize
        quantized_model = apply_quantization(distilled_model, quant_type, _log)

        # 5. Save quantized
        quantized_out = os.path.join(QUANTIZED_DIR, "quantized_model")
        save_model(quantized_model, tokenizer, quantized_out, _log, "quantized")

        _log("=== Done. Distilled + Quantized models saved. ===")
        messagebox.showinfo("Success", "Distilled and quantized models saved successfully.")

    except Exception as e:
        _log(f"ERROR: {e}")
        messagebox.showerror("Error", str(e))


# -----------------------------
# GUI APP
# -----------------------------
def main():
    root = tk.Tk()
    root.title("Distill & Quantize (TorchAO)")

    # Main frame
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # Model path
    model_path_var = tk.StringVar()

    def browse_model():
        if model_type_var.get() == "HuggingFace folder":
            path = filedialog.askdirectory(title="Select model folder")
        else:
            path = filedialog.askopenfilename(
                title="Select model file (.pt/.bin)",
                filetypes=[("PyTorch", "*.pt *.bin"), ("All files", "*.*")]
            )
        if path:
            model_path_var.set(path)

    ttk.Label(frame, text="Model source:").grid(row=0, column=0, sticky="w")
    path_entry = ttk.Entry(frame, textvariable=model_path_var, width=80)
    path_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=2)
    browse_btn = ttk.Button(frame, text="Browse...", command=browse_model)
    browse_btn.grid(row=1, column=2, sticky="e", padx=5)

    # Model type
    ttk.Label(frame, text="Model type:").grid(row=2, column=0, sticky="w", pady=(8, 0))
    model_type_var = tk.StringVar(value="HuggingFace folder")
    model_type_combo = ttk.Combobox(
        frame,
        textvariable=model_type_var,
        values=["HuggingFace folder", "PyTorch .pt/.bin file"],
        state="readonly",
        width=30,
    )
    model_type_combo.grid(row=3, column=0, sticky="w")

    # Quantization type
    ttk.Label(frame, text="Quantization type:").grid(row=2, column=1, sticky="w", pady=(8, 0))
    quant_type_var = tk.StringVar(value="INT8 (dynamic)")
    quant_type_combo = ttk.Combobox(
        frame,
        textvariable=quant_type_var,
        values=["INT4 (weight-only)", "INT8 (dynamic)"],
        state="readonly",
        width=30,
    )
    quant_type_combo.grid(row=3, column=1, sticky="w")

    # Run button
    def on_run():
        model_path = model_path_var.get().strip()
        if not model_path:
            messagebox.showwarning("Missing path", "Please select a model path.")
            return
        run_pipeline(model_path, model_type_var.get(), quant_type_var.get(), log_text)

    run_btn = ttk.Button(frame, text="Run distill + quantize", command=on_run)
    run_btn.grid(row=3, column=2, sticky="e", padx=5)

    # Log area
    ttk.Label(frame, text="Log:").grid(row=4, column=0, sticky="w", pady=(10, 0))
    log_text = tk.Text(frame, height=18, width=100)
    log_text.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(2, 0))

    frame.rowconfigure(5, weight=1)
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=0)
    frame.columnconfigure(2, weight=0)

    root.minsize(900, 450)
    root.mainloop()


if __name__ == "__main__":
    main() 
