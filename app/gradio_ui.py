"""
Full Gradio UI for the Distill & Quantize application.
"""
import gradio as gr
import torch
from core.logic import run_pipeline
from core.device_manager import get_device_manager
from settings.app_settings import (
    MODEL_TYPES,
    QUANT_TYPES,
    DEFAULT_MODEL_TYPE,
    DEFAULT_QUANT_TYPE,
    GRADIO_TITLE,
    GRADIO_DESCRIPTION,
    GRADIO_THEME,
)


class LogOutput:
    """Helper class to capture log output for Gradio."""
    def __init__(self):
        self.logs = []
    
    def log(self, msg: str):
        """Add a log message."""
        self.logs.append(msg)
        return "\n".join(self.logs)
    
    def clear(self):
        """Clear all logs."""
        self.logs = []
        return ""


def process_model(
    model_path: str,
    model_type: str,
    quant_type: str,
    progress=gr.Progress()
):
    """
    Process the model through distillation and quantization.
    
    Args:
        model_path: Path to the model
        model_type: Type of model
        quant_type: Type of quantization
        progress: Gradio progress tracker
        
    Returns:
        Log output string
    """
    log_output = LogOutput()
    
    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)
    
    try:
        if not model_path or not model_path.strip():
            return "ERROR: Please provide a model path."
        
        # Run the pipeline
        distilled_path, quantized_path = run_pipeline(
            model_path.strip(),
            model_type,
            quant_type,
            log_fn
        )
        
        log_output.log(f"\n✓ Distilled model saved to: {distilled_path}")
        log_output.log(f"✓ Quantized model saved to: {quantized_path}")
        log_output.log("\n=== SUCCESS ===")
        
        return log_output.log("")
    
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        return log_output.log("")


def get_device_info():
    """Get current device information for display."""
    device_manager = get_device_manager()
    device_name = device_manager.get_device_name()
    
    info_lines = [f"**Device:** {device_name}"]
    
    if torch.cuda.is_available():
        vram_info = device_manager.get_vram_info()
        if vram_info:
            total, allocated, free = vram_info
            info_lines.append(f"**VRAM:** {allocated:.2f}GB / {total:.2f}GB (Free: {free:.2f}GB)")
    
    info_lines.append("\n*The app will automatically use CPU if VRAM is limited.*")
    
    return "\n".join(info_lines)


def create_ui():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(theme=GRADIO_THEME, title=GRADIO_TITLE) as demo:
        gr.Markdown(f"# {GRADIO_TITLE}")
        gr.Markdown(GRADIO_DESCRIPTION)
        
        # Device information panel
        with gr.Row():
            device_info = gr.Markdown(get_device_info())
        
        with gr.Row():
            with gr.Column(scale=2):
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Enter path to model folder or file...",
                    info="Path to HuggingFace model folder or PyTorch .pt/.bin file"
                )
            
            with gr.Column(scale=1):
                model_type = gr.Dropdown(
                    choices=MODEL_TYPES,
                    value=DEFAULT_MODEL_TYPE,
                    label="Model Type",
                    info="Type of model to load"
                )
        
        with gr.Row():
            quant_type = gr.Dropdown(
                choices=QUANT_TYPES,
                value=DEFAULT_QUANT_TYPE,
                label="Quantization Type",
                info="TorchAO quantization method"
            )
            
            run_btn = gr.Button(
                "Run Distill + Quantize",
                variant="primary",
                size="lg"
            )
        
        with gr.Row():
            log_output = gr.Textbox(
                label="Live Log Output",
                lines=20,
                max_lines=30,
                interactive=False,
                placeholder="Logs will appear here..."
            )
        
        # Connect the button to the processing function
        run_btn.click(
            fn=process_model,
            inputs=[model_path, model_type, quant_type],
            outputs=[log_output]
        ).then(
            fn=lambda: get_device_info(),
            outputs=[device_info]
        )
        
        # Example section
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                ["microsoft/phi-2", "HuggingFace folder", "INT8 (dynamic)"],
                ["meta-llama/Llama-2-7b-hf", "HuggingFace folder", "INT4 (weight-only)"],
            ],
            inputs=[model_path, model_type, quant_type],
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
