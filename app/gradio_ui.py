"""
Full Gradio UI for the Distill & Quantize application with Knowledge Distillation and Local LLM support.
"""
import gradio as gr
import torch
import yaml
from pathlib import Path
from core.logic import run_pipeline
from core.device_manager import get_device_manager
from core.local_llm import get_local_llm_client
from settings.app_settings import (
    MODEL_TYPES,
    QUANT_TYPES,
    DEFAULT_MODEL_TYPE,
    DEFAULT_QUANT_TYPE,
    GRADIO_TITLE,
    GRADIO_DESCRIPTION,
    GRADIO_TITLE,
    GRADIO_DESCRIPTION,
)


def create_theme():
    """Create a custom dark theme with gradient background."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="radial-gradient(circle at 50% 0%, #1a1c2e 0%, #0f1016 100%)",
        block_background_fill="#1a1c2e",
        button_primary_background_fill="linear-gradient(90deg, #00f2fe 0%, #4facfe 100%)",
        button_primary_text_color="white",
        block_title_text_color="white",
        block_label_text_color="white",
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


def load_config():
    """Load settings from YAML config."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def save_config(config: dict):
    """Save settings to YAML config."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def detect_local_llm():
    """Detect available local LLM providers."""
    from core.local_llm import get_local_llm_client
    client = get_local_llm_client()
    provider = client.detect_provider()
    models = client.get_available_models() if provider else []
    return provider or "None detected", ", ".join(models[:5]) if models else "No models found"


def get_local_llm_client_instance():
    """Get local LLM client instance."""
    from core.local_llm import get_local_llm_client as get_client
    config = load_config()
    llm_config = config.get("local_llm", {})
    return get_client(
        provider=llm_config.get("provider", "auto"),
        base_url=llm_config.get("base_url", "http://localhost:1234/v1"),
        ollama_url=llm_config.get("ollama_url", "http://localhost:11434"),
        api_key=llm_config.get("api_key", ""),
        model_name=llm_config.get("model_name", "")
    )


def process_model(
    model_path: str,
    model_type: str,
    quant_type: str,
    distillation_mode: str,
    teacher_model_path: str,
    teacher_model_type: str,
    use_local_llm: bool,
    local_llm_provider: str,
    local_llm_url: str,
    progress=gr.Progress()
):
    """
    Process the model through distillation and quantization.
    
    Args:
        model_path: Path to the student model
        model_type: Type of model
        quant_type: Type of quantization
        distillation_mode: Distillation mode (placeholder or teacher_student)
        teacher_model_path: Path to teacher model (if teacher_student mode)
        teacher_model_type: Type of teacher model
        use_local_llm: Whether to use local LLM
        local_llm_provider: Local LLM provider
        local_llm_url: Local LLM URL
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
        
        # Update config with UI settings
        config = load_config()
        if "distillation" not in config:
            config["distillation"] = {}
        
        config["distillation"]["mode"] = distillation_mode
        config["distillation"]["enabled"] = True
        
        if distillation_mode == "teacher_student":
            if not teacher_model_path or not teacher_model_path.strip():
                return "ERROR: Teacher model path is required for teacher-student distillation."
            config["distillation"]["teacher_model_path"] = teacher_model_path.strip()
            config["distillation"]["teacher_type"] = teacher_model_type
        
        if "local_llm" not in config:
            config["local_llm"] = {}
        
        config["local_llm"]["enabled"] = use_local_llm
        if use_local_llm:
            config["local_llm"]["provider"] = local_llm_provider
            if local_llm_provider == "lm_studio" or local_llm_provider == "custom":
                config["local_llm"]["base_url"] = local_llm_url
            elif local_llm_provider == "ollama":
                config["local_llm"]["ollama_url"] = local_llm_url
        
        save_config(config)
        
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
        import traceback
        log_output.log(traceback.format_exc())
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
    
    # Load initial config
    config = load_config()
    dist_config = config.get("distillation", {})
    llm_config = config.get("local_llm", {})

    # Tkinter helper for file dialog
    def open_folder_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            folder_path = filedialog.askdirectory()
            root.destroy()
            return folder_path
        except Exception as e:
            return f"Error opening dialog: {str(e)}"

    custom_theme = create_theme()
    
    css = """
    body { background: radial-gradient(circle at 50% 0%, #1a1c2e 0%, #0f1016 100%) !important; }
    .gradio-container { background: transparent !important; }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 { 
        color: white !important; 
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Make the title huge and clean like the reference */
    h1 {
        font-size: 3.5rem !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    
    /* Make all text white */
    * {
        color: white !important;
    }
    
    /* Subtitles and prose text */
    .prose { color: white !important; }
    
    /* General text visibility fixes for dark mode */
    label, span, .gradio-markdown, .block-info, p {
        color: white !important;
    }
    
    /* Input fields and dropdowns */
    input, textarea {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Input fields styling */
    input, textarea {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Dropdown styling: match Examples table - dark background + white text */
    .gradio-dropdown,
    .gradio-portal,
    .gradio-popover,
    .gradio-modal,
    .gr-examples,
    .gr-samples {
        background: radial-gradient(circle at 50% 0%, #1a1c2e 0%, #0f1016 100%) !important;
        color: #fff !important;
    }

    /* Ensure visible button/trigger text */
    .gradio-dropdown button,
    .gradio-dropdown .gr-button,
    .gradio-dropdown .gr-selected,
    .gradio-dropdown .gr-input {
        background-color: #1a1c2e !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        padding: 8px 12px !important;
    }

    .gradio-dropdown input,
    .gradio-dropdown [role="combobox"],
    .gradio-dropdown [role="button"] {
        background-color: #1a1c2e !important;
        color: #fff !important;
    }

    /* Native select and options */
    select,
    .gradio-dropdown select {
        background-color: #1a1c2e !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        padding: 6px 10px !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }

    .gradio-portal [role="listbox"],
    .gradio-portal [role="option"],
    .gradio-dropdown [role="listbox"],
    .gradio-dropdown [role="option"] {
        background-color: #1a1c2e !important;
        color: #fff !important;
    }

    .gradio-portal [role="option"][aria-selected="true"],
    .gradio-portal [role="option"]:hover {
        background-color: rgba(255,255,255,0.06) !important;
        color: #fff !important;
    }

    option {
        background: #1a1c2e !important;
        color: #fff !important;
        padding: 6px 10px !important;
    }

    option:hover,
    option:focus,
    option:checked {
        background: rgba(255,255,255,0.03) !important;
        color: #fff !important;
    }

    /* Popup lists (Gradio sometimes places menus in portals) */
    .gradio-portal *,
    .gradio-popover * {
        background: #1a1c2e !important;
        color: #fff !important;
    }

    /* Keep high z-index so popups are above other layers */
    .gradio-portal,
    .gradio-popover {
        z-index: 9999 !important;
    }
    
    /* Checkbox and Radio labels */
    .gradio-checkbox label, .gradio-radio label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Fix for file explorer/browser buttons if needed */
    button.secondary {
        color: white !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Ensure info text is white */
    .gr-info-text, .info-text, [class*="info"] {
        color: white !important;
    }
    
    /* Ensure all block labels and descriptions are white */
    .block-label, .block-info, .gradio-label-small {
        color: white !important;
    }
    
    /* Examples section styling - dark background */
    .gr-examples, .gr-samples, .examples {
        background-color: #0f1016 !important;
    }
    
    /* Table styling for examples */
    table, tbody, thead, tr, td, th {
        background-color: #1a1c2e !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    th {
        background-color: #0f1016 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Red text for specific buttons and options */
    /* Target Browse System buttons */
    button {
        font-size: 0.875rem;
    }
    
    /* Make button text red for Browse and Detect buttons */
    .gr-button:nth-of-type(1),
    .gr-button:nth-of-type(2) {
        color: red !important;
    }
    
    /* Target all buttons and make specific ones red */
    button[type="button"] {
        transition: color 0.3s;
    }
    
    /* Use CSS to style buttons with specific content - this is a workaround */
    /* Red text for distillation mode options */
    .gradio-radio {
        display: flex;
        gap: 1rem;
    }
    
    .gradio-radio input[type="radio"] {
        accent-color: cyan;
    }
    
    .gradio-radio input[type="radio"] ~ label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Target button text more aggressively */
    button span, button div {
        color: red !important;
    }
    
    /* Target all button content */
    button {
        color: red !important;
    }
    
    button * {
        color: red !important;
    }
    
    /* Specific styling for red buttons */
    #browse_student_btn,
    #browse_teacher_btn,
    #detect_llm_btn {
        background-color: #1a1c2e !important;
        border: 2px solid red !important;
        color: red !important;
    }
    
    #browse_student_btn button,
    #browse_teacher_btn button,
    #detect_llm_btn button,
    #browse_student_btn *,
    #browse_teacher_btn *,
    #detect_llm_btn * {
        color: red !important;
        background-color: transparent !important;
    }
    
    /* Distillation mode radio options styling */
    #distillation_mode_radio {
        display: flex;
        gap: 1rem;
    }
    
    #distillation_mode_radio label {
        color: white !important;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background-color: #1a1c2e !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
    }

    #distillation_mode_radio label * {
        color: white !important;
    }

    #distillation_mode_radio input[type="radio"]:checked + label {
        background-color: rgba(255,255,255,0.06) !important;
    }

    /* Dropdown overrides (must be last to beat aggressive button text rules) */
    .gr-dropdown,
    .gr-dropdown > div,
    .gr-dropdown .wrap,
    .gr-dropdown .input-container,
    .gr-dropdown .single-value,
    .gr-dropdown .placeholder,
    .gr-dropdown input,
    .gr-dropdown [role="combobox"],
    [data-testid="dropdown"],
    [data-testid="dropdown"] * {
        background-color: #1a1c2e !important;
        color: #fff !important;
        border-color: rgba(255,255,255,0.12) !important;
    }

    .gr-dropdown svg,
    .gr-dropdown svg *,
    [data-testid="dropdown"] svg,
    [data-testid="dropdown"] svg * {
        color: #fff !important;
        fill: #fff !important;
    }

    .gradio-portal [role="listbox"],
    .gradio-portal [role="option"],
    [role="listbox"],
    [role="option"] {
        background-color: #1a1c2e !important;
        color: #fff !important;
    }

    [role="option"][aria-selected="true"],
    [role="option"]:hover {
        background-color: rgba(255,255,255,0.06) !important;
        color: #fff !important;
    }
    """
    
    # JavaScript to ensure dropdown/popover text is visible even when Gradio creates white popups
    js_code = """
    <script>
    function getEffectiveBackgroundColor(el) {
        try {
            let node = el;
            while (node && node !== document.documentElement) {
                const cs = window.getComputedStyle(node);
                if (!cs) break;
                const bg = cs.backgroundColor;
                if (bg && bg !== 'transparent' && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'hsla(0, 0%, 0%, 0)') {
                    return bg;
                }
                node = node.parentElement;
            }
        } catch (e) {}
        return null;
    }

    function applyImportantStyle(el, prop, val) {
        try { el.style.setProperty(prop, val, 'important'); } catch(e) {}
    }

    function colorSpecificElements() {
        try {
            // Force known Gradio popover/portal containers to dark background + red text
            document.querySelectorAll('.gradio-portal, .gradio-popover, .gradio-modal, .gr-examples, .gr-samples').forEach(el => {
                applyImportantStyle(el, 'background-color', '#1a1c2e');
                applyImportantStyle(el, 'color', 'white');
                el.querySelectorAll('*').forEach(c => {
                    applyImportantStyle(c, 'color', 'white');
                });
            });

            // For any element whose effective background is white, recolor background and text
            document.querySelectorAll('body *').forEach(el => {
                const bg = getEffectiveBackgroundColor(el);
                if (!bg) return;
                if (bg === 'rgb(255, 255, 255)' || bg === 'rgba(255, 255, 255, 1)' || bg.toLowerCase() === '#ffffff' || bg.toLowerCase() === 'white') {
                    applyImportantStyle(el, 'background-color', '#1a1c2e');
                    applyImportantStyle(el, 'color', 'white');
                    // apply to children
                    el.querySelectorAll('*').forEach(c => {
                        applyImportantStyle(c, 'color', 'white');
                        const childBg = getEffectiveBackgroundColor(c);
                        if (childBg && (childBg === 'rgb(255, 255, 255)' || childBg === 'rgba(255, 255, 255, 1)')) {
                            applyImportantStyle(c, 'background-color', '#1a1c2e');
                        }
                    });
                }
            });
        } catch (e) {
            console.warn('colorSpecificElements error', e);
        }
    }

    window.addEventListener('load', colorSpecificElements);
    setTimeout(colorSpecificElements, 200);
    setTimeout(colorSpecificElements, 600);
    setTimeout(colorSpecificElements, 1200);
    setTimeout(colorSpecificElements, 2000);

    const observer = new MutationObserver(() => colorSpecificElements());
    observer.observe(document.body, { childList: true, subtree: true });

    window.fixGradioColors = colorSpecificElements;
    </script>
    """

    with gr.Blocks(theme=custom_theme, title=GRADIO_TITLE, css=css) as demo:
        gr.Markdown(f"# {GRADIO_TITLE}")
        gr.Markdown(GRADIO_DESCRIPTION)
        
        # Add custom JavaScript for button and radio coloring
        gr.HTML(js_code)
        
        # Device information panel
        with gr.Row():
            device_info = gr.Markdown(get_device_info())
        
        # 1. Input Model Selection Section
        gr.Markdown("## Input Model Selection (Random/Student Model)")
        with gr.Row():
            with gr.Column(scale=3):
                model_path = gr.Textbox(
                    label="Input Model Path (Target for Processing)",
                    placeholder="Select a folder path...",
                    info="Path to the model you want to quantize/distill (e.g. HuggingFace folder)"
                )
            with gr.Column(scale=1):
                browse_student_btn = gr.Button("📂 Browse System", variant="secondary", elem_id="browse_student_btn")
            
            with gr.Column(scale=2):
                model_type = gr.Dropdown(
                    choices=MODEL_TYPES,
                    value=DEFAULT_MODEL_TYPE,
                    label="Input Model Library/Type",
                    info="Select the library or type of the model"
                )

        # Wire browse button
        browse_student_btn.click(
            fn=open_folder_dialog,
            outputs=model_path
        )
        
        # 2. Knowledge Distillation Configuration Section
        gr.Markdown("## Knowledge Distillation Configuration")
        with gr.Row():
            distillation_mode = gr.Radio(
                choices=["placeholder", "teacher_student"],
                value=dist_config.get("mode", "placeholder"),
                label="Distillation Mode",
                info="Placeholder: No training | Teacher-Student: Real KD training",
                elem_id="distillation_mode_radio"
            )
        
        with gr.Row(visible=True) as teacher_row:
            with gr.Column(scale=3):
                teacher_model_path = gr.Textbox(
                    label="Teacher Model Path",
                    placeholder="Select a folder path...",
                    value=dist_config.get("teacher_model_path", ""),
                    info="Required for teacher-student mode"
                )
            with gr.Column(scale=1):
                browse_teacher_btn = gr.Button("📂 Browse System", variant="secondary", elem_id="browse_teacher_btn")

            with gr.Column(scale=2):
                teacher_model_type = gr.Dropdown(
                    choices=MODEL_TYPES,
                    value=dist_config.get("teacher_type", DEFAULT_MODEL_TYPE),
                    label="Teacher Model Type"
                )
        
        # Wire teacher browse button
        browse_teacher_btn.click(
            fn=open_folder_dialog,
            outputs=teacher_model_path
        )

        # Show/hide teacher inputs based on mode
        def toggle_teacher_visibility(mode):
            return gr.update(visible=(mode == "teacher_student"))
        
        distillation_mode.change(
            fn=toggle_teacher_visibility,
            inputs=[distillation_mode],
            outputs=[teacher_row]
        )
        
        # 3. Local LLM Configuration Section
        gr.Markdown("## Local LLM Integration (Optional)")
        with gr.Row():
            use_local_llm = gr.Checkbox(
                label="Use Local LLM",
                value=llm_config.get("enabled", False),
                info="Connect to LM Studio, Ollama, or other local LLM servers"
            )
        
        with gr.Row(visible=llm_config.get("enabled", False)) as llm_row:
            with gr.Column():
                local_llm_provider = gr.Dropdown(
                    choices=["auto", "lm_studio", "ollama", "custom"],
                    value=llm_config.get("provider", "auto"),
                    label="LLM Provider",
                    info="Auto-detect or specify provider"
                )
                
                local_llm_url = gr.Textbox(
                    label="LLM Server URL",
                    value=llm_config.get("base_url", "http://localhost:1234/v1"),
                    info="Base URL for LM Studio (port 1234) or Ollama (port 11434)"
                )
        
        # Show/hide LLM inputs based on checkbox
        def toggle_llm_visibility(enabled):
            return gr.update(visible=enabled)
        
        use_local_llm.change(
            fn=toggle_llm_visibility,
            inputs=[use_local_llm],
            outputs=[llm_row]
        )
        
        # Detect button for local LLM
        with gr.Row():
            detect_llm_btn = gr.Button("Detect Local LLM", variant="secondary", elem_id="detect_llm_btn")
            llm_status = gr.Markdown("")
        
        def detect_llm():
            provider, models = detect_local_llm()
            return f"**Detected Provider:** {provider}\n**Available Models:** {models}"
        
        detect_llm_btn.click(fn=detect_llm, outputs=[llm_status])
        
        # 4. Quantization Configuration Section
        gr.Markdown("## Quantization Configuration")
        with gr.Row():
            quant_type = gr.Dropdown(
                choices=QUANT_TYPES,
                value=DEFAULT_QUANT_TYPE,
                label="Quantization Type",
                info="INT4 (weight-only), INT8 (dynamic), or FP8"
            )
        
        # Run Button
        with gr.Row():
            run_btn = gr.Button(
                "Run Distill + Quantize",
                variant="primary",
                size="lg"
            )
        
        # Log Output
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
            inputs=[
                model_path,
                model_type,
                quant_type,
                distillation_mode,
                teacher_model_path,
                teacher_model_type,
                use_local_llm,
                local_llm_provider,
                local_llm_url
            ],
            outputs=[log_output]
        ).then(
            fn=lambda: get_device_info(),
            outputs=[device_info]
        )
        
        # Example section
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                ["microsoft/phi-2", "HuggingFace Transformers (NLP/Vision/Audio)", "INT8 (dynamic)", "placeholder", "", "HuggingFace Transformers (NLP/Vision/Audio)"],
                ["meta-llama/Llama-2-7b-hf", "HuggingFace Transformers (NLP/Vision/Audio)", "INT4 (weight-only)", "teacher_student", "meta-llama/Llama-2-13b-hf", "HuggingFace Transformers (NLP/Vision/Audio)"],
            ],
            inputs=[model_path, model_type, quant_type, distillation_mode, teacher_model_path, teacher_model_type],
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
