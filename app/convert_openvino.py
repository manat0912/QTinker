
import os
try:
    from openvino import convert_model
except ImportError:
    try:
        # Fallback for older versions
        from openvino.tools.mo import convert_model
    except ImportError:
        convert_model = None

def convert_openvino(model_path, output_path, log_fn):
    """
    Converts a model to OpenVINO IR format.
    """
    if convert_model is None:
        raise ImportError("OpenVINO is not installed or convert_model could not be imported.")

    if not model_path:
        raise ValueError("Model path cannot be empty.")
        
    log_fn(f"Converting model to OpenVINO IR format: {model_path}")

    ov_model = convert_model(model_path)

    if not output_path:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}_openvino.xml")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # The output path for OpenVINO is just the base name, .xml and .bin are added
    base_output_path = os.path.splitext(output_path)[0]

    log_fn(f"Saving OpenVINO model to: {base_output_path}.xml/.bin")
    ov_model.save_model(base_output_path + ".xml")
    log_fn("OpenVINO conversion successful.")
    
    # Return path to the .xml file for consistency
    return base_output_path + ".xml"
