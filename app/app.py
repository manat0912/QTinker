"""
Main entry point for the application (Pinokio compatible).
"""
import os
import glob
import sys

# Add TensorRT to PATH dynamically
# Search for TensorRT* directory in the current working directory
tensorrt_paths = glob.glob(os.path.join(os.getcwd(), "TensorRT*"))
if tensorrt_paths:
    tensorrt_lib_path = os.path.join(tensorrt_paths[0], "lib")
    if os.path.isdir(tensorrt_lib_path):
        print(f"Found TensorRT lib path: {tensorrt_lib_path}")
        os.environ["PATH"] = tensorrt_lib_path + os.pathsep + os.environ["PATH"]
        # Also add to DLL search path for Python 3.8+ on Windows
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(tensorrt_lib_path)
            except Exception as e:
                print(f"Failed to add DLL directory: {e}")
    else:
        print(f"Warning: TensorRT lib directory not found in {tensorrt_paths[0]}")
else:
    print("Warning: TensorRT directory not found in current path.")

from gradio_ui import create_ui, custom_theme, css

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=custom_theme, css=css)
