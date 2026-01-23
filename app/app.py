"""
Main entry point for the application (Pinokio compatible).
"""
from gradio_ui import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
