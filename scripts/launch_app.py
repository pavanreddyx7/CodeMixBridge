"""
Application launcher script for CodeMixBridge.
Initializes model (attaching LoRA adapter if present) and serves the Gradio interface.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import create_gradio_app
from src.model_loader import load_gemma_model, verify_gpu_environment


def main():
    print("=" * 70)
    print("CodeMixBridge — Application Launcher")
    print("=" * 70)

    has_gpu, gpu_msg = verify_gpu_environment()
    print(gpu_msg)

    model = None
    tokenizer = None

    adapter_path = "artifacts/adapters/gemma_4_kannada_lora"
    if has_gpu:
        try:
            print("Loading Gemma 4 E2B model...")
            model, tokenizer = load_gemma_model(
                base_model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
                adapter_path=adapter_path if Path(adapter_path).exists() else None,
                load_in_4bit=True,
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"[Notice] Could not load model into VRAM: {e}")
            print("Starting UI in preview/mock mode.")
    else:
        print("[Notice] GPU not detected. Starting UI in interface preview mode.")

    app = create_gradio_app(model=model, tokenizer=tokenizer)
    # Launch with public share link enabled in Colab or local preview
    in_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
    app.launch(share=in_colab, inbrowser=True)


if __name__ == "__main__":
    main()
