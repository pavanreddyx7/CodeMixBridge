"""
Gradio web application for CodeMixBridge.
Features a responsive dark navy and cyan UI with dual side-by-side model panels
(Generic Gemma vs Fine-Tuned Gemma) and an actual benchmark viewer.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import gradio as gr
import pandas as pd

from src.inference import generate_dual_comparison
from src.prompts import (
    ALL_TASKS,
    LABEL_TO_TASK,
    TASK_LABELS,
    TASK_EN_TO_ROMANIZED,
    TASK_EN_TO_CODEMIXED,
    TASK_EN_TO_NATIVE,
    TASK_ROMANIZED_TO_EN,
    TASK_CODEMIXED_TO_EN,
    build_user_prompt,
)


# Curated realistic examples representing college announcements, student chat, and daily life
DEMO_EXAMPLES = [
    [
        "Our internal lab exams start tomorrow at 9 AM, please arrive on time.",
        TASK_LABELS[TASK_EN_TO_ROMANIZED],
    ],
    [
        "I am submitting the assignment in the computer lab today.",
        TASK_LABELS[TASK_EN_TO_CODEMIXED],
    ],
    [
        "The college library will remain closed on Sunday for maintenance work.",
        TASK_LABELS[TASK_EN_TO_NATIVE],
    ],
    [
        "Naale exam mugida mele cafeteria ge hogona, nanna jothe barthiya?",
        TASK_LABELS[TASK_ROMANIZED_TO_EN],
    ],
    [
        "ದಯವಿಟ್ಟು ಈ document ಅನ್ನು review ಮಾಡಿ feedback ಕಳಿಸಿ.",
        TASK_LABELS[TASK_CODEMIXED_TO_EN],
    ],
]

CUSTOM_CSS = """
/* Deep Navy & Vibrant Cyan Theme */
:root {
    --bg-primary: #070B19;
    --bg-surface: #0E162E;
    --bg-card: #152244;
    --border-color: #1F3260;
    --accent-cyan: #00F0FF;
    --accent-teal: #00B4D8;
    --text-primary: #F0F4FC;
    --text-secondary: #9DB2DB;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.card-container {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.app-header {
    text-align: center;
    padding: 24px 0 16px 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 24px;
}

.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00F0FF 0%, #00B4D8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.app-tagline {
    font-size: 1.05rem;
    color: var(--text-secondary);
}

.panel-header-generic {
    font-weight: 700;
    color: #9DB2DB;
    border-bottom: 2px solid #2A3F70;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

.panel-header-finetuned {
    font-weight: 700;
    color: #00F0FF;
    border-bottom: 2px solid #00F0FF;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

.btn-primary {
    background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #00F0FF 0%, #00B4D8 100%) !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
}
"""


def load_metrics_table(metrics_csv_path: str = "results/metrics.csv") -> pd.DataFrame:
    """
    Safely loads benchmark metrics from CSV or returns a structured pending table.
    """
    csv_file = Path(metrics_csv_path)
    if csv_file.exists():
        try:
            return pd.read_csv(csv_file)
        except Exception:
            pass

    # Default schema if evaluations are pending
    return pd.DataFrame({
        "Model": ["Generic Gemma (Base)", "Fine-Tuned Gemma (LoRA)"],
        "Romanized chrF": ["Pending", "Pending"],
        "Code-mixed chrF": ["Pending", "Pending"],
        "Native chrF": ["Pending", "Pending"],
        "Overall chrF": ["Pending", "Pending"],
        "Median Latency (s)": ["Pending", "Pending"],
        "Status": ["Awaiting Colab Run", "Awaiting Colab Run"],
    })


def create_gradio_app(model: Any = None, tokenizer: Any = None) -> gr.Blocks:
    """
    Builds the complete Gradio interface for CodeMixBridge.
    If model is None (local CPU preview), demonstrates input validation and UI preview safely.
    """
    def on_convert(input_text: str, task_label: str) -> Tuple[str, str, str, str]:
        if not input_text or not input_text.strip():
            raise gr.Error("Please enter a text message to convert.")

        task_key = LABEL_TO_TASK.get(task_label, TASK_EN_TO_ROMANIZED)
        full_prompt = build_user_prompt(task_key, input_text)

        if model is None or tokenizer is None:
            # Fallback message when loaded in local environment without weights
            generic_out = "(Model weights not loaded locally. Run in Colab with T4 GPU to generate live output.)"
            tuned_out = "(Model weights not loaded locally. Run in Colab with T4 GPU to generate live output.)"
            gen_time = "0.000s"
            tuned_time = "0.000s"
            return generic_out, tuned_out, gen_time, tuned_time

        # Run dual comparison
        results = generate_dual_comparison(
            model=model,
            tokenizer=tokenizer,
            prompt=full_prompt,
            max_new_tokens=128,
        )

        generic_pred = results["generic"]["prediction"]
        generic_lat = f"{results['generic']['latency_sec']}s"
        finetuned_pred = results["finetuned"]["prediction"]
        finetuned_lat = f"{results['finetuned']['latency_sec']}s"

        return generic_pred, finetuned_pred, generic_lat, finetuned_lat

    with gr.Blocks(title="CodeMixBridge — Kannada-English AI", css=CUSTOM_CSS) as demo:
        # Header
        gr.HTML(
            """
            <div class="app-header">
                <div class="app-title">CodeMixBridge</div>
                <div class="app-tagline">Bridging Kannada and English through AI — Powered by Google Gemma & LoRA</div>
            </div>
            """
        )

        with gr.Tabs():
            # Tab 1: Live Interactive Conversion
            with gr.TabItem("🔄 Live Model Comparison"):
                with gr.Row():
                    # Left column: Controls & Input
                    with gr.Column(scale=4):
                        with gr.Group():
                            task_selector = gr.Dropdown(
                                choices=list(TASK_LABELS.values()),
                                value=TASK_LABELS[TASK_EN_TO_ROMANIZED],
                                label="🎯 Select Conversion Task",
                                interactive=True,
                            )
                            input_box = gr.Textbox(
                                label="📝 Input Message",
                                placeholder="Type or paste your message here...",
                                lines=4,
                            )
                            convert_btn = gr.Button(
                                "🚀 Convert (Run Both Models)",
                                variant="primary",
                                elem_classes=["btn-primary"],
                            )

                    # Right column: Side-by-Side Outputs
                    with gr.Column(scale=6):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML("<div class='panel-header-generic'>MODEL A: Generic Gemma (Base)</div>")
                                generic_output = gr.Textbox(
                                    label="Output",
                                    lines=5,
                                    interactive=False,
                                    show_copy_button=True,
                                )
                                generic_meta = gr.Markdown("⏱ Latency: —")

                            with gr.Column(scale=1):
                                gr.HTML("<div class='panel-header-finetuned'>MODEL B: Fine-Tuned Gemma (LoRA)</div>")
                                finetuned_output = gr.Textbox(
                                    label="Output",
                                    lines=5,
                                    interactive=False,
                                    show_copy_button=True,
                                )
                                finetuned_meta = gr.Markdown("⏱ Latency: —")

                # Examples Section
                gr.Markdown("### 💡 Clickable Examples")
                gr.Examples(
                    examples=DEMO_EXAMPLES,
                    inputs=[input_box, task_selector],
                    label="Curated Demonstration Scenarios",
                )

                # Event Binding
                convert_btn.click(
                    fn=on_convert,
                    inputs=[input_box, task_selector],
                    outputs=[generic_output, finetuned_output, generic_meta, finetuned_meta],
                )

            # Tab 2: Actual Benchmark Results
            with gr.TabItem("📊 Benchmark & Evaluation"):
                gr.Markdown("### 🔬 Test Set Evaluation Metrics (SacreBLEU chrF)")
                gr.Markdown(
                    "Quantitative evaluation conducted on the held-out test split (15% unseen Kannada records). "
                    "Metrics are loaded directly from `results/metrics.csv`."
                )
                metrics_df = load_metrics_table()
                gr.Dataframe(value=metrics_df, interactive=False)

                gr.Markdown(
                    """
                    **Key Metric Highlights:**
                    - **chrF (Character n-gram F-score):** Primary metric per hackathon specification. Correctly credits informal romanized transliteration variations.
                    - **Leakage-Free Partitioning:** Held-out test set contains zero overlapping English source statements with the training data.
                    - **Fair Benchmark Protocol:** Both Generic and Fine-Tuned Gemma receive identical prompts and deterministic decoding settings (`do_sample=False`).
                    """
                )

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False)
