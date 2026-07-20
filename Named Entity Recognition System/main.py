import os
import time
import json
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, AutoTokenizer


PROJECT_NAME = "Named Entity Recognition System" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model")
MAX_LENGTH = 128

# Load the Model
tokenizer = None
model = None
id2label = {}
label_names = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(MODEL_PATH):
    try:
        print(" Loading the NER model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        
        with open(os.path.join(MODEL_PATH, "label_mapping.json"), "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
            id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            label_names = label_mapping["label_names"]
            
        model.to(device)
        model.eval()
        print(" SUCCESS: The NER model has been loaded perfectly!")

    except Exception as e:
        print(f" ERROR: Found the folder, but model failed to load.\nDetails: {e}\n")
else:
    print("️ WARNING: Model folder 'best_model' not found. Please place it in the directory.")


# Helper Functions for UI & Plots
def create_entity_plot(entity_counts):
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    fig.patch.set_facecolor('#0B1021')
    ax.set_facecolor('#0B1021')
    
    if not entity_counts:
        ax.text(0.5, 0.5, "No entities detected to analyze", 
                ha='center', va='center', fontsize=13, color='#E2E8F0', fontweight='600')
        ax.axis('off')
        return fig
    
    categories = list(entity_counts.keys())
    counts = list(entity_counts.values())
    
    colors = ['#A855F7', '#3B82F6', '#10B981', '#F472B6', '#8B5CF6', '#F97316']
    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
    
    bars = ax.bar(categories, counts, color=bar_colors, width=0.5, zorder=2, edgecolor='#1E293B')
    
    ax.set_ylabel('Count', fontsize=11, color='#E2E8F0', fontweight='600')
    ax.set_title('Entity Distribution Analysis', fontsize=13, color='#E2E8F0', fontweight='700', pad=15)
    ax.set_ylim(0, max(counts) + 1 if max(counts) > 0 else 1)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, 
                    color='#E2E8F0', fontweight='700')
                    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#3B82F6')
    ax.spines['bottom'].set_color('#3B82F6')
    ax.tick_params(axis='both', colors='#E2E8F0', labelsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.2, zorder=1, color='#1E293B')
    
    plt.tight_layout()
    return fig

def generate_metrics_html(entities_list, processing_time, entity_counts):
    total_detected = len(entities_list)
    entity_types = len(entity_counts)
    
    metrics_html = f"""
    <div style="display: flex; gap: 15px; width: 100%; font-family: system-ui, sans-serif; margin-top: 20px;" class="notranslate">
        <div style="flex: 1; background: #0F1435 !important; border: 1px solid #3B82F6; border-radius: 12px; padding: 18px 20px; text-align: center;">
            <span style="font-size: 0.75rem; color: #94A3B8; display: block; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">Total Entities</span>
            <strong style="font-size: 1.8rem; color: #E2E8F0; display: block; font-weight: 800;">{total_detected}</strong>
        </div>
        
        <div style="flex: 1; background: #0F1435 !important; border: 1px solid #A855F7; border-radius: 12px; padding: 18px 20px; text-align: center;">
            <span style="font-size: 0.75rem; color: #94A3B8; display: block; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">Entity Types</span>
            <strong style="font-size: 1.8rem; color: #E2E8F0; display: block; font-weight: 800;">{entity_types}</strong>
        </div>
        
        <div style="flex: 1; background: #0F1435 !important; border: 1px solid #10B981; border-radius: 12px; padding: 18px 20px; text-align: center;">
            <span style="font-size: 0.75rem; color: #94A3B8; display: block; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">Process Speed</span>
            <strong style="font-size: 1.8rem; color: #E2E8F0; display: block; font-weight: 800;">{processing_time:.3f}s</strong>
        </div>
    </div>
    """
    return metrics_html

def generate_breakdown_html(entity_counts):
    if not entity_counts:
        return """
        <div style="background: #0F1435 !important; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; font-family: system-ui, sans-serif; height: 100%; box-sizing: border-box;" class="notranslate">
            <h4 style="margin-top: 0; margin-bottom: 15px; font-size: 1rem; color: #E2E8F0; font-weight: 700; border-bottom: 2px solid #3B82F6; padding-bottom: 8px;">
                 Entity Breakdown
            </h4>
            <p style="margin: 0; color: #64748B; font-size: 0.9rem; text-align: center; padding: 20px 0;">Waiting for text analysis...</p>
        </div>
        """
        
    breakdown_items = ""
    for entity_type, count in entity_counts.items():
        breakdown_items += f"""
        <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #1E293B;">
            <span style="font-weight: 700; color: #E2E8F0; font-size: 0.9rem;">{entity_type}</span>
            <span style="font-weight: 800; color: #60A5FA; font-size: 0.9rem;">{count}</span>
        </div>
        """

    return f"""
    <div style="background: #0F1435 !important; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; font-family: system-ui, sans-serif; height: 100%; box-sizing: border-box;" class="notranslate">
        <h4 style="margin-top: 0; margin-bottom: 15px; font-size: 1rem; color: #E2E8F0; font-weight: 700; border-bottom: 2px solid #3B82F6; padding-bottom: 8px;">
             Entity Breakdown
        </h4>
        {breakdown_items}
    </div>
    """

def generate_summary_html(entities_list):
    total = len(entities_list)
    
    if not entities_list:
        return """
        <div style="background: #0F1435 !important; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; font-family: system-ui, sans-serif; height: 100%; box-sizing: border-box;" class="notranslate">
            <h4 style="margin: 0 0 15px 0; font-size: 1rem; color: #E2E8F0; font-weight: 700;">📋 Detected Entities</h4>
            <p style="margin: 0; color: #64748B; font-size: 0.9rem; text-align: center;">⚠️ No entities detected</p>
        </div>
        """
    
    badges_html = ""
    for entity in entities_list:
        badges_html += f"""
        <div style="
            display: inline-flex;
            background: linear-gradient(135deg, #60A5FA, #93C5FD);
            color: #0B1021;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
            box-shadow: 0 2px 6px rgba(96, 165, 250, 0.2);
            width: fit-content;
            max-width: 100%;
            align-items: center;
            transition: transform 0.2s ease;
        " onmouseover="this.style.transform='translateX(4px)'" onmouseout="this.style.transform='translateX(0)'">
            <span style="white-space: nowrap;">{entity}</span>
        </div>
        """
    
    return f"""
    <div style="background: #0F1435 !important; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; font-family: system-ui, sans-serif; height: 100%; box-sizing: border-box;" class="notranslate">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; font-size: 1rem; color: #E2E8F0; font-weight: 700;">📋 Detected Entities</h4>
            <div style="
                background: #10B981;
                color: #FFFFFF;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 700;
            ">
                {total} Found
            </div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-start; max-height: 400px; overflow-y: auto;">
            {badges_html}
        </div>
    </div>
    """

# Core Prediction Logic
def predict_ner(text):
    if model is None:
        return [], generate_summary_html([]), generate_metrics_html([], 0.0, {}), generate_breakdown_html({}), None
        
    if not text or not text.strip():
        return [], generate_summary_html([]), generate_metrics_html([], 0.0, {}), generate_breakdown_html({}), None
    
    start_time = time.time()
    try:
        words = text.split()
        encoding = tokenizer(
            words, is_split_into_words=True, return_tensors="pt",
            truncation=True, padding="max_length", max_length=MAX_LENGTH
        )
        
        word_ids = encoding.word_ids()
        inputs = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)[0]
        
        word_to_label = {}
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_to_label:
                word_to_label[word_idx] = id2label.get(predictions[i], "O")
        
        highlighted_text = []
        entities_list = []
        entity_counts = {}
        
        for idx, word in enumerate(words):
            label = word_to_label.get(idx, "O")
            if label != "O":
                entity_type = label.split("-")[-1]
                highlighted_text.append((word, entity_type))
                entities_list.append(f"{word} → {entity_type}")
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            else:
                highlighted_text.append((word, None))
        
        processing_time = time.time() - start_time
        summary_html = generate_summary_html(entities_list)
        
        metrics_html = generate_metrics_html(entities_list, processing_time, entity_counts)
        breakdown_html = generate_breakdown_html(entity_counts)
        plot_fig = create_entity_plot(entity_counts)
        
        return highlighted_text, summary_html, metrics_html, breakdown_html, plot_fig
    
    except Exception as e:
        return [("Error", "Error")], generate_summary_html([]), generate_metrics_html([], 0.0, {}), generate_breakdown_html({}), None

# Custom CSS 
custom_css = """
@keyframes floatAnimation {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

body, .gradio-container {
    background: #0B1021 !important; 
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    color: #E2E8F0 !important; 
}

/* Input Box Container */
.input-box-container {
    background: #0F1435 !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 16px !important;
    padding: 30px !important;
    width: 100% !important;
    margin: 0 auto 20px auto;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    position: relative !important;
    z-index: 1 !important;
}

#main_text_input,
#main_text_input textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #E2E8F0 !important;
    font-size: 1.05rem !important;
    padding: 16px !important;
    min-height: 120px !important;
    resize: vertical !important;
    width: 100% !important;
    position: relative !important;
    z-index: 50 !important;
    pointer-events: auto !important;
    cursor: text !important;
}

#main_text_input textarea::placeholder {
    color: #64748B !important;
}

#main_text_input textarea:focus {
    outline: none !important;
    border-color: #3B82F6 !important;
}

/* Predict Button Row */
.predict-btn-row {
    display: flex !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin-top: -10px !important;
    margin-bottom: 25px !important;
    padding-left: 30px !important;
}

/* Predict Button */
.btn-predict {
    background: linear-gradient(to right, #9333EA, #3B82F6) !important; 
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px 28px !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    width: auto !important;
    max-width: 250px !important;
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.btn-predict:hover {
    background: linear-gradient(to right, #A855F7, #60A5FA) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4) !important;
}

/* Workspace Containers */
.workspace-container {
    background: #0F1435 !important;
    border-radius: 18px !important;
    border: 1px solid #1E293B !important;
    padding: 30px !important;
    margin: 0 auto 40px auto;
    width: 100% !important;
    max-width: 100% !important;
}

/* Highlighted Text Output */
#highlighted_output_box {
    background: #0B1021 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 20px !important;
    min-height: 100px !important;
    color: #E2E8F0 !important;
}

#highlighted_output_box span,
#highlighted_output_box .token {
    color: #E2E8F0 !important;
}

#highlighted_output_box .wrap, #highlighted_output_box .contain {
    background: transparent !important;
    border: none !important;
}

/* Plot Box - Enlarged */
#entity_plot_box > div {
    background-color: #0F1435 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    min-height: 400px !important;
}

/* Hero Banner */
.hero-banner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: 50px auto 30px auto;
    max-width: 800px;
}

.hero-banner h1 {
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    background: linear-gradient(to right, #A855F7, #3B82F6); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px !important;
    letter-spacing: -1px;
    display: inline-block;
    animation: floatAnimation 3.5s ease-in-out infinite !important;
}

.hero-banner p {
    font-size: 0.95rem !important;
    color: #94A3B8 !important;
    max-width: 600px;
    line-height: 1.5;
    text-align: center !important;
    margin: 0 auto !important;
}

.charts-divider-header {
    border-top: 1px solid #1E293B;
    margin-top: 35px !important;
    padding-top: 30px;
    margin-bottom: 20px;
}

.charts-divider-header h3 {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #E2E8F0 !important; 
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.gradio-container, .gradio-container * {
    direction: ltr !important;
    text-align: left;
}
footer {
    display: none !important;
}
"""

head_html = """
<meta name="google" content="notranslate">
<script>
    document.addEventListener("DOMContentLoaded", function() {
        document.documentElement.lang = "en";
        document.documentElement.dir = "ltr";
        document.body.classList.add("notranslate");
    });
</script>
"""

# UI Definition
with gr.Blocks(title=PROJECT_NAME, css=custom_css, head=head_html) as demo:
    
    # Hero Header
    gr.HTML(
        f"""
        <div class="hero-banner notranslate">
            <h1>{PROJECT_NAME}</h1>
            <p>Your intelligent NLP companion for extracting structured information (Persons, Organizations, Locations, Dates) from unstructured text.</p>
        </div>
        """
    )

    # Input Box
    with gr.Column(elem_classes="input-box-container"):
        text_input = gr.Textbox(
            placeholder="e.g., Ali was studying in China at 2025 and working in Google...",
            lines=6,
            elem_id="main_text_input",
            show_label=False,
            container=False,
            interactive=True
        )

    # Predict Button
    with gr.Row(elem_classes="predict-btn-row"):
        predict_btn = gr.Button("🚀 Predict Entities", elem_classes="btn-predict")

    # Results Section
    with gr.Column(elem_classes="workspace-container"):
        gr.HTML('<h3 style="font-size: 1.25rem; font-weight:800; color:#E2E8F0; margin-bottom: 4px;" class="notranslate">Analysis Workspace</h3>')
        gr.HTML('<p style="font-size: 0.9rem; color:#94A3B8; margin-bottom: 25px;" class="notranslate">Review the highlighted entities and statistical breakdown below.</p>')

        # Highlighted Results 
        gr.HTML("<h4 style='margin-bottom: 8px; color: #E2E8F0; font-weight: 700;'> Highlighted Results</h4>")
        highlighted_output = gr.HighlightedText(
            label="",
            color_map={
                "PER": "#FBBF24", "ORG": "#60A5FA", "LOC": "#34D399", "MISC": "#FB923C"
            },
            show_legend=True,
            elem_id="highlighted_output_box"
        )

        gr.HTML('<div style="height: 20px;"></div>')

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                summary_output = gr.HTML(
                    value=generate_summary_html([]),
                    elem_id="summary-output-box"
                )
            with gr.Column(scale=1, min_width=300):
                breakdown_output = gr.HTML(
                    value=generate_breakdown_html({}),
                    elem_id="breakdown-result-box"
                )

        # 3 Metrics Boxes
        metrics_output = gr.HTML(
            value=generate_metrics_html([], 0.0, {}),
            elem_id="metrics-result-box"
        )

        gr.HTML(
            """
            <div class="charts-divider-header notranslate">
                <h3>📈 Analytics & Technical Reports</h3>
            </div>
            """
        )

        # Single Enlarged Plot
        entity_plot_output = gr.Plot(label="Entity Distribution", elem_id="entity_plot_box")

    # Footer & Disclaimer
    gr.HTML(
        """
        <div style="margin: 60px auto 0 auto; max-width: 100%; font-family: system-ui, sans-serif; text-align: center;" class="notranslate">
            <hr style="border: 0; border-top: 1.5px solid #1E293B; margin-bottom: 24px; width: 100%;">
            
            <p style="margin: 0; font-size: 1.15rem; color: #E2E8F0; font-weight: 700; text-align: center; width: 100%;">
                 Named Entity Recognition System — Powered by Transformers & Deep Learning
            </p>
            <p style="font-size: 0.85rem; margin-top: 12px; color: #94A3B8; font-weight: 600; text-align: center; width: 100%;">
                 Disclaimer: This tool is for educational and screening purposes. Model accuracy depends on the training data domain.
            </p>
        </div>
        """
    )

    # Interaction & Wiring    
    predict_btn.click(
        fn=predict_ner, 
        inputs=[text_input], 
        outputs=[highlighted_output, summary_output, metrics_output, breakdown_output, entity_plot_output]
    )
    
    text_input.submit(
        fn=predict_ner, 
        inputs=[text_input], 
        outputs=[highlighted_output, summary_output, metrics_output, breakdown_output, entity_plot_output]
    )

if __name__ == "__main__":
    print("\n Starting NER Interface...")
    print(" URL: http://127.0.0.1:7860\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


