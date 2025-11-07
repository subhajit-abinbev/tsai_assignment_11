import os

import gradio as gr

from custom_bpe import CustomBPE

# Expect the tokenizer files to be in the Space repo root or subdir
# If you uploaded as 'custom_tokenizer.json'
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "custom_tokenizer.json")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer file {TOKENIZER_PATH} not found. Place it in the Space root.")

tokenizer = CustomBPE.load(TOKENIZER_PATH)

def tokenize(text):
    if not text.strip():
        return {}, ""
    tokens, ids = tokenizer.encode_with_tokens(text)
    lens = [len(t) for t in tokens]
    avg_chars = sum(lens) / len(lens) if tokens else 0
    stats = {
        "num_tokens": len(tokens),
        "avg_chars_per_token": avg_chars,
        "tokens": tokens,
        "ids": ids
    }
    return stats, tokenizer.decode(ids)

with gr.Blocks(title="Bengali BPE Tokenizer") as demo:
    gr.Markdown("# Bengali BPE Tokenizer\nEnter Bengali text below.")
    with gr.Row():
        inp = gr.Textbox(label="Input Text", lines=5)
    with gr.Row():
        stats = gr.JSON(label="Tokenization Stats")
        decoded = gr.Textbox(label="Decoded (Round Trip)")
    btn = gr.Button("Tokenize")
    btn.click(fn=tokenize, inputs=inp, outputs=[stats, decoded])

demo.launch()