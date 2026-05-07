import gradio as gr
import torch
import torchaudio
import re
import gc
import os
import shutil
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------
HINDI_ASR_PATH = "models/asr/hindi/final"
ENGLISH_ASR_PATH = "models/asr/english/final"

HINDI_SUMMARIZATION_PATH = "models/summarization/hindi_indicbart_pretrained"
ENGLISH_SUMMARIZATION_PATH = "models/summarization/english_bart"

# --------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------
asr_processor = None
asr_model = None

summarizer = None
hindi_model = None
hindi_tokenizer = None

current_language = None

# --------------------------------------------------
# VRAM CLEANUP
# --------------------------------------------------
def clear_memory():
    global asr_processor
    global asr_model
    global summarizer
    global hindi_model
    global hindi_tokenizer

    asr_processor = None
    asr_model = None
    summarizer = None
    hindi_model = None
    hindi_tokenizer = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --------------------------------------------------
# DYNAMIC MODEL LOADING
# --------------------------------------------------
def load_models(language):
    global asr_processor
    global asr_model
    global summarizer
    global hindi_model
    global hindi_tokenizer
    global current_language

    # If language changes -> unload old models
    if current_language != language:
        print(f"\n--- Switching to {language} ---")
        clear_memory()

    # If already loaded correctly -> skip
    if current_language == language and asr_model is not None:
        return

    # ---------------- ENGLISH ----------------
    if language == "English":

        print("Loading English ASR Model...")

        asr_processor = Wav2Vec2Processor.from_pretrained(
            ENGLISH_ASR_PATH
        )

        asr_model = Wav2Vec2ForCTC.from_pretrained(
            ENGLISH_ASR_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        asr_model.eval()

        print("Loading English Summarization Model...")

        summarizer = pipeline(
            "summarization",
            model=ENGLISH_SUMMARIZATION_PATH,
            device=-1
        )

    # ---------------- HINDI ----------------
    else:

        print("Loading Hindi ASR Model...")

        asr_processor = Wav2Vec2Processor.from_pretrained(
            HINDI_ASR_PATH
        )

        asr_model = Wav2Vec2ForCTC.from_pretrained(
            HINDI_ASR_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        asr_model.eval()

        print("Loading Hindi Summarization Model...")

        hindi_tokenizer = AutoTokenizer.from_pretrained(
            HINDI_SUMMARIZATION_PATH
        )

        hindi_model = AutoModelForSeq2SeqLM.from_pretrained(
            HINDI_SUMMARIZATION_PATH
        ).to("cpu")

        hindi_model.eval()

    current_language = language

    print(f"✅ {language} models loaded successfully")

# --------------------------------------------------
# AUDIO PROCESSING
# --------------------------------------------------
def process_audio(audio_path):

    speech, sr = torchaudio.load(audio_path)

    # Convert stereo -> mono
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)

    # Limit to 15 seconds
    max_samples = 15 * 16000
    speech = speech[:, :max_samples]

    return speech.squeeze().numpy()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text, language):

    text = re.sub(r"\s+", " ", text)

    if language == "English":
        pattern = r"[^a-zA-Z0-9\s.,!?']"
    else:
        pattern = r"[^\u0900-\u097F\s.,!?0-9]"

    text = re.sub(pattern, " ", text)

    return re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------
# SUMMARIZATION
# --------------------------------------------------
def summarize_text(text, language):

    if not text.strip():
        return "No transcript available."

    # ---------------- ENGLISH ----------------
    if language == "English":

        summary = summarizer(
            text,
            max_length=80,
            min_length=20,
            do_sample=False
        )[0]["summary_text"]

        return summary

    # ---------------- HINDI ----------------
    else:

        inputs = hindi_tokenizer(
            f"<2hi> summarize: {text}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs.pop("token_type_ids", None)

        with torch.no_grad():

            summary_ids = hindi_model.generate(
                **inputs,
                max_length=60,
                min_length=15,
                num_beams=3,
                repetition_penalty=2.0
            )

        summary = hindi_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        summary = (
            summary
            .replace("<2hi>", "")
            .replace("summarize:", "")
            .strip()
        )

        return summary

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def full_pipeline(audio_input, language):

    if audio_input is None:
        return "Please upload or record audio.", "", None

    # --------------------------------------------------
    # SAVE RECORDED AUDIO AS input_audio.wav
    # --------------------------------------------------

    save_path = "input_audio.wav"

    # Remove old file if exists
    if os.path.exists(save_path):
        os.remove(save_path)

    # Copy uploaded/recorded file
    shutil.copy(audio_input, save_path)

    print(f"\nAudio saved as: {save_path}")

    # --------------------------------------------------
    # PROCESS AUDIO FIRST
    # --------------------------------------------------

    print("Processing audio to 16kHz mono...")

    speech = process_audio(save_path)

    # --------------------------------------------------
    # LOAD MODELS ONLY AFTER BUTTON CLICK
    # --------------------------------------------------

    load_models(language)

    # --------------------------------------------------
    # ASR
    # --------------------------------------------------

    print("Running ASR...")

    inputs = asr_processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    if device == "cuda":
        input_values = input_values.to(torch.float16)

    with torch.no_grad():
        logits = asr_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcript = asr_processor.batch_decode(
        predicted_ids
    )[0]

    transcript = clean_text(transcript, language)

    print("Transcript:", transcript)

    # --------------------------------------------------
    # SUMMARIZATION
    # --------------------------------------------------

    print("Running Summarization...")

    summary = summarize_text(transcript, language)

    summary = clean_text(summary, language)

    print("Summary:", summary)

    return transcript, summary, save_path

# --------------------------------------------------
# UI (UNCHANGED)
# --------------------------------------------------
with gr.Blocks(title="Speech-to-Text Summarization System") as demo:

    gr.Markdown("# 🎙️ Speech-to-Text Summarization System")

    with gr.Row():

        language_input = gr.Radio(
            choices=["English", "Hindi"],
            value="English",
            label="Select Language"
        )

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Audio Input"
        )

    generate_button = gr.Button(
        "🚀 Generate Summary",
        variant="primary"
    )

    audio_output = gr.Audio(
        label="Playback",
        type="filepath",
        interactive=False
    )

    transcript_output = gr.Textbox(
        label="Processed Transcript",
        lines=8
    )

    summary_output = gr.Textbox(
        label="Final Summary",
        lines=4
    )

    generate_button.click(
        fn=full_pipeline,
        inputs=[audio_input, language_input],
        outputs=[
            transcript_output,
            summary_output,
            audio_output
        ]
    )

# --------------------------------------------------
# LAUNCH
# --------------------------------------------------
if __name__ == "__main__":

    # NO PRELOADING
    demo.queue().launch(share=True)