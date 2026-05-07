import gradio as gr
import torch
import torchaudio
import re
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
# GLOBAL MODEL VARIABLES
# --------------------------------------------------
asr_processor = None
asr_model = None
summarizer = None
current_language = None
hindi_tokenizer = None
hindi_model = None

# --------------------------------------------------
# LOAD MODELS DYNAMICALLY
# --------------------------------------------------
def load_models(language):

    global asr_processor
    global asr_model
    global summarizer
    global hindi_model
    global hindi_tokenizer
    global current_language

    if current_language == language:
        return

    print(f"Loading {language} models...")

    # ----------------------------------------------
    # ENGLISH MODELS
    # ----------------------------------------------
    if language == "English":

        asr_processor = Wav2Vec2Processor.from_pretrained(
            ENGLISH_ASR_PATH
        )

        asr_model = Wav2Vec2ForCTC.from_pretrained(
            ENGLISH_ASR_PATH
        ).to(device)

        asr_model.eval()

        summarizer = pipeline(
            "summarization",
            model=ENGLISH_SUMMARIZATION_PATH,
            device=0 if torch.cuda.is_available() else -1
        )

    # ----------------------------------------------
    # HINDI MODELS
    # ----------------------------------------------
    else:

        asr_processor = Wav2Vec2Processor.from_pretrained(
            HINDI_ASR_PATH
        )

        asr_model = Wav2Vec2ForCTC.from_pretrained(
            HINDI_ASR_PATH
        ).to(device)

        asr_model.eval()

        hindi_tokenizer = AutoTokenizer.from_pretrained(
            HINDI_SUMMARIZATION_PATH
        )

        hindi_model = AutoModelForSeq2SeqLM.from_pretrained(
            HINDI_SUMMARIZATION_PATH
        ).to(device)

        hindi_model.eval()

    current_language = language
    print(f"✅ {language} models loaded on {device}")

# --------------------------------------------------
# AUDIO LOADING
# --------------------------------------------------
def load_audio(audio_data):

    sampling_rate, speech_array = audio_data

    speech_array = torch.tensor(
        speech_array,
        dtype=torch.float32
    )

    # Normalize int16 microphone input to float32
    # Microphone: int16 range (-32768, 32767)
    # Files: already float32 range (-1.0, 1.0)
    if speech_array.abs().max() > 1.0:
        speech_array = speech_array / 32768.0

    # Stereo → mono
    if len(speech_array.shape) > 1:
        speech_array = torch.mean(speech_array, dim=1)

    # Resample expects [channel, time]
    if len(speech_array.shape) == 1:
        speech_array = speech_array.unsqueeze(0)

    # Resample to 16kHz
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            sampling_rate, 16000
        )
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    speech_array = speech_array.squeeze(0).numpy()

    return speech_array, sampling_rate

# --------------------------------------------------
# TRANSCRIPTION
# --------------------------------------------------
def transcribe_audio(audio_data, max_seconds=15):

    speech, sr = load_audio(audio_data)

    # Limit audio length to avoid slow processing
    max_audio_length = max_seconds * 16000
    speech = speech[:max_audio_length]

    print(f"   Audio length: {len(speech)/16000:.1f}s")

    inputs = asr_processor(
        speech,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = asr_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = asr_processor.batch_decode(
        predicted_ids
    )[0]

    return transcription

# --------------------------------------------------
# TRANSCRIPT CLEANING
# --------------------------------------------------
def clean_transcript(text, language):

    text = re.sub(r"\s+", " ", text)

    if language == "English":
        text = re.sub(
            r"[^a-zA-Z0-9\s.,!?']",
            " ",
            text
        )
    else:
        text = re.sub(
            r"[^\u0900-\u097F\s.,!?0-9]",
            " ",
            text
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# --------------------------------------------------
# SUMMARIZATION
# --------------------------------------------------
def summarize_text(text, language):

    # ----------------------------------------------
    # ENGLISH BART
    # ----------------------------------------------
    if language == "English":

        summary = summarizer(
            text,
            max_length=80,
            min_length=20,
            do_sample=False
        )[0]["summary_text"]

        return summary

    # ----------------------------------------------
    # HINDI INDICBART
    # ----------------------------------------------
    else:

        inputs = hindi_tokenizer(
            f"<2hi> summarize: {text}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs.pop("token_type_ids", None)

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            summary_ids = hindi_model.generate(
                **inputs,
                max_length=60,
                min_length=15,
                num_beams=3,
                repetition_penalty=2.0,
                early_stopping=True
            )

        summary = hindi_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        summary = summary.replace("<2hi>", "")
        summary = summary.replace("summarize:", "")
        summary = summary.strip()

        return summary

# --------------------------------------------------
# SUMMARY CLEANING
# --------------------------------------------------
def clean_summary(summary, language):

    summary = re.sub(r"\s+", " ", summary)

    if language == "English":
        summary = re.sub(
            r"[^a-zA-Z0-9\s.,!?']",
            " ",
            summary
        )
    else:
        summary = re.sub(
            r"[^\u0900-\u097F\s.,!?0-9]",
            " ",
            summary
        )

    summary = re.sub(r"\s+", " ", summary)

    return summary.strip()

# --------------------------------------------------
# FULL PIPELINE
# --------------------------------------------------
def full_pipeline(audio, language):

    if audio is None:
        return "", "", None

    print(f"\n--- Pipeline started ({language}) ---")
    print(f"Device: {device}, CUDA: {torch.cuda.is_available()}")

    # Load models
    load_models(language)

    # ASR
    print("Running ASR...")
    raw_transcript = transcribe_audio(audio, max_seconds=15)
    print(f"   Raw: {raw_transcript[:80]}...")

    # Clean transcript
    cleaned_transcript = clean_transcript(raw_transcript, language)
    print(f"   Cleaned: {cleaned_transcript[:80]}...")

    # Summarization
    print("Running summarization...")
    raw_summary = summarize_text(cleaned_transcript, language)

    # Clean summary
    cleaned_summary = clean_summary(raw_summary, language)
    print(f"   Summary: {cleaned_summary[:80]}...")

    print("--- Pipeline done ---\n")

    # Return audio back so it doesn't go silent
    return cleaned_transcript, cleaned_summary, audio

# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------
with gr.Blocks(
    title="Speech-to-Text Summarization System"
) as demo:

    gr.Markdown(
        """
        # 🎙️ Speech-to-Text Summarization System
        Audio → ASR → Transcript Cleaning → Summarization → Smart Notes
        """
    )

    with gr.Row():

        language_input = gr.Radio(
            choices=["English", "Hindi"],
            value="English",
            label="Select Language"
        )

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Record or Upload Audio (max 15s for mic)"
        )

    generate_button = gr.Button(
        "🚀 Generate Summary",
        variant="primary"
    )

    # Playback so audio doesn't go silent after processing
    audio_output = gr.Audio(
        label="Your Recording (Playback)",
        type="numpy",
        interactive=False
    )

    transcript_output = gr.Textbox(
        label="Processed Transcript",
        lines=10
    )

    summary_output = gr.Textbox(
        label="Final Summary",
        lines=6
    )

    generate_button.click(
        fn=full_pipeline,
        inputs=[audio_input, language_input],
        outputs=[transcript_output, summary_output, audio_output],
        show_progress=True
    )

# --------------------------------------------------
# LAUNCH
# --------------------------------------------------
if __name__ == "__main__":
    demo.launch(ssl_verify=False, share=True)