import os

# ---- IMPORT YOUR MODULES ----

from asr.english_asr import transcribe_en
from asr.hindi_asr import transcribe_hi
from translation.translate import translate_text
from summarization.summarize import summarize_text

def run_pipeline(audio_path, input_lang, output_lang, do_summary=True):
print("\n[INFO] Starting Pipeline...")

```
# -------------------------
# ASR STEP
# -------------------------
if input_lang == "en":
    print("[INFO] Using English ASR")
    transcript = transcribe_en(audio_path)

elif input_lang == "hi":
    print("[INFO] Using Hindi ASR")
    transcript = transcribe_hi(audio_path)

else:
    raise ValueError("Unsupported language")

print("\n[TRANSCRIPT]")
print(transcript)

# -------------------------
# TRANSLATION (if needed)
# -------------------------
if input_lang != output_lang:
    print("\n[INFO] Translating...")
    transcript = translate_text(transcript, src=input_lang, tgt=output_lang)

    print("\n[TRANSLATED TEXT]")
    print(transcript)

# -------------------------
# SUMMARIZATION
# -------------------------
if do_summary:
    print("\n[INFO] Summarizing...")
    summary = summarize_text(transcript, lang=output_lang)

    print("\n[SUMMARY]")
    print(summary)

    return transcript, summary

return transcript, None
```

if **name** == "**main**":
audio_file = "sample.wav"

```
# Example run
run_pipeline(
    audio_path=audio_file,
    input_lang="hi",     # "hi" or "en"
    output_lang="en",    # "hi" or "en"
    do_summary=True
)
```
