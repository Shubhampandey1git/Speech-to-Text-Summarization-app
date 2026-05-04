from transformers import pipeline

summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, lang="en"):
if lang == "en":
return summarizer_en(text, max_length=100, min_length=30)[0]["summary_text"]

```
else:
    # fallback: return text (or plug IndicBART later)
    return text
```
