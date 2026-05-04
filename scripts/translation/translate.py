from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "ai4bharat/indictrans2-en-indic"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def translate_text(text, src="hi", tgt="en"):
input_text = f"{src} to {tgt}: {text}"

```
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

outputs = model.generate(**inputs, max_length=256)

return tokenizer.decode(outputs[0], skip_special_tokens=True)
```
