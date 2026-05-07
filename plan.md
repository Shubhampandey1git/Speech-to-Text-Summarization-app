# ***BEST POSSIBLE PLAN (Aligned with paper)***

**Phase 1 - ASR Comparison**

Whisper -> baseline  

Wav2Vec2 -> main model 

Custom -> main contribution

(optional mention Conformer from literature)


**Phase 2 - Multilingual Handling**

Train Wav2Vec2 on Hindi and English seperately(Language Aware Models)

Train/ Add a Translation model(to Latin by default)

Test code-mixed speech


**Phase 3 - Improvement Layer (OWN IDEA)**

Grammar correction  

Script normalization  

Dictionary filtering

Use mT5 finetuned model


**Phase 4 - Summarization**

BART / T5 / IndicBART


**Phase 5 - Chat Bot/ QA layer**

{Not decided yet}


**Phase 6 - User Interface**

Android java/kotlin App


**FINAL FLOW SYSTEM:**

Audio Input -> ASR -> Text Normalization & Correction -> Summarization -> Chatbot/ QA Layer -> UI (App)

---
---

# **ASR PLANS**

### Whisper Evaluation
1. Using Whisper Base(because of hardware limits), run the model through the whole dataset(train sets only).
2. Get the results as CSV and check the observations of the baseline model.
---
### Wav2Vec2 improvement
1. Using base, use only part of dataset for faster processing/training.
2. Get the eval results on test set as well as train set and compare with whisper.
3. Accordingly improve upon the Wav2Vec2;
    - Option1: by more training -> use multilingual data/ current data
    - Option2: use better model -> use colab/kaggle TPU for high processing
4. Move to custom model.
---
### AI4BHARAT wav2vec2 model
1. Use for Hindi as well as english
2. Compare pre-trained and fine-tuned models and use accordingly.
---
### Custom Acoustic model(not ASR)
1. Large scale custom ASR is not possible for the dataset size and hardware at hand, thus we make a lightweight ASR.
2. We Can do: CNN/ Conv1D encoder + BiLSTM/ Tranformer + CTC loss
3. Architecture:
    - INPUT: Raw audio -> Mel Spectogram
    - Model: [Conv layers] -> [BiLSTM or Transformer] -> [Linear] -> CTC
    - Output: Character-level tokens (model's vocab.json)
4. Design decisions:
    - Tokenization: use vocab.json (Already done)
    - Audio representation: Mel Spectogram (80 bins)
    - Model size(Small): 2-4 Conv layers, 2-3 BiLSTM layers, Hidden Size -> 256-512
5. Training Strategy:
    - Dataset Cleaning: remove noise, Null values and normalize the text (Done)
    - Language balancing: 50-50 split for hindi and english
    - Optional Augmentation: noise, speed perturbation
6. Evaluate and compare with whisper and Wav2Vec2.

---
---

# **TEXT CORRECTION PLAN**

### System Flow
ASR Output -> Text Correction Model(Correction/ Normalization) -> Summarization Model

### Procedure
1. Basic Cleaning.
2. Hinglish -> Hindi/english normalization (use mappinngs).
3. Script detection: Is the word hindi or english.
4. Finetune mT5 model using dataset with noisy ->  clean values.
5. Use finetuned mT5(multiligual) for LM based grammar correction.

====================================================================
# **SUMMARIZATION PLAN**

### MODEL
1. Use BART, PEGASUS for research perpose as they will give bad results.
2. Need Multilingual (Hindi + English) nose resistant context-aware model(Transformer/ LLM).
3. Options:

    - Instruction-tuned multilingual LLM- Requirements:
        - handles noisy text
        - understands phonetics via context
        - works for summary + chatbot
        - no retraining needed
    
    - mT5(small/ base)- is multilingual but weaker conversational summarization and sensitive to noise.
    
    - IndicBART(Hindi-specialized)- optimized for Indian Languages, uses devanagri normalization.

*NOTE(IMP): Need to use hybrid architecture used in industries for the final app:*

### UPDATED ARCHITECTURE
```mermaid
flowchart TD

    A[📱 App (Local)] --> B[🎤 Audio Input]

    B --> C[🧠 Local ASR<br/>Wav2Vec2]

    C --> D[🌐 API Call]

    D --> E[🤖 LLM<br/>Summarization / Chat]

    E --> F[📱 Smart Notes Output]
```
---
---

# **DEPLOYMENT**

### DEMO
1. Build a simple interface
2. Options:
    - Streamlit (fastest)
    - Gradio (clean UI, great for demos)
    - Tkinter (desktop app)
    - Flask (for backend feel)
3. Integrate with pipeline:
    - Upload audio
    - → ASR (your model)
    - → LLM (summary/chat)
    - → Display output
4. Run it locally using: 
    ```bash
        python app.py
    ```
or

Desktop app: Use- PyInstaller → convert to .exe
---
### Final Demo Architecture
```mermaid
flowchart TD

    A[🎤 Audio Input<br/>Hindi / English]

    A --> B[🧠 ASR Engine<br/>IndicWav2Vec Hindi (Fine-tuned)<br/>Wav2Vec2-Base-960h (Fine-tuned)]

    B --> C[📝 Raw Transcripts]

    C --> D[🧹 Transcript Preprocessing]

    D --> E[🤖 Summarization Engine<br/>IndicBART (Pretrained)<br/>BART-Base (Fine-tuned)]

    E --> F[📄 Summary Output]

    F --> G[✨ Summary Cleaning]

    G --> H[💬 Display Summary / Feed to LLM]
```
---
### Future Upgrades
1. Fully portable (advanced):
    - Quantized models
    - ONNX / GGUF
    - Smaller ASR + small LLM
2. Mobile app:
    - React Native / Flutter frontend
    - Backend API for models
3. Edge AI version:
    - Tiny ASR model
    - Tiny LLM (distilled)
4. Better LLM cum Summarizer:
    - LLM instead of a summarizer
5. Better custom ASRs:
    - Better transcription
    - Context and speaker aware
6. More languages:
    - Add other Indian languages like Marathi
---