from jiwer import wer, cer

def compute_asr_metrics(references, predictions):
    # ensure correct type
    references = list(references)
    predictions = list(predictions)
    assert len(references) == len(predictions), "Mismatch in lengths"
    return {
        "WER": wer(references, predictions),
        "CER": cer(references, predictions)
    }