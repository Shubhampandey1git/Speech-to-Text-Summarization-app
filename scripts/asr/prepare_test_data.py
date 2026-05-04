import pandas as pd

HINDI_CSV = "data/processed/asr/hindi/dev.csv"
ENGLISH_CSV = "data/processed/asr/english/test.csv"

OUTPUT = "data/combine/asr/combined_test.csv"

df_hi = pd.read_csv(HINDI_CSV)
df_hi["lang"] = "hi"

df_en = pd.read_csv(ENGLISH_CSV)
df_en["lang"] = "en"

df = pd.concat([df_hi, df_en]).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

print("Combined test file created:", OUTPUT)
print("Total samples:", len(df))