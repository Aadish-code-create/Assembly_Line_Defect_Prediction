""" Project: Assembly Line Defect prediction
    Function: Sample Data for numeric 
    Objective: Raw Data Subsampling / Memory Management"""
    
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "raw" / "train_numeric.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "train_numeric_sample.csv"

CHUNK_SIZE = 100_000  # read 100k rows at a time, not the whole file at once

positive_rows = [] # rows where response = 1
negative_rows = [] # rows where response = 0

NEGATIVE_SAMPLE_SIZE = 0.05 # undersampling

chunk_counter = 0
for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk_counter += 1
    print(f"Processing chunk {chunk_counter}...")
    
    # Separate positive and negative samples
    positive_chunk = chunk[chunk['Response'] == 1]
    negative_chunk = chunk[chunk['Response'] == 0]
    
    # Append positive samples
    positive_rows.append(positive_chunk)
    
    # Undersample negative samples
    negative_sampled = negative_chunk.sample(frac=NEGATIVE_SAMPLE_SIZE, random_state=42)
    negative_rows.append(negative_sampled)
    
final_df = pd.concat(positive_rows + negative_rows, ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)
final_df["Id"].to_csv("data/processed/sampled_ids.csv", index=False)


print("Done.")
print("Total rows in the sampled dataset:", len(final_df))
print("Positive rows: ", (final_df['Response'] == 1).sum())
print("Negative rows: ", (final_df['Response'] == 0).sum())

