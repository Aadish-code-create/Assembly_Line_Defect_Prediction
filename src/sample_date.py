""" Project: Assembly Line Defect Prediction
    Function: Filter date data to match sampled Ids
    Objective: Keep only the same parts we already sampled in numeric """

import pandas as pd

INPUT_FILE = "data/raw/train_date.csv"
OUTPUT_FILE = "data/processed/train_date_sample.csv"
IDS_FILE = "data/processed/sampled_ids.csv"

CHUNK_SIZE = 100_000

sampled_ids = pd.read_csv(IDS_FILE)["Id"]
sampled_ids_set = set(sampled_ids)

print("Total Ids to match:", len(sampled_ids_set))

matched_chunks = []
chunk_counter = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk_counter += 1
    print(f"Processing chunk {chunk_counter}...")

    matched = chunk[chunk["Id"].isin(sampled_ids_set)]
    matched_chunks.append(matched)

final_df = pd.concat(matched_chunks, ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print("Total rows matched:", len(final_df))
print("Expected rows:", len(sampled_ids_set))