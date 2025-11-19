import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()


for filename in [
    "longmemeval_oracle.json",
    "longmemeval_s_cleaned.json"
]:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename=filename,
        local_dir="./data/longmemeval",
        local_dir_use_symlinks=False,
        repo_type="dataset",
    )
    print(f"✓ Downloaded {filename}")

print("Download complete!")
