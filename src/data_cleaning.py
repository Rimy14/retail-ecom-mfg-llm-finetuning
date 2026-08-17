import os
import json
import random

def clean_text(text):
    """
    Cleans and normalizes raw text input.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip leading/trailing whitespaces and normalize internal spacing
    text = " ".join(text.split())
    
    # Standardize curly quotes and apostrophes to straight ones
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    
    return text

def clean_and_split_data(raw_data_dir="data/raw", processed_data_dir="data/processed"):
    """
    Cleans raw JSON data, converts to standard instruction-response pairs, 
    removes duplicates, shuffles deterministically, and splits 80/10/10.
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Files
    retail_raw_path = os.path.join(raw_data_dir, "retail_ecommerce_raw.json")
    mfg_raw_path = os.path.join(raw_data_dir, "manufacturing_raw.json")
    
    combined_pairs = []
    
    # Load and clean Retail data
    print("[*] Loading Retail dataset...")
    if os.path.exists(retail_raw_path):
        with open(retail_raw_path, "r", encoding="utf-8") as f:
            retail_data = json.load(f)
            
        for row in retail_data:
            instruction = clean_text(row.get("instruction", ""))
            response = clean_text(row.get("response", ""))
            if instruction and response:
                combined_pairs.append({
                    "instruction": instruction,
                    "response": response
                })
        print(f"[+] Loaded {len(retail_data)} raw retail items.")
    else:
        print(f"[-] WARNING: Retail raw data file not found at {retail_raw_path}")

    # Load and clean Manufacturing data
    print("[*] Loading Manufacturing dataset...")
    if os.path.exists(mfg_raw_path):
        with open(mfg_raw_path, "r", encoding="utf-8") as f:
            mfg_data = json.load(f)
            
        for row in mfg_data:
            instruction = clean_text(row.get("instruction", ""))
            response = clean_text(row.get("response", ""))
            if instruction and response:
                combined_pairs.append({
                    "instruction": instruction,
                    "response": response
                })
        print(f"[+] Loaded {len(mfg_data)} raw manufacturing items.")
    else:
        print(f"[-] WARNING: Manufacturing raw data file not found at {mfg_raw_path}")

    total_loaded = len(combined_pairs)
    print(f"[*] Total combined instruction-response pairs loaded: {total_loaded}")
    
    # Deduplication
    unique_pairs = []
    seen_instructions = set()
    for pair in combined_pairs:
        # Deduplicate based on instruction content
        if pair["instruction"] not in seen_instructions:
            seen_instructions.add(pair["instruction"])
            unique_pairs.append(pair)
            
    total_unique = len(unique_pairs)
    print(f"[+] Deduplication complete. Remaining unique records: {total_unique} (Removed {total_loaded - total_unique} duplicates).")
    
    # Deterministic Shuffle for reproducibility
    print("[*] Shuffling dataset deterministically...")
    random.seed(42)
    random.shuffle(unique_pairs)
    
    # Split 80 / 10 / 10
    total = len(unique_pairs)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_split = unique_pairs[:train_end]
    val_split = unique_pairs[train_end:val_end]
    test_split = unique_pairs[val_end:]
    
    print(f"\n[+] Split splits count:")
    print(f"    - Train Split (80%): {len(train_split)} items")
    print(f"    - Val Split (10%): {len(val_split)} items")
    print(f"    - Test Split (10%): {len(test_split)} items")
    
    # Save splits
    splits = {
        "train.json": train_split,
        "val.json": val_split,
        "test.json": test_split
    }
    
    for filename, split_data in splits.items():
        output_path = os.path.join(processed_data_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved split file: {output_path}")

if __name__ == "__main__":
    # Clean and split locally if run directly
    clean_and_split_data()
