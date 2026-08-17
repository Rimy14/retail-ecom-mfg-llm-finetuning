import os
import json
from datasets import load_dataset

def collect_datasets(raw_data_dir="data/raw"):
    """
    Downloads domain-specific datasets from Hugging Face Hub and saves them to local storage.
    
    Args:
        raw_data_dir (str): Path to save raw dataset files.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    print(f"[*] Target raw data directory: {raw_data_dir}")
    
    # -------------------------------------------------------------
    # 1. Retail & E-commerce Customer Support Dataset
    # -------------------------------------------------------------
    retail_dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    retail_output_path = os.path.join(raw_data_dir, "retail_ecommerce_raw.json")
    
    print(f"\n[*] Downloading Retail & E-commerce dataset: {retail_dataset_name}...")
    try:
        # Load the dataset
        retail_ds = load_dataset(retail_dataset_name, split="train")
        
        # Convert to list of dicts
        retail_data = [
            {
                "instruction": row["instruction"],
                "response": row["response"],
                "category": row["category"],
                "intent": row["intent"]
            }
            for row in retail_ds
        ]
        
        # Save to JSON
        with open(retail_output_path, "w", encoding="utf-8") as f:
            json.dump(retail_data, f, ensure_ascii=False, indent=2)
            
        print(f"[+] Retail dataset saved to {retail_output_path}")
        print(f"    - Total records: {len(retail_data)}")
        print(f"    - Sample Query: '{retail_data[0]['instruction']}'")
        
    except Exception as e:
        print(f"[-] Error downloading Retail dataset: {e}")
        
    # -------------------------------------------------------------
    # 2. Manufacturing / Lean Six Sigma Q&A Dataset
    # -------------------------------------------------------------
    mfg_dataset_name = "cw18/lean-six-sigma-qna-v1"
    mfg_output_path = os.path.join(raw_data_dir, "manufacturing_raw.json")
    
    print(f"\n[*] Downloading Manufacturing & Operations dataset: {mfg_dataset_name}...")
    try:
        # Load the dataset
        mfg_ds = load_dataset(mfg_dataset_name, split="train")
        
        # Convert to list of dicts
        # Note: This dataset contains QA pairs on Lean Six Sigma concepts (DMAIC, process control, etc.)
        mfg_data = []
        for row in mfg_ds:
            # Match schema structure based on dataset features
            # Standard instruction-tuning keys: instruction (question) and response (answer)
            instruction = row.get("question", row.get("instruction", ""))
            response = row.get("answer", row.get("response", ""))
            
            # If default keys are missing, search available columns
            if not instruction or not response:
                keys = list(row.keys())
                instruction = row.get(keys[0], "")
                response = row.get(keys[1], "") if len(keys) > 1 else ""
                
            mfg_data.append({
                "instruction": instruction,
                "response": response,
                "domain": "manufacturing_process_improvement"
            })
            
        # Save to JSON
        with open(mfg_output_path, "w", encoding="utf-8") as f:
            json.dump(mfg_data, f, ensure_ascii=False, indent=2)
            
        print(f"[+] Manufacturing dataset saved to {mfg_output_path}")
        print(f"    - Total records: {len(mfg_data)}")
        print(f"    - Sample Query: '{mfg_data[0]['instruction']}'")
        
    except Exception as e:
        print(f"[-] Error downloading Manufacturing dataset: {e}")

if __name__ == "__main__":
    # If run standalone, collect data locally
    collect_datasets()
