import os
import json
import random
import re

def extract_context_from_sample(instruction, response, item):
    """
    Synthesizes a declarative, fact-grounded context passage from the instruction and response.
    Acts as the retrieved reference document passage for RAG-aware training.
    """
    text_corpus = (instruction + " " + response).lower()
    
    # Check domain heuristics
    is_mfg = (
        item.get("domain") in ["manufacturing_process_improvement", "lean_six_sigma", "statistical_process_control", "root_cause_analysis"]
        or any(k in text_corpus for k in ["dmaic", "six sigma", "spc", "x-bar", "p-chart", "defect", "assembly line", "root cause", "5 whys", "pareto", "calibration"])
    )
    
    prefix = "Process Operations Manual: " if is_mfg else "Customer Support Policy Guide: "
    
    # Split response into sentences
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
    if not raw_sentences:
        clean_facts = response.strip()
    else:
        # Take up to first 2 sentences containing core instructions/facts
        candidate = " ".join(raw_sentences[:min(2, len(raw_sentences))])
        
        # Remove colloquial greetings/conversational openers
        clean_facts = re.sub(
            r'^(i apologize|we apologize|i am very sorry|thank you for contacting|certainly!?|yes,?|no,?|i have checked|looking at|sure!?|hello,?|hi,?|honored to assist!?|i understand|rest assured,?)\s*',
            '',
            candidate,
            flags=re.IGNORECASE
        )
        
        # Remove conversational closing questions/disclaimers
        clean_facts = re.sub(
            r'(please let us know|feel free to contact|how can i help|is there anything else|thank you for your patience|reach out to us).*$',
            '',
            clean_facts,
            flags=re.IGNORECASE
        )
        clean_facts = clean_facts.strip()
        
        if not clean_facts:
            clean_facts = raw_sentences[0]

    # Ensure clean capitalisation and period
    if clean_facts and not clean_facts[0].isupper():
        clean_facts = clean_facts[0].upper() + clean_facts[1:]
    if clean_facts and not clean_facts.endswith('.'):
        clean_facts += '.'
        
    return f"{prefix}{clean_facts}"

def process_dataset(input_path, output_path, is_train=False, synthetic_path=None):
    """
    Loads dataset, merges synthetic data if training, generates 'context' field for all records, and saves output.
    """
    records = []
    if os.path.exists(input_path):
        print(f"[*] Loading data from: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"[+] Loaded {len(records)} records from {input_path}")
    else:
        print(f"[!] Warning: File {input_path} not found.")

    if is_train and synthetic_path and os.path.exists(synthetic_path):
        print(f"[*] Loading synthetic QA pairs from: {synthetic_path}")
        with open(synthetic_path, "r", encoding="utf-8") as f:
            synthetic_records = json.load(f)
        print(f"[+] Merging {len(synthetic_records)} synthetic records into training set...")
        records = records + synthetic_records

    # Add 'context' field to each record
    v4_records = []
    for item in records:
        inst = item.get("instruction", "")
        resp = item.get("response", "")
        
        # If context does not already exist, generate it
        ctx = item.get("context")
        if not ctx:
            ctx = extract_context_from_sample(inst, resp, item)
            
        new_item = {
            "instruction": inst,
            "context": ctx,
            "response": resp
        }
        
        # Preserve original metadata if present
        for key in ["category", "intent", "domain"]:
            if key in item:
                new_item[key] = item[key]
                
        v4_records.append(new_item)

    if is_train:
        random.seed(42)
        random.shuffle(v4_records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(v4_records, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Saved {len(v4_records)} RAG-aware records to: {output_path}")
    if v4_records:
        print(f"    - Sample Instruction: {v4_records[0]['instruction'][:80]}...")
        print(f"    - Sample Context:     {v4_records[0]['context'][:100]}...")
    return len(v4_records)

def create_v4_configs(project_root):
    """
    Creates v4 configuration files pointing to models/qwen_v4 and models/llama_v4.
    """
    print("[*] Creating v4 LoRA configuration files...")
    configs_dir = os.path.join(project_root, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    qwen_v4_cfg = {
        "model_type": "qwen",
        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "peft_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
        },
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16"
        }
    }
    
    llama_v4_cfg = {
        "model_type": "llama",
        "base_model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "peft_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
        },
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16"
        }
    }
    
    with open(os.path.join(configs_dir, "qwen_lora_config_v4.json"), "w", encoding="utf-8") as f:
        json.dump(qwen_v4_cfg, f, indent=2)
    with open(os.path.join(configs_dir, "llama_lora_config_v4.json"), "w", encoding="utf-8") as f:
        json.dump(llama_v4_cfg, f, indent=2)
        
    print("[+] configs/qwen_lora_config_v4.json and configs/llama_lora_config_v4.json successfully written.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, "data", "processed")
    
    print("\n=======================================================")
    print("[*] DAY 10: RAG-Aware Dataset (v4) Compilation")
    print(f"[*] Project Root: {project_root}")
    print("=======================================================\n")
    
    # Priority for base training data: train_v3.json -> train_v2.json -> train.json
    train_orig = os.path.join(processed_dir, "train_v3.json")
    synth_path = None
    if not os.path.exists(train_orig):
        train_orig = os.path.join(processed_dir, "train_v2.json")
        if not os.path.exists(train_orig):
            train_orig = os.path.join(processed_dir, "train.json")
            synth_path = os.path.join(processed_dir, "synthetic_qa.json")
            
    # 1. Compile train_v4.json
    process_dataset(
        input_path=train_orig,
        output_path=os.path.join(processed_dir, "train_v4.json"),
        is_train=True,
        synthetic_path=synth_path
    )
    
    # 2. Compile val_v4.json
    val_path = os.path.join(processed_dir, "val_v3.json")
    if not os.path.exists(val_path):
        val_path = os.path.join(processed_dir, "val.json")
    process_dataset(
        input_path=val_path,
        output_path=os.path.join(processed_dir, "val_v4.json"),
        is_train=False
    )
    
    # 3. Compile test_v4.json
    test_path = os.path.join(processed_dir, "test_v3.json")
    if not os.path.exists(test_path):
        test_path = os.path.join(processed_dir, "test.json")
    process_dataset(
        input_path=test_path,
        output_path=os.path.join(processed_dir, "test_v4.json"),
        is_train=False
    )
    
    # 4. Generate configs
    create_v4_configs(project_root)
    print("\n[+] Day 10 dataset preparation complete!")

if __name__ == "__main__":
    main()
