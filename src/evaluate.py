import os
import argparse
import json
import torch
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data if not present (handled quietly)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model against reference dataset.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face base model identifier (e.g. Qwen/Qwen2.5-7B-Instruct or meta-llama/Meta-Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Path to the LoRA adapter directory. If None, evaluates the base model only."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test.json",
        help="Path to the test JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the evaluation results JSON file."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random samples to evaluate (default: 100)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    print("\n==============================================")
    print(f"[*] Base Model: {args.model_id}")
    print(f"[*] Adapter Path: {args.adapter_dir}")
    print(f"[*] Output Path: {args.output_file}")
    print("==============================================\n")
    
    # 1. Load Tokenizer
    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Model in 4-bit Quantization (to fit in T4 GPU VRAM)
    print("[*] Loading base model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Force all bfloat16 parameters and buffers in the base model to float16 to prevent bfloat16 propagation
    for name, param in base_model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buf in base_model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
            
    # 3. Load LoRA Adapter if provided
    if args.adapter_dir:
        print(f"[*] Loading LoRA adapter from {args.adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        print("[*] No adapter provided. Evaluating raw base model.")
        model = base_model
        
    model.eval()
    
    # 4. Load Test Dataset
    print(f"[*] Loading test file: {args.test_file}...")
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
        
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    if len(test_data) > args.num_samples:
        print(f"[*] Sampling {args.num_samples} records from {len(test_data)} total test records.")
        # Ensure repeatable sampling using seeded random
        test_samples = random.sample(test_data, args.num_samples)
    else:
        print(f"[*] Using all {len(test_data)} test records.")
        test_samples = test_data
        
    # 5. Setup Scorers
    rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    results = []
    total_r1, total_r2, total_rl, total_bleu = 0.0, 0.0, 0.0, 0.0
    
    # 6. Evaluation Generation Loop
    print("\n[*] Starting text generation and evaluation...")
    for idx, sample in enumerate(test_samples):
        instruction = sample["instruction"]
        reference = sample["response"]
        
        # Build prompt using SFT instruction-tuning prompt template
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Slice outputs to retrieve only the generated completion (ignoring prompt tokens)
        prompt_len = inputs.input_ids.shape[1]
        generation_tokens = outputs[0][prompt_len:]
        prediction = tokenizer.decode(generation_tokens, skip_special_tokens=True).strip()
        
        # Compute ROUGE
        rouge_scores = rouge_scorer_inst.score(reference, prediction)
        r1 = rouge_scores['rouge1'].fmeasure
        r2 = rouge_scores['rouge2'].fmeasure
        rl = rouge_scores['rougeL'].fmeasure
        
        # Compute BLEU (word level)
        ref_tokens = nltk.word_tokenize(reference.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        
        # Accumulate scores
        total_r1 += r1
        total_r2 += r2
        total_rl += rl
        total_bleu += bleu
        
        results.append({
            "instruction": instruction,
            "reference": reference,
            "prediction": prediction,
            "metrics": {
                "rouge1": r1,
                "rouge2": r2,
                "rougeL": rl,
                "bleu": bleu,
                "length": len(prediction)
            }
        })
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_samples):
            print(f"    Processed {idx + 1}/{len(test_samples)} samples...")
            
    # Calculate Summary Scores
    num_evaluated = len(test_samples)
    summary = {
        "mean_rouge1": total_r1 / num_evaluated,
        "mean_rouge2": total_r2 / num_evaluated,
        "mean_rougeL": total_rl / num_evaluated,
        "mean_bleu": total_bleu / num_evaluated
    }
    
    output_data = {
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "summary": summary,
        "results": results
    }
    
    # 7. Write Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print("\n========================= SUMMARY =========================")
    print(f"[+] ROUGE-1 F-Measure: {summary['mean_rouge1']:.4f}")
    print(f"[+] ROUGE-2 F-Measure: {summary['mean_rouge2']:.4f}")
    print(f"[+] ROUGE-L F-Measure: {summary['mean_rougeL']:.4f}")
    print(f"[+] BLEU Score:        {summary['mean_bleu']:.4f}")
    print("===========================================================\n")
    print(f"[+] Detailed evaluation records saved to: {args.output_file}")

if __name__ == "__main__":
    main()
