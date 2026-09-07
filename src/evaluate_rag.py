import os
import json
import random
import argparse
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import our RAG Pipeline
from rag_pipeline import RetailMfgRAGPipeline

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End RAG Pipeline on 200 Queries")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base Foundation Model")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Path to Fine-Tuned v3 LoRA Adapter")
    parser.add_argument("--test_file", type=str, default="data/processed/test_v3.json", help="Test dataset path")
    parser.add_argument("--kb_dir", type=str, default="data/knowledge_base", help="Knowledge base documents directory")
    parser.add_argument("--chroma_dir", type=str, default="data/chroma_db", help="ChromaDB persistence directory")
    parser.add_argument("--output_file", type=str, default="models/evaluation/rag_pipeline_200_results.json", help="Output results path")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of test queries to benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    print("\n=======================================================")
    print("[*] DAY 9: End-to-End RAG Pipeline Benchmark (200 Queries)")
    print(f"[*] Base Model:      {args.model_id}")
    print(f"[*] Adapter Path:    {args.adapter_dir}")
    print(f"[*] Test Dataset:    {args.test_file}")
    print(f"[*] Number of Tests: {args.num_samples}")
    print("=======================================================\n")
    
    # 1. Initialize & Index Vector Database
    rag_pipe = RetailMfgRAGPipeline(persist_dir=args.chroma_dir)
    rag_pipe.index_knowledge_directory(args.kb_dir)
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    
    # 2. Load Model & Tokenizer
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("[*] Loading Model in 4-bit Quantization...")
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
        torch_dtype=torch.float16,
        token=token
    )
    
    for name, param in base_model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buf in base_model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
            
    adapter_path = args.adapter_dir
    if "{gdrive_dir}" in adapter_path or not os.path.exists(adapter_path):
        for candidate_root in ["/content/drive/MyDrive/Retail LLM", "/content/drive/MyDrive/Retail", "/content/Retail", "."]:
            clean_sub = adapter_path.replace("{gdrive_dir}/", "").replace("{gdrive_dir}", "")
            candidate = os.path.join(candidate_root, clean_sub)
            if os.path.exists(os.path.join(candidate, "adapter_config.json")):
                adapter_path = candidate
                break
                
    print(f"[*] Attaching Fine-Tuned v3 LoRA Adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # 3. Load Test Data
    if not os.path.exists(args.test_file):
        # Fallback to test.json if test_v3.json not directly in local path
        fallback = args.test_file.replace("_v3", "")
        if os.path.exists(fallback):
            args.test_file = fallback
            
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    if len(test_data) > args.num_samples:
        print(f"[*] Randomly sampling {args.num_samples} evaluation queries from {len(test_data)} available.")
        eval_samples = random.sample(test_data, args.num_samples)
    else:
        print(f"[*] Evaluating all {len(test_data)} test queries.")
        eval_samples = test_data

    # 4. Scorers
    rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    results = []
    total_r1, total_r2, total_rl, total_bleu = 0.0, 0.0, 0.0, 0.0
    
    print("\n[*] Running RAG Retrieval + LLM Generation...")
    for idx, sample in enumerate(eval_samples):
        query = sample.get("instruction", "")
        reference = sample.get("response", "")
        
        # End-to-end RAG call
        rag_output = rag_pipe.generate_rag_response(
            model=model,
            tokenizer=tokenizer,
            query=query,
            top_k=2,
            max_new_tokens=150
        )
        prediction = rag_output["response"]
        retrieved_context = rag_output["context"]
        
        # Metric Calculations
        rouge_scores = rouge_scorer_inst.score(reference, prediction)
        r1 = rouge_scores['rouge1'].fmeasure
        r2 = rouge_scores['rouge2'].fmeasure
        rl = rouge_scores['rougeL'].fmeasure
        
        ref_tokens = nltk.word_tokenize(reference.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        
        total_r1 += r1
        total_r2 += r2
        total_rl += rl
        total_bleu += bleu
        
        results.append({
            "query": query,
            "retrieved_context": retrieved_context,
            "reference": reference,
            "prediction": prediction,
            "sources": rag_output["sources"],
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
            "bleu": bleu
        })
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_samples):
            print(f"    - Processed {idx + 1}/{len(eval_samples)} queries | Current Avg BLEU: {total_bleu/(idx+1):.4f}")

    n = len(eval_samples)
    summary = {
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "total_evaluated": n,
        "mean_rouge1": total_r1 / n,
        "mean_rouge2": total_r2 / n,
        "mean_rougeL": total_rl / n,
        "mean_bleu": total_bleu / n
    }
    
    print("\n=================== RAG EVALUATION SUMMARY (200 QUERIES) ===================")
    print(f"[+] Mean ROUGE-1: {summary['mean_rouge1']:.4f}")
    print(f"[+] Mean ROUGE-2: {summary['mean_rouge2']:.4f}")
    print(f"[+] Mean ROUGE-L: {summary['mean_rougeL']:.4f}")
    print(f"[+] Mean BLEU:    {summary['mean_bleu']:.4f}")
    print("============================================================================\n")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Detailed RAG evaluation report saved to: {args.output_file}")

if __name__ == "__main__":
    main()
