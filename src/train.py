import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Pipeline for Retail & Manufacturing LLMs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration JSON file (e.g. configs/qwen_lora_config.json)"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train.json",
        help="Path to the training data file"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/processed/val.json",
        help="Path to the validation data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the fine-tuned model checkpoints (defaults to models/{model_type}_v1)"
    )
    parser.add_argument(
        "--test_subset",
        action="store_true",
        help="If set, runs a quick training check on a tiny dataset slice (50 items) for 5 steps."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, limits the training dataset to this number of samples."
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="If set, limits the validation dataset to this number of samples."
    )
    return parser.parse_args()

def format_example(example):
    """
    Converts a single dataset row into an instruction-tuning prompt string.
    """
    instruction = example['instruction']
    response = example['response']
    example['text'] = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )
    return example

def main():
    args = parse_args()
    
    # 1. Load Model Settings Configuration
    print(f"[*] Loading configuration from: {args.config}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"[-] Config file not found at {args.config}")
        
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    model_type = config.get("model_type", "model")
    model_name = config.get("base_model_name_or_path")
    peft_settings = config.get("peft_config", {})
    quant_settings = config.get("quantization_config", {})
    
    # Auto-assign output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"models/{model_type}_v1"
    print(f"[+] Output Directory: {args.output_dir}")
    
    # 2. Setup Quantization Configuration
    compute_dtype_str = quant_settings.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = torch.bfloat16 if compute_dtype_str == "bfloat16" else torch.float16
    
    if not torch.cuda.is_available():
        raise RuntimeError("[-] CUDA is not available! QLoRA training requires an active GPU runtime.")
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_settings.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_settings.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_settings.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    # 3. Load Tokenizer & Model
    print(f"[*] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right" # SFTTrainer requires padding side right
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[*] Loading base model {model_name} in 4-bit quantization (this will take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("[+] Base model loaded in 4-bit.")
    
    # 1. Force all parameters and buffers in the base model to float16 to prevent bfloat16 propagation
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buf in model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
            
    # 2. Set model config torch_dtype to float32 so PEFT initializes adapters in float32
    model.config.torch_dtype = torch.float32
            
    # 4. Prepare Model for PEFT/LoRA Training
    model = prepare_model_for_kbit_training(model)
    
    # 5. Configure LoRA
    print("[*] Configuring LoRA Adapter...")
    lora_config = LoraConfig(
        r=peft_settings.get("r", 16),
        lora_alpha=peft_settings.get("lora_alpha", 32),
        target_modules=peft_settings.get("target_modules", []),
        lora_dropout=peft_settings.get("lora_dropout", 0.05),
        bias=peft_settings.get("bias", "none"),
        task_type="CAUSAL_LM"
    )
    # NOTE: We do NOT call get_peft_model() here — SFTTrainer applies it via peft_config
    print("[+] LoRA config ready.")
    
    # 6. Load Dataset
    print(f"[*] Loading dataset files: {args.train_file} & {args.val_file}...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file
        }
    )
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))
        print(f"[+] Sliced train dataset to {len(train_dataset)} samples.")
        
    if args.max_val_samples is not None:
        val_dataset = val_dataset.select(range(min(len(val_dataset), args.max_val_samples)))
        print(f"[+] Sliced val dataset to {len(val_dataset)} samples.")
        
    train_dataset = train_dataset.map(format_example)
    val_dataset = val_dataset.map(format_example)
    print(f"[+] Formatted datasets with 'text' column.")
    
    # 7. Configure Training Arguments
    if args.test_subset:
        print("\n==============================================")
        print("[!] TEST MODE ENABLED: Slicing datasets and steps")
        print("==============================================")
        train_dataset = train_dataset.select(range(min(len(train_dataset), 50)))
        val_dataset = val_dataset.select(range(min(len(val_dataset), 10)))
        print(f"[+] Sliced datasets: Train = {len(train_dataset)} | Val = {len(val_dataset)}")
        
        training_args = SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=512,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5, # Run only 5 steps to verify loop
            learning_rate=2e-4,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="no",
            fp16=True,
            report_to="none", # Disable W&B logging for quick tests
            remove_unused_columns=False,
            disable_tqdm=False
        )
    else:
        print(f"[+] Datasets loaded: Train = {len(train_dataset)} | Val = {len(val_dataset)}")
        training_args = SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=512,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4, # Effective batch size = 16
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            fp16=True,
            report_to="wandb" if os.environ.get("WANDB_DISABLED", "").lower() != "true" else "none",
            lr_scheduler_type="cosine",
            remove_unused_columns=False
        )
        training_args.warmup_ratio = 0.03

    # 8. Initialize SFTTrainer
    print("[*] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args
    )
    
    # Force cast any remaining bfloat16 parameters or buffers inside the trainer model to float32
    # to prevent bfloat16 gradient scaling crash on T4 GPU.
    for name, param in trainer.model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
    for name, buf in trainer.model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float32)
            
    # 9. Launch Training
    print("[*] Starting training...")
    trainer.train()
    print("[+] Training completed successfully!")
    
    # Save final adapter weights
    print(f"[*] Saving adapter weights to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[+] Saving complete.")

if __name__ == "__main__":
    main()
