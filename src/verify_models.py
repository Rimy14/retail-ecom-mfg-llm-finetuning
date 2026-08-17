import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

def load_config(config_path):
    """
    Loads JSON configuration files.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[-] Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_gpu_memory(stage=""):
    """
    Utility to check and print CUDA memory usage.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"[GPU Memory - {stage}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        print(f"[Memory - {stage}] CPU RAM in use (CUDA is not available).")

def verify_quantized_load(config_path):
    """
    Loads a model based on its config file in 4-bit quantization and verifies it on CUDA.
    """
    print(f"\n=======================================================")
    print(f"[*] Starting Load Verification for: {config_path}")
    print(f"=======================================================")
    
    # 1. Load config settings
    config = load_config(config_path)
    model_name = config.get("base_model_name_or_path")
    peft_settings = config.get("peft_config", {})
    quant_settings = config.get("quantization_config", {})
    
    print(f"[+] Base Model: {model_name}")
    print(f"[+] Quantization Config: {json.dumps(quant_settings, indent=2)}")
    
    # 2. Check memory before loading
    check_gpu_memory("Pre-load")
    
    # 3. Setup BitsAndBytes Config
    # Map compute dtype string to torch dtype
    compute_dtype_str = quant_settings.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = torch.bfloat16 if compute_dtype_str == "bfloat16" else torch.float16
    
    if not torch.cuda.is_available():
        print("[-] ERROR: CUDA is not available. 4-bit quantization requires a GPU. Aborting loading.")
        return False
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_settings.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_settings.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_settings.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    # 4. Load tokenizer and model in 4-bit
    try:
        print(f"[*] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Ensure pad token is configured
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"[*] Loading model in 4-bit (this can take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("[+] Model loaded successfully.")
        check_gpu_memory("Post-load")
        
        # 5. Apply LoRA Config
        print(f"[*] Applying PEFT/LoRA adapter...")
        lora_config = LoraConfig(
            r=peft_settings.get("r", 16),
            lora_alpha=peft_settings.get("lora_alpha", 32),
            target_modules=peft_settings.get("target_modules", []),
            lora_dropout=peft_settings.get("lora_dropout", 0.05),
            bias=peft_settings.get("bias", "none"),
            task_type=peft_settings.get("task_type", "CAUSAL_LM")
        )
        
        model = get_peft_model(model, lora_config)
        print("[+] LoRA Adapter applied successfully.")
        
        # Print trainable parameter percentage
        model.print_trainable_parameters()
        
        # Test basic forward pass dummy generation
        print("[*] Running quick forward pass verification...")
        test_input = tokenizer("Translate this message: Hello World!", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**test_input, max_new_tokens=10)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[+] Output verified successfully: '{decoded}'")
        
        # Clean memory for the next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print("[+] Cleaned up VRAM cache.")
        return True
        
    except Exception as e:
        print(f"[-] ERROR: Failed to load and verify model: {e}")
        # Make sure cache is cleared
        torch.cuda.empty_cache()
        return False

if __name__ == "__main__":
    # Test loading Qwen or Llama based on argument or default paths
    qwen_config = "configs/qwen_lora_config.json"
    llama_config = "configs/llama_lora_config.json"
    
    if os.path.exists(qwen_config):
        verify_quantized_load(qwen_config)
    if os.path.exists(llama_config):
        verify_quantized_load(llama_config)
