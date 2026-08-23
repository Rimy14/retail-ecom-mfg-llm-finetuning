import os
import json
import random

def generate_retail_qa():
    qa_list = []
    
    # Define vocabulary and templates for retail issues
    order_ids = [f"ORD-{random.randint(10000, 99999)}" for _ in range(100)]
    tracking_statuses = ["in transit", "delayed at sorting facility", "customs clearance delay", "out for delivery"]
    refund_reasons = ["damaged during shipping", "wrong size delivered", "defective item", "not as described"]
    payment_methods = ["Credit Card", "PayPal", "Apple Pay", "Google Pay"]
    carrier_names = ["FedEx", "DHL", "UPS", "USPS"]
    
    # 1. Shipping & Transit Delays
    for i in range(70):
        oid = random.choice(order_ids)
        status = random.choice(tracking_statuses)
        carrier = random.choice(carrier_names)
        
        q_templates = [
            f"My order {oid} is taking too long. Can you track it?",
            f"Why is my package {oid} delayed? It has been in transit for days.",
            f"I checked my tracking for {oid} and it says '{status}' via {carrier}. What does this mean?",
            f"Where is my package {oid}? The carrier is {carrier} and I haven't received it.",
            f"Can you get an update on order {oid}? It says '{status}'."
        ]
        
        a_templates = [
            f"I sincerely apologize for the delay with your order {oid}. Looking at the tracking data via {carrier}, the shipment is currently {status}. We are contacting the carrier to expedite delivery and will keep you updated.",
            f"We are sorry for the inconvenience. Your package {oid} is currently experiencing a {status}. This is usually resolved within 48 hours. We are monitoring it closely for you.",
            f"Thank you for contacting customer support. Order {oid} has a status of '{status}' at the {carrier} facility. We expect it to resume transit by tomorrow. Please let us know if it does not arrive by Friday."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "category": "shipping_delay",
            "intent": "track_order"
        })
        
    # 2. Refund Disputes & Damage
    for i in range(70):
        oid = random.choice(order_ids)
        reason = random.choice(refund_reasons)
        method = random.choice(payment_methods)
        
        q_templates = [
            f"I received my order {oid} but the item is {reason}. I want a full refund.",
            f"Can I get a refund for {oid}? The product was {reason}.",
            f"My order {oid} was delivered but it's {reason}. How do I return it and get my money back?",
            f"I want to request a return on {oid} because of a {reason}.",
            f"The package {oid} arrived damaged. Can you process a refund back to my {method}?"
        ]
        
        a_templates = [
            f"I am very sorry to hear that your order {oid} arrived {reason}. We will gladly process a full refund to your original payment method ({method}). I have also emailed you a prepaid shipping label so you can return the item.",
            f"We apologize for the defective item in order {oid}. A return request has been authorized for the reason: '{reason}'. Your refund of the purchase amount will be processed back to your {method} as soon as we scan the package.",
            f"That is certainly not the experience we want you to have. I have initiated a refund for order {oid} because it was {reason}. The funds should appear on your {method} statement within 3-5 business days."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "category": "refund_request",
            "intent": "dispute_refund"
        })
        
    # 3. Order Cancellations & Modifications
    for i in range(60):
        oid = random.choice(order_ids)
        
        q_templates = [
            f"I need to cancel my order {oid} immediately before it ships.",
            f"Can I change the shipping address on my order {oid}?",
            f"I accidentally ordered the wrong size for order {oid}. Can we update it?",
            f"Can you stop order {oid}? I changed my mind.",
            f"Please cancel order {oid} and refund my card."
        ]
        
        a_templates = [
            f"I have checked order {oid} and since it has not left our warehouse yet, I was able to successfully cancel it. A confirmation email has been sent and a full refund is being processed.",
            f"I have successfully updated the shipping address for your order {oid} as requested. Please check your email for the updated order confirmation details.",
            f"We have updated the item details for order {oid} in our system. You will receive a new confirmation email shortly reflecting these changes."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "category": "order_modification",
            "intent": "cancel_order"
        })
        
    # 4. Custom Orders & Requests
    for i in range(60):
        oid = random.choice(order_ids)
        
        q_templates = [
            f"Do you accept custom branding on orders over 100 units?",
            f"Can I add engraving or customization to my order {oid}?",
            f"What is the lead time for customized bulk retail shipments?",
            f"Is it possible to request gift wrapping and a custom note for order {oid}?"
        ]
        
        a_templates = [
            f"Yes, we offer custom branding options (including logo printing) for corporate and bulk orders over 100 units. Please email our sales team at bulk@retailcorp.com with your specifications.",
            f"Certainly! I have added your requested engraving details to order {oid}. This customization adds approximately 24 hours of processing time before shipping.",
            f"For bulk customized shipments, the standard production lead time is 10-14 business days after logo approval, followed by 3-5 days of transit time via FedEx."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "category": "customization",
            "intent": "custom_request"
        })
        
    return qa_list

def generate_manufacturing_qa():
    qa_list = []
    
    # Define vocabulary and templates for manufacturing processes
    dmaic_phases = ["Define", "Measure", "Analyze", "Improve", "Control"]
    spc_charts = ["X-bar chart", "R-chart", "p-chart", "c-chart"]
    wastes = ["Overproduction", "Waiting", "Transport", "Overprocessing", "Inventory", "Motion", "Defects"]
    control_actions = ["quarantine the batch", "re-calibrate the machine sensors", "perform a root cause analysis", "halt the assembly line"]
    
    # 1. Lean Six Sigma & DMAIC Operations
    for i in range(100):
        phase = random.choice(dmaic_phases)
        waste = random.choice(wastes)
        
        q_templates = [
            f"What is the main goal of the {phase} phase in a Lean Six Sigma DMAIC project?",
            f"How do we identify and eliminate '{waste}' waste in manufacturing assembly lines?",
            f"Which tools are commonly used during the {phase} stage of process operations?",
            f"Can you explain how eliminating the '{waste}' waste improves our Lean Six Sigma score?"
        ]
        
        a_templates = [
            f"The primary goal of the {phase} phase is to establish a clear structure for process improvement. In operations, this involves mapping inputs/outputs, verifying measurement systems, and stabilizing variation. For instance, focusing on reducing '{waste}' directly optimizes throughput.",
            f"To systematically address '{waste}' waste, we utilize Value Stream Mapping (VSM) and Standardized Work instructions. During the {phase} phase, we gather cycle-time data to isolate non-value-added activities and implement 5S practices to stabilize the manufacturing floor."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "domain": "lean_six_sigma"
        })
        
    # 2. Quality Control & SPC charts
    for i in range(100):
        chart = random.choice(spc_charts)
        action = random.choice(control_actions)
        
        q_templates = [
            f"What should the floor supervisor do if an alert points to an out-of-control point on the {chart}?",
            f"How does an operator determine if variation on the {chart} is due to common cause or assignable cause?",
            f"What standard operating procedure is triggered when the {chart} violates Western Electric rules?",
            f"How often should we calibrate our sensors to prevent false alarms on the {chart}?"
        ]
        
        a_templates = [
            f"When a data point falls outside the control limits on the {chart}, the supervisor must immediately {action}. An assignable cause variation must be investigated using a Fishbone diagram and process logs.",
            f"An out-of-control signal on the {chart} implies assignable cause variation. The operator is trained to {action} immediately and document the corrective action in the SPC quality logbook."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "domain": "statistical_process_control"
        })
        
    # 3. Root Cause Analysis (RCA) & Troubleshooting
    for i in range(60):
        action = random.choice(control_actions)
        
        q_templates = [
            "How do we apply the '5 Whys' methodology to isolate a machinery defect?",
            "What is the difference between corrective action and preventive action (CAPA) in quality audits?",
            f"If a component tolerance check fails, what is the sequence of troubleshooting steps to take?",
            "How do we construct a Pareto chart to prioritize quality defect mitigation?"
        ]
        
        a_templates = [
            f"The '5 Whys' begins with the defect symptom and drills down to the mechanical or procedural root cause. Once the root cause is isolated, we implement a permanent corrective action, such as to {action}.",
            f"Corrective action addresses an existing process failure, requiring the line team to {action}. Preventive action addresses systemic issues to stop future failure modes before they occur."
        ]
        
        qa_list.append({
            "instruction": random.choice(q_templates),
            "response": random.choice(a_templates),
            "domain": "root_cause_analysis"
        })
        
    return qa_list

def main():
    print("[*] Generating synthetic domain-specific QA expansion pairs...")
    
    retail_synthetic = generate_retail_qa()
    mfg_synthetic = generate_manufacturing_qa()
    
    all_synthetic = retail_synthetic + mfg_synthetic
    print(f"[+] Generated {len(retail_synthetic)} Retail/Ecom samples.")
    print(f"[+] Generated {len(mfg_synthetic)} Manufacturing samples.")
    print(f"[+] Total Synthetic QA pairs: {len(all_synthetic)}")
    
    # Save synthetic records
    synthetic_file = "data/processed/synthetic_qa.json"
    os.makedirs(os.path.dirname(synthetic_file), exist_ok=True)
    with open(synthetic_file, "w", encoding="utf-8") as f:
        json.dump(all_synthetic, f, ensure_ascii=False, indent=2)
    print(f"[+] Synthetic data saved to {synthetic_file}")
    
    # Merge with original train.json
    train_file = "data/processed/train.json"
    train_v2_file = "data/processed/train_v2.json"
    
    if os.path.exists(train_file):
        print(f"[*] Loading original train dataset from {train_file}...")
        with open(train_file, "r", encoding="utf-8") as f:
            original_train = json.load(f)
            
        print(f"[*] Merging {len(original_train)} original records with {len(all_synthetic)} synthetic records...")
        train_v2 = original_train + all_synthetic
        
        # Shuffle train_v2 for model training distribution
        random.shuffle(train_v2)
        
        with open(train_v2_file, "w", encoding="utf-8") as f:
            json.dump(train_v2, f, ensure_ascii=False, indent=2)
            
        print(f"[+] train_v2.json successfully compiled and saved to {train_v2_file}")
        print(f"    - Total train_v2 records: {len(train_v2)}")
    else:
        print(f"[!] Warning: Original train file {train_file} not found. Cannot merge. Creating standalone train_v2.json.")
        with open(train_v2_file, "w", encoding="utf-8") as f:
            json.dump(all_synthetic, f, ensure_ascii=False, indent=2)
            
    # Write configs/qwen_lora_config_v2.json &configs/llama_lora_config_v2.json
    print("[*] Creating configs for v2 fine-tuning runs...")
    
    qwen_v2_cfg = {
        "model_type": "qwen",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/qwen_v2",
        "peft_settings": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "bnb_4bit_compute_dtype": "float16"
    }
    
    llama_v2_cfg = {
        "model_type": "llama",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "output_dir": "models/llama_v2",
        "peft_settings": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "bnb_4bit_compute_dtype": "float16"
    }
    
    with open("configs/qwen_lora_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(qwen_v2_cfg, f, indent=2)
    with open("configs/llama_lora_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(llama_v2_cfg, f, indent=2)
        
    print("[+] v2 configuration files written successfully.")

if __name__ == "__main__":
    main()
