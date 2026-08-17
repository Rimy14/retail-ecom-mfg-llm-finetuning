# Retail, E-commerce, and Manufacturing LLM Dataset Preparation

## Purpose
The final goal of this project is to build specialized AI assistants for the Retail, E-commerce, and Manufacturing sectors. To achieve this, we are:
1. **Fine-Tuning**: Training Qwen and Llama models on our collected datasets so they understand specific industry language and customer support queries.
2. **Adding Grounded Knowledge (RAG)**: Connecting the models to a search database (ChromaDB) containing industrial manuals and documents, so the AI can retrieve facts and answer questions without guessing.
3. **Final Delivery**: Providing a high-accuracy, reliable chatbot assistant ready to be deployed for company use.

## Project Overview
This project contains the pipeline and files for preparing domain-specific datasets to fine-tune **Qwen-RetailEcomManufacturing** and **Llama-RetailEcomManufacturing** models.

*   **Industry Focus**: Retail, E-commerce, and Manufacturing
*   **Target Architectures**: Qwen & Llama
*   **Current Phase**: Dataset Setup and Collection

---

## Project Structure
```
Retail/
├── data/
│   ├── raw/
│   │   ├── retail_ecommerce_raw.json
│   │   └── manufacturing_raw.json
│   └── processed/
├── notebooks/
│   └── setup_and_data_collection.ipynb
├── src/
│   ├── __init__.py
│   └── data_collection.py
├── models/
├── .gitignore
└── README.md
```

---

## Collected Domain Datasets

1. **Retail & E-commerce Customer Support**
   - **Source ID**: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
   - **Records**: 26,872 items
   - **Features**: `instruction`, `response`, `category`, `intent`
   - **Purpose**: Conversational QA for retail inquiries, order status, returns, and payment issues.

2. **Manufacturing Quality & Process Operations**
   - **Source ID**: `cw18/lean-six-sigma-qna-v1`
   - **Records**: 102 items
   - **Features**: `instruction`, `response`, `domain`
   - **Purpose**: Industry QA on Lean Six Sigma methodology, manufacturing defect controls, and operational efficiency.

---

## Environment & Requirements
The training and preparation pipeline runs on **Google Colab** with GPU runtime acceleration.

**Core Dependencies**:
*   `transformers` (Model loading & tokenization)
*   `datasets` (Hugging Face datasets handler)
*   `accelerate` & `bitsandbytes` (4-bit & 8-bit quantization support)
*   `peft` (Parameter-Efficient Fine-Tuning/LoRA)
*   `pytorch` (Base tensor operations)
*   `wandb` & `chromadb` (Metric logs & vector store backup)



## Future Work
1. **Day 2**: Run data cleaning pipelines, split data, and write LoRA configurations (Rank 16, Alpha 32).
2. **Day 3**: Write training pipelines, quantized model loading checks, and connect to Weights & Biases (W&B).
3. **Day 4**: Execute simultaneous fine-tuning of Qwen and Llama models (v1 checkpoints).
4. **Day 5-8**: Evaluate v1 models, generate synthetic QA expansion files, fine-tune v2/v3 models.
5. **Day 9-10**: Set up ChromaDB, index domain manuals, and train v4 models (RAG-Aware).
6. **Day 11-12**: Run final comparison benchmarks and generate master evaluation report.

---

## Notes
*   **Data Integrity**: Do not edit raw files under `data/raw/` directly. Any modifications must create a separate output file under `data/processed/`.
*   **Licensing**: Review dataset licenses on Hugging Face before using for commercial purposes.
