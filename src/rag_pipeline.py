import os
import re
import glob
import torch
import chromadb
from chromadb.utils import embedding_functions

class RetailMfgRAGPipeline:
    """
    Production-grade RAG pipeline using ChromaDB for vector retrieval
    and fine-tuned QLoRA models (Llama-3 / Qwen) for domain-grounded generation.
    """
    
    def __init__(self, persist_dir="data/chroma_db", collection_name="retail_mfg_knowledge", embedding_model="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        
        print(f"[*] Initializing ChromaDB Client at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Initialize Sentence Transformer Embedding Function
        print(f"[*] Loading Embedding Model: {embedding_model}...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[+] Collection '{self.collection_name}' ready. Current indexed documents: {self.collection.count()}")

    def chunk_markdown_document(self, file_path):
        """
        Parses and chunks markdown documents by sections and bullet points.
        """
        chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        file_basename = os.path.basename(file_path)
        domain = "manufacturing" if "manufacturing" in file_basename.lower() or "sop" in file_basename.lower() else "retail"
        
        # Split by sections (## Section)
        sections = re.split(r'\n(?=##\s+)', content)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
                
            lines = sec.split("\n")
            section_title = lines[0].replace("#", "").strip()
            
            # Split section into individual bullet items or paragraphs
            body_paragraphs = [p.strip() for p in sec.split("\n- ") if p.strip()]
            
            for idx, para in enumerate(body_paragraphs):
                # Clean header from the first bullet if attached
                if idx == 0 and "\n" in para:
                    para_parts = para.split("\n", 1)
                    if len(para_parts) > 1 and para_parts[1].strip():
                        para = para_parts[1].strip()
                        
                clean_text = para.replace("- ", "").replace("**", "").strip()
                if len(clean_text) > 30:
                    prefix = "Process Operations Manual: " if domain == "manufacturing" else "Customer Support Policy Guide: "
                    formatted_chunk = f"{prefix}{section_title} - {clean_text}"
                    
                    chunk_id = f"{file_basename}_{section_title[:20]}_{idx}".replace(" ", "_").replace("/", "_")
                    chunks.append({
                        "id": chunk_id,
                        "text": formatted_chunk,
                        "metadata": {
                            "source": file_basename,
                            "section": section_title,
                            "domain": domain
                        }
                    })
        return chunks

    def index_knowledge_directory(self, kb_dir="data/knowledge_base"):
        """
        Indexes all markdown and text documents from knowledge base directory into ChromaDB.
        """
        doc_files = glob.glob(os.path.join(kb_dir, "*.md")) + glob.glob(os.path.join(kb_dir, "*.txt"))
        if not doc_files:
            print(f"[!] Warning: No knowledge base files found in {kb_dir}")
            return 0
            
        all_chunks = []
        for fpath in doc_files:
            print(f"[*] Parsing knowledge file: {fpath}...")
            chunks = self.chunk_markdown_document(fpath)
            all_chunks.extend(chunks)
            
        if all_chunks:
            # Upsert into ChromaDB
            ids = [c["id"] for c in all_chunks]
            documents = [c["text"] for c in all_chunks]
            metadatas = [c["metadata"] for c in all_chunks]
            
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"[+] Successfully indexed {len(all_chunks)} knowledge passages into ChromaDB!")
            print(f"    - Total database records: {self.collection.count()}")
            
        return len(all_chunks)

    def retrieve_context(self, query, top_k=2):
        """
        Retrieves top_k most relevant knowledge passages for a given user query.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_docs = results["documents"][0] if results["documents"] else []
        retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
        
        combined_context = " ".join(retrieved_docs)
        return combined_context, retrieved_docs, retrieved_metas

    def generate_rag_response(self, model, tokenizer, query, top_k=2, max_new_tokens=150, temperature=0.7):
        """
        Retrieves relevant context and generates a grounded response using the fine-tuned LLM.
        """
        context_str, raw_docs, metadatas = self.retrieve_context(query, top_k=top_k)
        
        # Build Alpaca RAG Prompt
        prompt = (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{query}\n\n"
            f"### Context:\n{context_str}\n\n"
            f"### Response:\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generation_tokens = outputs[0][prompt_len:]
        response = tokenizer.decode(generation_tokens, skip_special_tokens=True).strip()
        
        return {
            "query": query,
            "context": context_str,
            "retrieved_passages": raw_docs,
            "sources": [m.get("source", "") for m in metadatas],
            "response": response
        }

if __name__ == "__main__":
    # Test indexing standalone
    pipeline = RetailMfgRAGPipeline()
    pipeline.index_knowledge_directory("data/knowledge_base")
    
    sample_query = "What is the return window and refund time for damaged items?"
    ctx, docs, _ = pipeline.retrieve_context(sample_query, top_k=2)
    print(f"\n[*] Sample Query: {sample_query}")
    print(f"[*] Retrieved Context: {ctx}")
