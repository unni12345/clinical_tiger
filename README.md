---

# 🐅 Clinical Tiger: Multimodal RAG with Verification for Product Monographs

A section-aware, multimodal Retrieval-Augmented Generation (RAG) system that extracts **text**, **images**, and **tables** from pharmaceutical product monographs (PDFs), stores them in a vector database (Qdrant), and supports **clinical assistant agents** that generate verified answers with **inline citations** and **image retrieval**.

---

## 🏗️ Architecture

![full_flow_clinical_tiger](https://github.com/user-attachments/assets/15da8e06-853b-4724-b985-bb6749046339)



---

## ⚙️ Setup Instructions

### 1. Clone and install

```bash
git clone https://github.com/yourname/clinical-tiger.git
cd clinical-tiger
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_key
MODEL_ID=emilyalsentzer/Bio_ClinicalBERT
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_key
```

---

## 🧠 Components

### `src/chunk_documents.py`
- Extracts `text`, `images`, and `tables` from PDFs using:
  - Sentence chunking via spaCy
  - Image captioning via OpenAI Vision (GPT-4o)
- Outputs to: `data/chunks.json`

### `src/embed_and_store.py`
- Embeds `text` chunks using ClinicalBERT
- Uploads vectors and metadata to Qdrant

### `src/retriever.py`
- Embeds queries and retrieves top-k chunks
- Supports filtering by `section`, `type`, and `source`
- Maps user intent to predefined monograph sections

### `src/tools.py`
- Tool functions used by LangChain agents:
  - Section classification
  - Drug identification
  - Retrieval (text, images)
  - Chain-of-verification (fact-checking answers)
  - Inline citation generation

### `src/rag_agent.py`
- Defines three agents:
  - `Agent`: retrieves and answers with citation
  - `VerifiedAgent`: runs additional factual verification
  - `ImageAgent`: retrieves image and caption based on query

### `streamlit_app.py`
- Frontend interface to interact with the agents.
- Modes:
  - Generate Answer
  - Generate + Verify
  - Retrieve Image

---

## 🚀 Running the System

### 1. Extract and Store Chunks
```bash
python src/chunk_documents.py
python src/embed_and_store.py
```

### 2. Run the App
```bash
streamlit run streamlit_app.py
```

---

## 🧪 Example Queries

- “What are the side effects of Lipitor in elderly patients?”
- “Show me any images related to Metformin dosage.”
- “Is there a section for drug interactions for Lipitor?”

---

## 🔍 Features

- ✅ Multimodal: Supports **text**, **images**, **(planned: tables)**
- ✅ Section-aware RAG with **token-bounded**, **sentence-preserving** chunking
- ✅ **Intent classification** for focused retrieval
- ✅ **Inline Vancouver-style citations**
- ✅ Optional **verification chain** for factual robustness
- ✅ User-friendly Streamlit frontend

---

## 📁 Project Structure

```
src/
├── chunk_documents.py       # PDF parser and chunker
├── embed_and_store.py       # Vector embedding + Qdrant upload
├── retriever.py             # Query embedding and filtering
├── tools.py                 # LangChain tools for agent logic
├── rag_agent.py             # Core agent workflows
├── streamlit_app.py         # UI app
data/
├── raw/                     # Raw PDFs
├── chunks.json              # Output of chunking
├── images/                  # Extracted images
```

---

## 📌 Citation Format

Inline:
> This medication may cause muscle pain and liver issues [1].

References:
```
Baheti et al. (2024) – Verification before answering: How Answer Verification Can Improve Factuality in Multi-Hop QA.
→ Basis for your chain-of-verification pipeline.
https://aclanthology.org/2024.findings-acl.212

Jaiswal et al. (2023) – RAGAS: An Evaluation System for Retrieval-Augmented Generation.
→ Informs future evaluation metrics for grounding, relevance, and answer faithfulness.
https://arxiv.org/abs/2309.02016

Alsentzer et al. (2019) – ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission.
→ Embedding model used for chunk representation and semantic retrieval.
https://arxiv.org/abs/1904.05342

Dettmers et al. (2023) – QLoRA: Efficient Finetuning of Quantized LLMs.
→ Supports your proposed personalization pathway using low-rank adapters.
https://arxiv.org/abs/2305.14314
```
---

## 🩺 UI

![ui-clinical-tigetr](https://github.com/user-attachments/assets/d3b2ca66-a6bc-4283-90d4-efb55000b617)


---

---

## 🩺 Disclaimer

This is an AI prototype intended for exploratory and educational use only. **Not intended for clinical decision making.**

---
