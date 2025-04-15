# src/chunk_documents.py
"""Section‑aware, sentence‑preserving, token‑bounded chunker **with multimodal support**.

• Extracts **text**, **images**, and **tables** from PDF product monographs.
• Adds a mandatory `type` field to every chunk: `"text" | "image" | "table"`.
• Text chunks are created by:
  1. Detecting SECTION headers (all‑caps lines)
  2. Splitting into sentences (spaCy)
  3. Packing sentences until a token limit is reached, with sentence overlap
• Image chunks are extracted with PyMuPDF and stored as PNG files in `data/images/`.
• Table chunks are detected by layout and extracted with `get_text("dict")` from PyMuPDF.

Result is written to `data/chunks.json` and ready for embedding.
"""

from __future__ import annotations
import os, re, json, uuid, pathlib, base64
from typing import List, Dict
from dotenv import load_dotenv
import fitz  # PyMuPDF
import spacy
from transformers import AutoTokenizer
import openai
from PIL import Image
from openai import OpenAI
import camelot
import base64

# ---------- configuration ---------- #
# Load environment variables
load_dotenv()

MAX_TOK = 400
OVERLAP_SENT = 2
SECTION_RE = re.compile(r"\n([A-Z][A-Z \d\-]{3,})\n")
MODEL_ID = os.getenv("MODEL_ID")
IMAGES_DIR = pathlib.Path("data/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
# Create output directory for CSV files if it doesn't exist.
EXTRACTED_TABLES_DIR = pathlib.Path("data/extracted_tables")
EXTRACTED_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of source names to their corresponding PDF URLs.
SOURCE_URLS = {
    "lipitor": "https://pdf.hres.ca/dpd_pm/00048312.PDF",
    "metformin": "https://pdf.hres.ca/dpd_pm/00021412.PDF"
}

# ---------- NLP init ---------- #
_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
_nlp.add_pipe("sentencizer")
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- helpers ---------- #

def _sentences(text: str) -> List[str]:
    return [s.text.strip() for s in _nlp(text).sents if s.text.strip()]

def _token_len(txt: str) -> int:
    return len(_tok.encode(txt, add_special_tokens=False))

def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def generate_openai_caption(image_path):
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate a clinical-style caption for this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ],
            }
        ],
        max_tokens=250,
    )
    
    return response.choices[0].message.content

# ---------- text chunking ---------- #

def _pack_sentences(sents: List[str], section: str, src: str, page: int) -> List[Dict]:
    chunks, buf, buf_tok = [], [], 0
    for sent in sents:
        stoks = _token_len(sent)
        if buf and buf_tok + stoks > MAX_TOK:
            chunks.append(_flush(buf, section, src, page))
            buf = buf[-OVERLAP_SENT:]
            buf_tok = sum(_token_len(s) for s in buf)
        buf.append(sent)
        buf_tok += stoks
    if buf:
        chunks.append(_flush(buf, section, src, page))
    return chunks

def _flush(sentence_buffer: List[str], section: str, src: str, page: int) -> Dict:
    text = " ".join(sentence_buffer)
    return {
        "type": "text",
        "text": text,
        "section": section,
        "page": page,
        "source": src,
        "url": SOURCE_URLS.get(src.lower(), ""),
        "chunk_id": _make_id(f"{src}_p{page}_{section[:15].replace(' ', '_')}"),
        "token_len": _token_len(text),
    }

# ---------- image extraction with captioning ---------- #

def _extract_images(page: fitz.Page, src: str, page_no: int) -> List[Dict]:
    chunks = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = page.parent.extract_image(xref)
        img_bytes = pix["image"]
        img_ext = pix.get("ext", "png")
        img_name = f"{src}_p{page_no}_{img_index}.{img_ext}"
        img_path = IMAGES_DIR / img_name
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        caption = generate_openai_caption(str(img_path))
        chunks.append({
            "type": "image",
            "image_path": str(img_path),
            "text": caption,
            "section": "IMAGE",
            "page": page_no,
            "source": src,
            "url": SOURCE_URLS.get(src.lower(), ""),
            "chunk_id": _make_id(f"{src}_p{page_no}_IMG{img_index}"),
        })
    return chunks

# ---------- table extraction using layout ---------- #
def extract_table_csv_from_image(img_path: str) -> str:
    """
    Uses GPT-4o's vision capabilities to extract a table as CSV text from an image.
    """
    with open(img_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the entire table shown in the image and output only the CSV text."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ]
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------- main extraction per PDF ---------- #

def extract_chunks_from_pdf(filepath: str, source_name: str) -> List[Dict]:
    doc = fitz.open(filepath)
    all_chunks: List[Dict] = []
    for page_num, page in enumerate(doc, start=1):
        all_chunks.extend(_extract_images(page, source_name, page_num))
        # Use the Camelot-based table extractor, passing the file path
        # all_chunks.extend(_extract_tables(filepath, source_name, page_num))
    
        raw = page.get_text("text")
        if not raw.strip():
            continue
    
        parts = SECTION_RE.split(raw)
        preamble = parts[0]
        if preamble.strip():
            all_chunks.extend(_pack_sentences(_sentences(preamble), "UNSECTIONED", source_name, page_num))
        for sec, body in zip(parts[1::2], parts[2::2]):
            all_chunks.extend(_pack_sentences(_sentences(body), sec.strip(), source_name, page_num))
    return all_chunks


# ---------- utility ---------- #

def save_chunks_to_json(chunks: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

# ---------- CLI ---------- #
if __name__ == "__main__":
    docs_dir = "data/raw"
    files = ["lipitor.pdf", "metformin.pdf"]
    output_path = "data/chunks.json"

    all_chunks: List[Dict] = []
    for idx, file in enumerate(files, start=1):
        print(f"📄 Processing {file} ({idx}/{len(files)}) …")
        fp = os.path.join(docs_dir, file)
        chunks = extract_chunks_from_pdf(fp, source_name=file.replace(".pdf", ""))
        all_chunks.extend(chunks)

    save_chunks_to_json(all_chunks, output_path)
    print(f"✅ Saved {len(all_chunks)} chunks (text + images + tables) → {output_path}")
