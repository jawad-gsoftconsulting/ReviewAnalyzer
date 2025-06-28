#!/usr/bin/env python3
import os
import json
import pickle
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in your .env")

CURRENT_DIR       = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV         = os.path.join(
    "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/ReviewAnalyzerRagAda/RagAda",
    "FinalReviews.csv"
)
OUTPUT_DIR        = os.path.join(CURRENT_DIR, "embeddings")
FAISS_INDEX_PATH  = os.path.join(OUTPUT_DIR, "reviews_faiss_index.index")
METADATA_PATH     = os.path.join(OUTPUT_DIR, "reviews_metadata.pkl")

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM   = 1536

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── LOGGER SETUP ────────────────────────────────────────────────────────────

logger = logging.getLogger("embedder")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

def embed_text(text: str) -> Optional[np.ndarray]:
    if not text.strip():
        return None
    try:
        resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return None

def create_combined_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    def add_if_str(field: str, label: str):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(f"{label}: {val.strip()}")

    # reviewerTitle and ratingValue can be non-strings but safe:
    if row.get("reviewerTitle"):
        parts.append(f"Reviewer: {row['reviewerTitle']}")
    if row.get("ratingValue") is not None:
        parts.append(f"Rating: {row['ratingValue']}")

    # only text fields via add_if_str
    add_if_str("ratingText",       "Review")
    add_if_str("reviewReply",      "Reply")
    add_if_str("sentimentAnalysis","Sentiment")

    # date & locationId
    if row.get("date"):
        parts.append(f"Date: {row['date']}")
    if row.get("locationId"):
        parts.append(f"Location ID: {row['locationId']}")

    return "\n".join(parts)

# ─── CORE ────────────────────────────────────────────────────────────────────

def generate_embeddings_and_index(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    total_rows = len(df)
    mask_has_text = (
        df["ratingText"].notnull()
        & df["ratingText"].astype(str).str.strip().astype(bool)
    )
    no_text_count = total_rows - mask_has_text.sum()
    logger.info(f"{no_text_count}/{total_rows} reviews have no ratingText → will be skipped.")

    skipped_no_text_ids: List[str] = df.loc[~mask_has_text, "rag_id"].astype(str).tolist()
    skipped_error_ids:   List[str] = []
    metadata_list:       List[Dict[str, Any]] = []
    embeddings_list:     List[np.ndarray]    = []

    to_process = df.loc[mask_has_text]
    logger.info(f"Starting embedding on {len(to_process)} reviews with text...")

    pbar = tqdm(
        to_process.iterrows(),
        total=len(to_process),
        desc="Embedding reviews",
        unit="rev"
    )
    for _, row in pbar:
        rag_id = str(row.get("rag_id", "<no-id>"))
        combined = create_combined_text(row.to_dict())
        if not combined.strip():
            logger.debug(f"[{rag_id}] combined_text empty → skipping")
            skipped_no_text_ids.append(rag_id)
            continue

        emb = embed_text(combined)
        if emb is None:
            logger.debug(f"[{rag_id}] embed_text returned None → skipping")
            skipped_error_ids.append(rag_id)
            continue

        # metadata
        meta: Dict[str, Any] = {
            "rag_id":        rag_id,
            "combined_text": combined,
            "original_row":  row.to_dict()
        }
        frj = row.get("full_review_json")
        if frj:
            try:
                meta["full_review_json"] = json.loads(frj) if isinstance(frj, str) else frj
            except json.JSONDecodeError:
                logger.warning(f"[{rag_id}] full_review_json parse failed")
                meta["full_review_json"] = frj

        metadata_list.append(meta)
        embeddings_list.append(emb)

        pbar.set_postfix({
            "skips_no_text": len(skipped_no_text_ids),
            "skips_error":   len(skipped_error_ids),
            "embedded":      len(embeddings_list)
        })
    pbar.close()

    logger.info(f"Done. Embedded {len(metadata_list)} vectors successfully.")
    logger.info(f"Skipped_no_text: {len(skipped_no_text_ids)} | Skipped_errors: {len(skipped_error_ids)}")
    if skipped_no_text_ids:
        logger.debug("First 10 skipped_no_text IDs: " + ", ".join(skipped_no_text_ids[:10]))
    if skipped_error_ids:
        logger.debug("First 10 skipped_error IDs:  " + ", ".join(skipped_error_ids[:10]))

    if not embeddings_list:
        raise RuntimeError("0 embeddings generated—check your CSV and ratingText fields!")

    return metadata_list, np.vstack(embeddings_list)

def main():
    create_output_dir()
    logger.info(f"Loading CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Total rows in CSV: {len(df)}")

    metadata, embeddings = generate_embeddings_and_index(df)

    # Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors (dim={dim})")

    # Save index & metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved FAISS index → {FAISS_INDEX_PATH}")
    logger.info(f"Saved metadata pickle → {METADATA_PATH}")

if __name__ == "__main__":
    main()
