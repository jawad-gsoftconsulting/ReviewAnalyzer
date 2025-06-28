import os
import json
import pickle
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# ─── Configuration ─────────────────────────────────────────────────────────────

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# File paths
CURRENT_DIR         = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV           = os.path.join(CURRENT_DIR, "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/ReviewAnalyzerRagAda/RagAda/CleanReviews.csv")
OUTPUT_DIR          = os.path.join(CURRENT_DIR, "embeddings")
FAISS_INDEX_PATH    = os.path.join(OUTPUT_DIR, "reviews_faiss_index.index")
METADATA_PATH       = os.path.join(OUTPUT_DIR, "reviews_metadata.pkl")

# ─── Checkpoint files ──────────────────────────────────────────────────────────

CKPT_META_PATH      = os.path.join(OUTPUT_DIR, "metadata_checkpoint.pkl")
CKPT_EMBS_PATH      = os.path.join(OUTPUT_DIR, "embeddings_checkpoint.npy")

# Embedding model config
EMBEDDING_MODEL     = "text-embedding-ada-002"
EMBEDDING_DIM       = 1536  # as per text-embedding-ada-002

# ─── Helpers ───────────────────────────────────────────────────────────────────

def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def embed_text(text: str) -> Optional[np.ndarray]:
    """Generate embedding; return zero‐vector on empty, None on API failure."""
    if not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def create_combined_text(row: Dict[str, Any]) -> str:
    parts = []
    if row.get("reviewerTitle"):   parts.append(f"Reviewer: {row['reviewerTitle']}")
    if row.get("ratingValue"):     parts.append(f"Rating: {row['ratingValue']}")
    if row.get("ratingText"):      parts.append(f"Review: {row['ratingText']}")
    if row.get("reviewReply"):     parts.append(f"Reply: {row['reviewReply']}")
    if row.get("sentimentAnalysis"): parts.append(f"Sentiment: {row['sentimentAnalysis']}")
    if row.get("date"):            parts.append(f"Date: {row['date']}")
    if row.get("locationId"):      parts.append(f"Location ID: {row['locationId']}")
    return "\n".join(parts)

# ─── Main embedding + checkpoint logic ─────────────────────────────────────────

def generate_embeddings_and_index(df: pd.DataFrame):
    # 1) Try to load checkpoint if it exists
    if os.path.exists(CKPT_META_PATH) and os.path.exists(CKPT_EMBS_PATH):
        with open(CKPT_META_PATH, "rb") as f:
            metadata_list: List[Dict[str,Any]] = pickle.load(f)
        embeddings = np.load(CKPT_EMBS_PATH)
        start_idx = len(metadata_list)
        print(f"Resuming from checkpoint: {start_idx} rows already done.")
    else:
        metadata_list = []
        embeddings = None
        start_idx = 0
        print("No checkpoint found, starting from scratch.")

    # 2) Loop from start_idx to end
    total = len(df)
    for i in tqdm(range(start_idx, total), desc="Generating embeddings"):
        row = df.iloc[i].to_dict()
        combined = create_combined_text(row)
        emb = embed_text(combined)
        if emb is None:
            # skip this row if embedding failed
            continue

        # build metadata entry
        meta: Dict[str,Any] = {
            "rag_id":     row.get("rag_id", ""),
            "combined_text": combined,
            "original_row": row
        }
        # try parse full_review_json if present
        if row.get("full_review_json"):
            try:
                meta["full_review_json"] = json.loads(row["full_review_json"])
            except Exception:
                meta["full_review_json"] = row["full_review_json"]

        metadata_list.append(meta)

        # stack embedding
        if embeddings is None:
            embeddings = emb[np.newaxis, :]
        else:
            embeddings = np.vstack([embeddings, emb[np.newaxis, :]])

        # 3) checkpoint every 500 rows
        if (i + 1) % 50 == 0 or i == total - 1:
            with open(CKPT_META_PATH, "wb") as f:
                pickle.dump(metadata_list, f)
            np.save(CKPT_EMBS_PATH, embeddings)
            print(f"  → Checkpoint saved at row {i+1}/{total}")

    # 4) At end, write final outputs
    print("Finished embeddings. Building FAISS index…")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"  → FAISS index has {index.ntotal} vectors.")

    # save index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_list, f)
    print("FAISS index and metadata saved.")

    # 5) remove checkpoint files
    try:
        os.remove(CKPT_META_PATH)
        os.remove(CKPT_EMBS_PATH)
        print("Removed checkpoint files.")
    except OSError:
        pass

def main():
    create_output_dir()
    print(f"Reading CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} reviews from CSV.")
    generate_embeddings_and_index(df)
    print("Done.")

if __name__ == "__main__":
    main()
