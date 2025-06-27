import os
from openai import OpenAI
import pandas as pd
import numpy as np
import faiss
import pickle
import json
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Optional

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# File paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(CURRENT_DIR, "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/ReviewAnalyzerRagAda/RagAda/FinalReviews.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "reviews_faiss_index.pkl")
METADATA_PATH = os.path.join(OUTPUT_DIR, "reviews_metadata.pkl")

# Embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536  # Dimension of OpenAI's text-embedding-ada-002 embeddings

def create_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def embed_text(text: str) -> Optional[np.ndarray]:
    """Generate embedding for a given text using OpenAI's API."""
    if not text or text.strip() == "":
        # Return zero vector for empty text
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def create_combined_text(row: Dict[str, Any]) -> str:
    """Create a combined text field from relevant columns for embedding."""
    parts = []
    
    # Add reviewer name and title
    if row.get("reviewerTitle"):
        parts.append(f"Reviewer: {row['reviewerTitle']}")
    
    # Add rating value
    if row.get("ratingValue"):
        parts.append(f"Rating: {row['ratingValue']}")
    
    # Add review text (this is the most important part for semantic search)
    if row.get("ratingText") and str(row["ratingText"]).strip():
        parts.append(f"Review: {row['ratingText']}")
    
    # Add reply if available
    if row.get("reviewReply") and str(row["reviewReply"]).strip():
        parts.append(f"Reply: {row['reviewReply']}")
    
    # Add sentiment analysis if available
    if row.get("sentimentAnalysis") and str(row["sentimentAnalysis"]).strip():
        parts.append(f"Sentiment: {row['sentimentAnalysis']}")
    
    # Add date
    if row.get("date"):
        parts.append(f"Date: {row['date']}")
    
    # Add any other relevant fields (locationId, etc.)
    if row.get("locationId"):
        parts.append(f"Location ID: {row['locationId']}")
    
    # Join all parts with newlines
    return "\n".join(parts)

def generate_embeddings_and_index(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Generate embeddings for each row and create metadata."""
    metadata_list = []
    embeddings_list = []
    
    # Process each row with a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        # Convert row to dictionary for easier handling
        row_dict = row.to_dict()
        
        # Create combined text for embedding
        combined_text = create_combined_text(row_dict)
        
        # Generate embedding
        embedding = embed_text(combined_text)
        
        if embedding is not None:
            # Store metadata (we want to keep all fields for retrieval)
            metadata = {
                "rag_id": row_dict.get("rag_id", ""),
                "combined_text": combined_text,
                # Include all original fields for comprehensive retrieval
                "original_row": row_dict
            }
            
            # Try parsing the full_review_json field if it exists
            if "full_review_json" in row_dict and row_dict["full_review_json"]:
                try:
                    full_json = json.loads(row_dict["full_review_json"])
                    metadata["full_review_json"] = full_json
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse full_review_json for row with rag_id: {row_dict.get('rag_id')}")
                    metadata["full_review_json"] = row_dict.get("full_review_json")
            
            # Store metadata and embedding
            metadata_list.append(metadata)
            embeddings_list.append(embedding)
    
    # Stack embeddings into a single numpy array
    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
        return metadata_list, embeddings
    else:
        raise ValueError("No valid embeddings were generated.")

def main():
    """Main function to generate embeddings and create FAISS index."""
    # Create output directory
    create_output_dir()
    
    # Read CSV file
    print(f"Reading CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} reviews from CSV.")
    
    # Generate embeddings and metadata
    metadata_list, embeddings = generate_embeddings_and_index(df)
    
    # Create FAISS index (using L2 distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}.")
    
    # Save FAISS index and metadata
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_list, f)
    
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("Ready to use with retriever_responder.py!")

if __name__ == "__main__":
    main()
