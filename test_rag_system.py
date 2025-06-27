import os
from openai import OpenAI
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# File paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(CURRENT_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "reviews_faiss_index.pkl")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "reviews_metadata.pkl")

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o-mini"  # Can be changed to gpt-4o or other preferred model

def load_index_and_metadata():
    """Load FAISS index and metadata from disk."""
    try:
        with open(FAISS_INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"Loaded FAISS index with {index.ntotal} vectors.")
        return index, metadata
    except FileNotFoundError:
        print(f"Error: Index files not found. Please run embedding_generator.py first.")
        exit(1)

def embed_text(text: str) -> Optional[np.ndarray]:
    """Generate embedding for a given text using OpenAI's API."""
    if not text or text.strip() == "":
        return None
    
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

def normalize_for_search(text: str) -> str:
    """Normalize text for better search results."""
    # Convert to lowercase
    text = text.lower()
    
    # Handle specific keyword normalizations
    normalizations = {
        r"obenan": "obenan",
        r"omnipulse": "omnipulse",
        r"erhan seven": "erhan seven",
    }
    
    for pattern, replacement in normalizations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def retrieve_relevant_documents(query: str, index, metadata: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top_k most relevant documents for the query."""
    # Normalize the query for better matching
    normalized_query = normalize_for_search(query)
    
    # Embed the query
    query_embedding = embed_text(normalized_query)
    if query_embedding is None:
        print("Failed to generate embedding for the query.")
        return []
    
    # Reshape for FAISS
    query_embedding = np.array([query_embedding])
    
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get the corresponding metadata
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1:  # -1 means no match was found
            result = metadata[idx].copy()
            result["distance"] = float(distance)  # Convert numpy float to Python float
            results.append(result)
    
    return results

def prepare_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved documents as context for the LLM."""
    if not results:
        return "No relevant reviews found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        context_part = f"---\nREVIEW #{i}:\n"
        
        # Add combined text if available (already formatted nicely)
        if "combined_text" in result:
            context_part += result["combined_text"]
        else:
            # Otherwise, construct from original_row
            orig = result.get("original_row", {})
            context_part += f"Reviewer: {orig.get('reviewerTitle', 'Unknown')}\n"
            context_part += f"Rating: {orig.get('ratingValue', 'N/A')}\n"
            if "ratingText" in orig and orig["ratingText"]:
                context_part += f"Review: {orig['ratingText']}\n"
            if "reviewReply" in orig and orig["reviewReply"]:
                context_part += f"Reply: {orig['reviewReply']}\n"
            if "date" in orig:
                context_part += f"Date: {orig['date']}\n"
        
        # Add match score
        context_part += f"\nRelevance Score: {1.0 / (1.0 + result.get('distance', 0)):.4f}\n"
        
        context_parts.append(context_part)
    
    return "\n".join(context_parts)

def refine_with_llm(query: str, context: str, model: str = LLM_MODEL) -> str:
    """Use LLM to generate a refined answer based on retrieved context."""
    system_prompt = """You are an expert research assistant specializing in review analysis.
Your task is to provide accurate, comprehensive answers based ONLY on the review context provided.
All answers should be backed by the specific reviews mentioned in the context.
If the query cannot be adequately answered by the provided reviews, acknowledge this limitation clearly.

IMPORTANT GUIDELINES:
1. NEVER make up or hallucinate information not present in the reviews.
2. If reviews contain contradictory information, acknowledge the different perspectives.
3. Use exact quotes from the reviews when directly referencing content.
4. Cite specific review numbers (e.g., "According to Review #2...") when answering.
5. Apply critical thinking to assess the review quality and relevance.
6. Note the relevance score of each review - higher scores indicate greater relevance.

SPECIAL TERM HANDLING:
- Normalize any mentions of "Obenan" including misspellings or phonetic variants.
- Do the same for "Omnipulse" and "Erhan Seven".

FORMAT YOUR RESPONSE:
- Start with a direct answer to the query.
- Include supporting evidence from the reviews.
- If relevant, note the general sentiment, consensus, or trends across reviews.
- Be professional, clear, and concise."""

    user_prompt = f"""QUERY: {query}

CONTEXT FROM RETRIEVED REVIEWS:
{context}

Based solely on the above context, please answer the query."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for more accurate responses
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: Failed to generate response. {str(e)}"

def process_query(query: str, top_k: int = 5):
    """Process a query and return the answer."""
    print(f"\n\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"{'=' * 80}")
    
    # Load index and metadata
    index, metadata = load_index_and_metadata()
    
    # Retrieve relevant documents
    print(f"\nRetrieving top {top_k} most relevant reviews...")
    results = retrieve_relevant_documents(query, index, metadata, top_k=top_k)
    
    if not results:
        print("No relevant results found.")
        return
    
    print(f"Found {len(results)} relevant reviews")
    
    # Prepare context
    context = prepare_context(results)
    print("\nContext prepared. Generating answer with LLM...")
    
    # Generate refined answer
    answer = refine_with_llm(query, context)
    
    print(f"\n{'=' * 80}")
    print("ANSWER:")
    print(f"{answer}")
    print(f"{'=' * 80}")
    return answer

def main():
    """Run test queries to evaluate the RAG system."""
    print("World-Class RAG System Test")
    print("===========================")
    
    test_queries = [
        "What are the most common complaints in the reviews?",
        "Are there any reviews about customer service?",
        "What's the average rating of these reviews?",
        "Find reviews that mention response time",
        "What do customers like most about the service?",
        "Are there any negative reviews about pricing?",
        # Add more test queries here
    ]
    
    for query in test_queries:
        process_query(query)

if __name__ == "__main__":
    main()
