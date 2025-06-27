import os
from openai import OpenAI
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv
import json
import re
import datetime
from dateutil import parser as date_parser

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# LLM model to use for parsing queries (faster and cheaper than the main completion model)
QUERY_PARSER_MODEL = "gpt-3.5-turbo-0125"  # Fast model for parsing

def parse_query_with_llm(query: str) -> Dict[str, Any]:
    """
    Use LLM to extract structured filters and the semantic core of the query.
    Returns a dictionary with the extracted information.
    Handles relative date references and date ranges.
    """
    try:
        # Get current date for relative date understanding
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        Extract structured filters from the following query.
        TODAY'S DATE IS: {current_date}
        
        Analyze the query: "{query}"
        
        Extract any dates or date ranges mentioned (convert relative dates to absolute dates using the current date).
        Extract any numeric ratings mentioned.
        Identify if this is a COUNT query (asking how many) or a RETRIEVAL query (show me, find, etc.).
        Identify any sentiment filtering (positive, negative, neutral).
        Extract any mention of the number of results the user wants to see (e.g., "top 10", "first 20", "all", etc.)
        
        Return a JSON object with these fields:
        - core_query: The query with filter information removed (just the core semantic search intent).
        - query_type: Either "count" (if asking for quantity/how many) or "retrieval" (if asking for content).
        - date_start: The start date of any date range in YYYY-MM-DD format, if present.
        - date_end: The end date of any date range in YYYY-MM-DD format, if present.
        - rating: Any numeric rating mentioned (e.g., 1-5), if present.
        - sentiment: "positive", "negative", or "neutral" if specified in the query.
        - max_results: The number of results requested (as an integer). If "all" is specified, use 100. If not specified, omit this field.
        
        If any field is not present in the query, omit it from the JSON.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured data from queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content
            print(f"LLM parsing response: {response_content}")
            
            # Extract the JSON object
            try:
                # Find JSON object in the response
                match = re.search(r'\{[^\{\}]*\"core_query\"[^\{\}]*\}', response_content)
                if match:
                    json_str = match.group(0)
                    parsed_data = json.loads(json_str)
                    
                    # Ensure query_type is set with a default if not present
                    if "query_type" not in parsed_data:
                        # Detect if it's likely a count query based on basic keyword matching
                        if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]):
                            parsed_data["query_type"] = "count"
                        else:
                            parsed_data["query_type"] = "retrieval"
                            
                    return parsed_data
                else:
                    print("No JSON object found in the LLM response")
                    # Basic fallback detection of query type
                    query_type = "count" if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]) else "retrieval"
                    return {"core_query": query, "query_type": query_type}
            except Exception as e:
                print(f"Error parsing JSON from LLM response: {e}")
                # Basic fallback detection of query type
                query_type = "count" if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]) else "retrieval"
                return {"core_query": query, "query_type": query_type}
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Basic fallback detection of query type
            query_type = "count" if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]) else "retrieval"
            return {"core_query": query, "query_type": query_type}
    except Exception as e:
        print(f"Error in parse_query_with_llm: {e}")
        # Basic fallback detection of query type
        query_type = "count" if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]) else "retrieval"
        return {"core_query": query, "query_type": query_type}
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Basic fallback detection of query type
        query_type = "count" if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]) else "retrieval"
        return {"core_query": query, "query_type": query_type}

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
    # For example, normalize company names, locations, etc.
    normalizations = {
        r"obenan": "obenan",
        r"omnipulse": "omnipulse",
        r"erhan seven": "erhan seven",
        # Add more normalizations as needed
    }
    
    for pattern, replacement in normalizations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def extract_date_from_query(query: str) -> Optional[str]:
    """Extract date from query if present."""
    # Regular expression patterns to match different date formats
    date_patterns = [
        # YYYY-MM-DD format
        r'\b(\d{4}-\d{1,2}-\d{1,2})\b',
        # DD-MM-YYYY or DD/MM/YYYY format
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
        # Natural language date patterns
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*,?\s*\d{4}\b',
        r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*\d{4}\b',
        # Special case for our query
        r'\bon\s+(\d{4}-\d{1,2}-\d{1,2})\s+date\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, query.lower())
        if matches:
            try:
                # Try to parse the extracted date string
                date_str = matches[0]
                date_obj = date_parser.parse(date_str)
                # Return in YYYY-MM-DD format for consistent matching
                return date_obj.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Failed to parse date '{date_str}': {e}")
                continue
    
    return None

def extract_numeric_value_from_query(query: str, field_name: str) -> Optional[float]:
    """Extract numeric values like ratings from query."""
    # Pattern for ratings, e.g., "rating of 4", "4 stars", etc.
    if field_name.lower() == 'rating':
        patterns = [
            r'\b' + field_name + r'\s+of\s+(\d+(?:\.\d+)?)\b',
            r'\b' + field_name + r'\s*:\s*(\d+(?:\.\d+)?)\b',
            r'\b(\d+(?:\.\d+)?)\s+stars?\b',
            r'\b(\d+(?:\.\d+)?)\s+' + field_name + r'\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    pass
    
    return None

def retrieve_relevant_documents(query: str, index, metadata: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top_k most relevant documents for the query using hybrid retrieval with LLM parsing."""
    # Use LLM to extract structured filters and the core semantic query
    parsed_query = parse_query_with_llm(query)
    
    # Extract filters from parsed query
    date_start = parsed_query["filters"]["date_start"]
    date_end = parsed_query["filters"]["date_end"]
    rating_filter = parsed_query["filters"]["rating"]
    
    # Use the semantic core for vector search
    semantic_query = parsed_query["semantic_query"]
    
    # Print detected filters for debugging
    if date_start or date_end:
        print(f"Detected date range filter: {date_start} to {date_end}")
    if rating_filter is not None:
        print(f"Detected rating filter: {rating_filter}")
    
    # Determine if we have strong filters that should override semantic search
    has_strong_filter = date_start is not None or date_end is not None  # Date is considered a strong filter
    
    if has_strong_filter:
        # For strong filters like dates, filter first then apply semantic ranking
        print(f"Detected strong filter (date range: {date_start} to {date_end}). Searching all documents for matches.")
        
        # Apply filters to all documents
        filtered_results = []
        for idx, item in enumerate(metadata):
            # Apply date filter if specified
            if not filter_by_date(item, date_start, date_end):
                continue
                
            # Apply rating filter if specified
            if rating_filter is not None and not filter_by_rating(item, rating_filter):
                continue
            
            # Keep track of the document's original index for later semantic ranking
            result = item.copy()
            result["original_index"] = idx
            filtered_results.append(result)
        
        print(f"Found {len(filtered_results)} relevant reviews")
        
        if filtered_results:
            # Now that we have filtered results, we can rank them semantically if desired
            if len(filtered_results) > top_k:
                # Only do semantic ranking if we have more results than needed
                # Normalize the query for semantic search
                normalized_query = normalize_for_search(semantic_query)
                
                # Embed the query
                query_embedding = embed_text(normalized_query)
                if query_embedding is not None:
                    # Get indices of filtered results
                    filtered_indices = [r["original_index"] for r in filtered_results]
                    
                    # Extract embeddings of filtered results (stored in the FAISS index)
                    # This requires reconstructing the vectors from the index
                    filtered_embeddings = np.zeros((len(filtered_indices), index.d), dtype=np.float32)
                    for i, idx in enumerate(filtered_indices):
                        filtered_embeddings[i] = index.reconstruct(int(idx))
                    
                    # Create a temporary index with just the filtered embeddings
                    temp_index = faiss.IndexFlatL2(index.d)
                    temp_index.add(filtered_embeddings)
                    
                    # Search this temporary index
                    query_embedding_array = np.array([query_embedding])
                    distances, indices = temp_index.search(query_embedding_array, min(top_k, len(filtered_indices)))
                    
                    # Map back to original results
                    ranked_results = []
                    for i, idx in enumerate(indices[0]):
                        if idx != -1:  # Skip invalid results
                            result = filtered_results[idx].copy()
                            result["distance"] = float(distances[0][i])  # Add distance score
                            ranked_results.append(result)
                    
                    return ranked_results
            
            # If we have fewer results than top_k or semantic ranking failed, return all filtered results
            # Add a placeholder distance score
            for result in filtered_results:
                result["distance"] = 0.0
            
            return filtered_results[:top_k]
        else:
            return []  # No matches found after filtering
    else:
        # For regular queries without strong filters, use the hybrid approach (semantic first, then filter)
        # Normalize the query for semantic search
        normalized_query = normalize_for_search(semantic_query)
        
        # Embed the query
        query_embedding = embed_text(normalized_query)
        if query_embedding is None:
            print("Failed to generate embedding for the query.")
            return []
        
        # We'll retrieve more results initially for filtering
        initial_k = min(top_k * 3, len(metadata))  # Retrieve 3x results or all if small dataset
        
        # Reshape for FAISS
        query_embedding = np.array([query_embedding])
        
        # Search the index
        distances, indices = index.search(query_embedding, initial_k)
        
        # Get the corresponding metadata and apply structured filters
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # -1 means no match was found
                result = metadata[idx].copy()
                result["distance"] = float(distance)  # Convert numpy float to Python float
                
                # Apply date filter if specified
                if not filter_by_date(result, date_start, date_end):
                    continue
                    
                # Apply rating filter if specified
                if rating_filter is not None and not filter_by_rating(result, rating_filter):
                    continue
                    
                results.append(result)
        
        print(f"Found {len(results)} relevant reviews")
        
        # Limit to top_k results
        return results[:top_k]

def handle_count_query(query: str, index=None, metadata=None) -> str:
    """
    Handle a count-type query.
    
    Args:
        query: The user query string
        index: The FAISS index for vector search
        metadata: The metadata associated with the index
        
    Returns:
        String response with the count information
    """
    # Parse the query using LLM
    parsed_query = parse_query_with_llm(query)
    print(f"[COUNT QUERY] Parsed query: {parsed_query}")
    
    # Extract filters
    filters = {
        'date_start': parsed_query.get("date_start"),
        'date_end': parsed_query.get("date_end"),
        'rating': parsed_query.get("rating"),
        'sentiment': parsed_query.get("sentiment")
    }
    
    # Count matching documents
    count = count_matching_documents(metadata, filters)
    print(f"Final count of matching documents: {count}")
    
    # Get samples of matching documents for context
    matching_docs = []
    sample_size = min(5, count)  # Get up to 5 examples
    sample_count = 0
    
    if count > 0:
        print(f"Finding {sample_size} sample documents for count query context...")
        for item in metadata:
            # Apply filters
            date_match = True
            if filters['date_start'] or filters['date_end']:
                date_match = filter_by_date(item, filters['date_start'], filters['date_end'])
            
            rating_match = filter_by_rating(item, filters['rating']) if filters['rating'] else True
            sentiment_match = filter_by_sentiment(item, filters['sentiment']) if filters['sentiment'] else True
            
            # Add to samples if all filters match
            if date_match and rating_match and sentiment_match:
                matching_docs.append(item)
                sample_count += 1
                
            # Stop once we have enough samples
            if sample_count >= sample_size:
                break
    
    # Prepare examples for the prompt
    examples = ""
    if matching_docs:
        for i, doc in enumerate(matching_docs):
            # Extract the most important fields for display
            rating = doc.get("ratingValue", "unknown rating")
            date_str = "unknown date"
            if "full_review_json" in doc and doc["full_review_json"]:
                if isinstance(doc["full_review_json"], dict):
                    date_str = doc["full_review_json"].get("date", "unknown date")
            
            review_text = doc.get("ratingText", "No review text available")
            examples += f"Example {i+1} (Rating: {rating}, Date: {date_str}):\n{review_text}\n\n"
    
    # Format system and user prompts for a direct API call
    system_prompt = """You are a helpful assistant analyzing review data. 
    When providing count information, always be precise and mention the exact count.
    If there are reviews matching the criteria, summarize patterns or insights from the examples.
    If there are no reviews matching the criteria, clearly state that there are 0 matching reviews."""

    user_prompt = f"""The user asked: "{query}"

    Based on our database search, there are {count} reviews that match the query criteria:
    - Date range: {filters['date_start'] or 'any'} to {filters['date_end'] or 'any'}
    - Rating filter: {filters['rating'] or 'any'}
    - Sentiment filter: {filters['sentiment'] or 'any'}

    {examples if matching_docs else 'No matching examples found.'}

    Please provide a natural language response answering the user's question.
    Be sure to explicitly mention that there are {count} reviews matching their criteria."""

    try:
        # Call OpenAI API directly
        print("Making direct call to OpenAI API for count query response...")
        response = client.chat.completions.create(
            model="gpt-4o",  # Use a capable model for summarization
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5  # Lower temperature for more factual responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating count response with OpenAI API: {e}")
        # Fallback to basic response
        response = f"Found {count} reviews that match your criteria."
        if count > 0 and matching_docs:
            response += f"\n\nExample: {matching_docs[0].get('ratingText', 'No text available')}"
            
    return response

def handle_retrieval_query(query: str, index=None, metadata=None, max_results: int = 10) -> str:
    """
    Handle a retrieval-type query using a filter-then-search hybrid approach:
    1. First filter all documents based on structured criteria (dates, ratings, etc.)
    2. Then apply semantic search on the filtered subset to rank by relevance
    
    Args:
        query: The user query
        index: The FAISS index for vector search
        metadata: The metadata associated with the index
        max_results: Maximum number of results to retrieve for summarization
        
    Returns:
        String response with summarized information
    """
    # Parse the query using LLM
    parsed_query = parse_query_with_llm(query)
    print(f"[RETRIEVAL QUERY] Parsed query: {parsed_query}")
    
    core_query = parsed_query.get("core_query", query)
    date_start = parsed_query.get("date_start")
    date_end = parsed_query.get("date_end")
    rating = parsed_query.get("rating")
    sentiment = parsed_query.get("sentiment")
    
    # Phase 1: Get all documents that match the filters
    print(f"Phase 1: Filtering all documents based on criteria...")
    filtered_items = []
    filtered_ids = []
    
    for idx, item in enumerate(metadata):
        # Apply date filter if specified
        date_match = True
        if date_start or date_end:
            date_match = filter_by_date(item, date_start, date_end)
            
        # Apply rating filter if specified
        rating_match = True
        if rating:
            rating_match = filter_by_rating(item, rating)
        
        # Apply sentiment filter if specified
        sentiment_match = True
        if sentiment:
            sentiment_match = filter_by_sentiment(item, sentiment)
        
        # Add to filtered list if all filters match
        if date_match and rating_match and sentiment_match:
            filtered_items.append(item)
            filtered_ids.append(idx)
    
    print(f"Found {len(filtered_items)} documents matching all filters")
    
    # If no filters were applied or few results, consider all documents
    if not (date_start or date_end or rating or sentiment) and len(filtered_items) == 0:
        print("No filters applied and no results found, using all documents")
        filtered_items = metadata
        filtered_ids = list(range(len(metadata)))
    
    # If we don't have enough matching documents, return what we have
    if len(filtered_items) == 0:
        context = "No documents found matching the specified filters."
        return refine_with_llm(query, context)
    
    # Phase 2: Apply semantic search on filtered results
    print(f"Phase 2: Ranking {len(filtered_items)} filtered documents by semantic relevance...")
    filtered_results = []
    
    try:
        # Create embeddings for the core query
        query_embedding = embed_text(core_query)
        
        # Make sure query embedding is properly shaped for FAISS
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D array with one row
        
        if len(filtered_ids) > 0:
            # Option 1: If we're working with a subset, create a temporary index
            if len(filtered_ids) < len(metadata):
                # Extract vectors for filtered items
                filtered_vectors = np.vstack([index.reconstruct(idx) for idx in filtered_ids])
                
                # Create a temporary index for the filtered subset
                temp_index = faiss.IndexFlatL2(filtered_vectors.shape[1])
                temp_index.add(filtered_vectors)
                
                # Search the temporary index
                D, I = temp_index.search(query_embedding, min(max_results, len(filtered_items)))
                
                # Map temporary indices back to the filtered items
                for i, tmp_idx in enumerate(I[0]):
                    if D[0][i] != -1 and tmp_idx < len(filtered_items):
                        item = filtered_items[tmp_idx]
                        item["distance"] = float(D[0][i])
                        filtered_results.append(item)
            else:
                # Option 2: If all docs are filtered in, just search the main index
                D, I = index.search(query_embedding, min(max_results, len(filtered_items)))
                
                # Process the results
                for i, idx in enumerate(I[0]):
                    if D[0][i] != -1 and idx < len(metadata):
                        item = metadata[idx]
                        item["distance"] = float(D[0][i])
                        filtered_results.append(item)
        else:
            # Fallback if filtered_ids is empty
            print("Warning: filtered_ids is empty, no semantic search performed")
            # Take the first few filtered items without ranking
            filtered_results = filtered_items[:max_results]
    except Exception as e:
        print(f"Error during semantic ranking: {e}")
        # Fallback: Use the first few filtered items without semantic ranking
        filtered_results = filtered_items[:max_results]
    
    print(f"Final result count after semantic ranking: {len(filtered_results)}")
    
    # Prepare context and get the refined answer
    context = prepare_context(filtered_results)
    response = refine_with_llm(query, context)
    
    return response

def retrieve_relevant_documents(query: str, top_k: int = 5, initial_k: int = 100) -> List[Dict[str, Any]]:
    """Retrieve relevant documents from the FAISS index."""
    # Parse the query using LLM
    parsed_query = parse_query_with_llm(query)
    print(f"Parsed query: {parsed_query}")
    
    core_query = parsed_query.get("core_query", query)
    date_start = parsed_query.get("date_start")
    date_end = parsed_query.get("date_end")
    rating = parsed_query.get("rating")
    sentiment = parsed_query.get("sentiment")
    
    # Embed the semantic part of the query
    query_embedding = embed_text(core_query)
    
    # Retrieve initial results based on semantic similarity
    D, I = index.search(query_embedding, min(initial_k, len(metadata)))  # Get more initially to filter
    
    # Filter results
    filtered_results = []
    for i, idx in enumerate(I[0]):
        item = metadata[idx]
        
        # Skip if distance is -1 (faiss error)
        if D[0][i] == -1:
            continue
        
        # Apply date filter if specified
        date_match = filter_by_date(item, date_start, date_end)
        if not date_match:
            continue
            
        # Apply rating filter if specified
        if rating and not filter_by_rating(item, rating):
            continue
        
        # Apply sentiment filter if specified
        if sentiment and not filter_by_sentiment(item, sentiment):
            continue
            
        # Add distance for reference
        item["distance"] = float(D[0][i])
        
        filtered_results.append(item)
        
        # Stop once we have top_k results
        if len(filtered_results) >= top_k:
            break
            
    return filtered_results

def filter_by_date(result: Dict[str, Any], date_start: Optional[str], date_end: Optional[str]) -> bool:
    """Filter a result by date range."""
    # Debug info
    print(f"\nDEBUG - Filter by date - Date range: {date_start} to {date_end}")
    print(f"  DEBUG - Result keys: {list(result.keys())}")
    
    # If no date filters specified, all results pass the filter
    if date_start is None and date_end is None:
        print("  DEBUG - No date filters, returning True")
        return True
    
    # Three possible locations for the date:
    # 1. Direct 'date' field in the result
    # 2. Inside 'full_review_json' field as a JSON string
    # 3. Alternative field name like 'reviewDate'
    
    result_date_str = None
    
    # Method 1: Direct date field
    if 'date' in result:
        result_date_str = result['date']
        print(f"  DEBUG - Found direct date field: {result_date_str}")
    
    # Method 2: Check inside full_review_json
    elif 'full_review_json' in result and result['full_review_json']:
        try:
            print("  DEBUG - Checking full_review_json for date")
            # Check if full_review_json is already a dictionary
            if isinstance(result['full_review_json'], dict):
                json_data = result['full_review_json']
                print("  DEBUG - full_review_json is already a dictionary")
            else:
                # Parse it from a string if it's not already a dictionary
                json_data = json.loads(result['full_review_json'])
                print("  DEBUG - Parsed full_review_json from string")
                
            if 'date' in json_data:
                result_date_str = json_data['date']
                print(f"  DEBUG - Extracted date from full_review_json: {result_date_str}")
        except Exception as e:
            print(f"  DEBUG - Error processing full_review_json: {e}")
            # Print a sample of the full_review_json to debug
            sample = str(result['full_review_json'])[:100] + '...' if len(str(result['full_review_json'])) > 100 else str(result['full_review_json'])
            print(f"  DEBUG - Sample of full_review_json: {sample}")
            # If it's a dict, try to directly access 'date'
            if isinstance(result['full_review_json'], dict) and 'date' in result['full_review_json']:
                result_date_str = result['full_review_json']['date']
                print(f"  DEBUG - Directly extracted date from dict: {result_date_str}")
    
    # Method 3: Alternative field names
    elif 'reviewDate' in result:
        result_date_str = result['reviewDate']
        print(f"  DEBUG - Found reviewDate field: {result_date_str}")
    
    # If we still don't have a date, we can't filter
    if not result_date_str:
        print("  DEBUG - Could not find any date field, returning False")
        return False
    
    try:
        # Parse the result date string to a datetime object
        # The format is like '2025-01-09 12:09:36.261000+00:00'
        result_date = date_parser.parse(result_date_str).date()
        print(f"  DEBUG - Parsed date: {result_date}")
        
        # Apply start date filter if specified
        if date_start is not None:
            start_date = datetime.datetime.strptime(date_start, '%Y-%m-%d').date()
            print(f"  DEBUG - Start date filter: {start_date}")
            if result_date < start_date:
                print("  DEBUG - Date before start date, returning False")
                return False
        
        # Apply end date filter if specified
        if date_end is not None:
            end_date = datetime.datetime.strptime(date_end, '%Y-%m-%d').date()
            print(f"  DEBUG - End date filter: {end_date}")
            if result_date > end_date:
                print("  DEBUG - Date after end date, returning False")
                return False
        
        # If we got here, the result is within the date range
        print("  DEBUG - Date within range, returning True")
        return True
    except Exception as e:
        print(f"  DEBUG - Error in date filtering: {e}")
        return False

def filter_by_rating(result: Dict[str, Any], rating_filter: float) -> bool:
    """Filter a result by rating."""
    if not rating_filter:
        return True  # No rating filter
    
    # Get rating from the result
    if "original_row" in result and "ratingValue" in result["original_row"]:
        rating_str = result["original_row"]["ratingValue"]
    elif "ratingValue" in result:
        rating_str = result["ratingValue"]
    elif "full_review_json" in result and result["full_review_json"]:
        try:
            # Check if full_review_json is already a dictionary
            if isinstance(result["full_review_json"], dict):
                json_data = result["full_review_json"]
            else:
                json_data = json.loads(result["full_review_json"])
                
            if "ratingValue" in json_data:
                rating_str = json_data["ratingValue"]
            else:
                return False
        except Exception as e:
            print(f"Error parsing full_review_json for rating: {e}")
            return False
    else:
        return False
    
    # Skip if rating is None, NaN, empty string, etc.
    if not rating_str or str(rating_str).lower() == "nan":
        return False
    
    try:
        rating = float(rating_str)
        return abs(rating - rating_filter) < 0.01  # Allow for floating-point comparison
    except ValueError:
        return False

def filter_by_sentiment(result: Dict[str, Any], sentiment_filter: str) -> bool:
    """Filter a result by sentiment (positive, negative, neutral)."""
    if not sentiment_filter:
        return True  # No sentiment filter
    
    # Try to get explicit sentiment if available
    sentiment = None
    
    # Check in original_row
    if "original_row" in result and "sentiment" in result["original_row"]:
        sentiment = result["original_row"]["sentiment"]
    # Direct field
    elif "sentiment" in result:
        sentiment = result["sentiment"]
    # Check in full_review_json
    elif "full_review_json" in result and result["full_review_json"]:
        try:
            # Check if full_review_json is already a dictionary
            if isinstance(result["full_review_json"], dict):
                json_data = result["full_review_json"]
            else:
                json_data = json.loads(result["full_review_json"])
                
            if "sentiment" in json_data:
                sentiment = json_data["sentiment"]
        except Exception as e:
            print(f"Error parsing full_review_json for sentiment: {e}")
    
    # If we found an explicit sentiment value, use it
    if sentiment and isinstance(sentiment, str) and sentiment.strip().lower() != "nan":
        return sentiment.strip().lower() == sentiment_filter.lower()
    
    # If no explicit sentiment, infer from rating
    rating = None
    
    # Try to get rating from different locations
    if "original_row" in result and "ratingValue" in result["original_row"]:
        rating_str = result["original_row"]["ratingValue"]
    elif "ratingValue" in result:
        rating_str = result["ratingValue"]
    elif "full_review_json" in result and result["full_review_json"]:
        try:
            # Check if full_review_json is already a dictionary
            if isinstance(result["full_review_json"], dict):
                json_data = result["full_review_json"]
            else:
                json_data = json.loads(result["full_review_json"])
                
            if "ratingValue" in json_data:
                rating_str = json_data["ratingValue"]
            else:
                return False
        except Exception:
            return False
    else:
        return False
    
    # Skip if rating is None, NaN, empty string, etc.
    if not rating_str or str(rating_str).lower() == "nan":
        return False
    
    # Infer sentiment from rating
    try:
        rating = float(rating_str)
        inferred_sentiment = ""
        
        if rating >= 4:
            inferred_sentiment = "positive"
        elif rating <= 2:
            inferred_sentiment = "negative"
        else:
            inferred_sentiment = "neutral"
            
        return inferred_sentiment.lower() == sentiment_filter.lower()
    except ValueError:
        return False
        
def count_matching_documents(metadata: List[Dict[str, Any]], filters: Dict[str, Any]) -> int:
    """
    Count documents that match all specified filters.
    This handles counting for 'count' type queries, scanning the entire metadata.
    
    Args:
        metadata: List of all document metadata
        filters: Dictionary with filter values (date_start, date_end, rating, sentiment)
    
    Returns:
        int: Count of documents matching all filters
    """
    matching_count = 0
    date_start = filters.get('date_start')
    date_end = filters.get('date_end')
    rating = filters.get('rating')
    sentiment = filters.get('sentiment')
    
    print(f"Counting documents with filters: {filters}")
    
    for item in metadata:
        # Check if the document passes all active filters
        date_match = True
        if date_start or date_end:
            # Get the date from the document (already handled in filter_by_date)
            if "full_review_json" in item and item["full_review_json"]:
                if isinstance(item["full_review_json"], dict):
                    json_data = item["full_review_json"]
                else:
                    try:
                        json_data = json.loads(item["full_review_json"])
                    except:
                        continue  # Skip if JSON parsing fails
                
                if "date" in json_data:
                    date_str = json_data["date"]
                    date_match = filter_by_date(item, date_start, date_end)
                else:
                    date_match = False  # No date field
            else:
                date_match = False  # No full_review_json
        
        # Rating filter
        rating_match = filter_by_rating(item, rating) if rating else True
        
        # Sentiment filter
        sentiment_match = filter_by_sentiment(item, sentiment) if sentiment else True
        
        # Count if all filters match
        if date_match and rating_match and sentiment_match:
            matching_count += 1
    
    return matching_count

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
7. You are basically a refiner and your job is that i will provide myou query and Response which is fetched from RAG and you have to refine the response and retirn response like here are the reviews here is this do not include like these type of statements that Based on Reviews a, because i know that my response s being refined by you but the person using my app doesnt know so act like very intelligent
8. Also you have to carefully analyze the query and then the response fetched from embeddings you have to refine the response and only provide that content which user is asking in query not wrong content
9. You have to be so intelligent that for example if user ask that provide me reviews which have negative keywords and you get some reviews in your context then you have to only select those reviews which are matching for examplke if you get some revieww in that reviews which have rating 1 and there is no review text , then you dont have to provide that review because you have to provide only with some review text which have negative and harsh keywords, this is just a use case example that dont be so silly provide answer intelligently you are middle man which see query of user and some text from rag and in that context its not confirmed that everything in that content could be right so you have to provide accoedsing to user query
9. Also please provide a response in very beautiful format dont include Review #1 or any number you respons eshould be very formatted

SPECIAL TERM HANDLING:
- Normalize any mentions of "Obenan" including misspellings or phonetic variants.
- Do the same for "Omnipulse" and "Erhan Seven".

FORMAT YOUR RESPONSE:
- If the query asks for a list of reviews or information from multiple reviews, include ALL reviews in your response.
- For each review in the context, include a brief section with its key information.
- Start with a direct answer to the query.
- Include supporting evidence from all the reviews.
- If relevant, note the general sentiment, consensus, or trends across reviews.
- Be comprehensive yet clear and structured."""

    # Count the number of reviews in the context to inform the LLM
    review_count = context.count('REVIEW #') if context else 0
    
    user_prompt = f"""QUERY: {query}

CONTEXT FROM RETRIEVED REVIEWS ({review_count} reviews total):
{context}

IMPORTANT: Include information from ALL {review_count} reviews in your answer. Do not skip any reviews.
Based solely on the above context, please answer the query comprehensively."""

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

def main():
    """Main function to process queries either interactively or with a test query."""
    print("Loading FAISS index and metadata...")
    global index, metadata
    index, metadata = load_index_and_metadata()
    print("Ready to process queries!")
    
    interactive = True  # Set to False to use the hardcoded test query
    
    if interactive:
        print("\nEnter queries below. Type 'exit' to quit.")
        while True:
            # Get user query
            try:
                query = input("\nQUERY: ")
                if query.lower() in ['exit', 'quit']:
                    print("Exiting. Goodbye!")
                    break
                    
                if not query.strip():
                    continue
                    
                # Process the query
                # Pass index and metadata explicitly to ensure they're available
                process_query(query, index=index, metadata=metadata)
                
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
    else:
        # Test query
        test_query = "How many reviews i got in last 10 months"
        print(f"\nTEST QUERY: {test_query}")
        # Pass index and metadata explicitly to ensure they're available
        process_query(test_query, index=index, metadata=metadata)

def process_query(query: str, index=None, metadata=None):
    """Process a single query and return the result."""
    print("Processing query...")
    start_time = datetime.datetime.now()
    
    # Parse the query to determine its type
    parsed_query = parse_query_with_llm(query)
    query_type = parsed_query.get("query_type", "retrieval")
    print("-------------")
    print(query_type)
    print("-------------")
    # Get max_results if specified, otherwise use default values
    # Use a higher default for retrieval queries (15) and an even higher value (100) if "all" is implied
    max_results = parsed_query.get("max_results", 15)  # Default to 15 results
    
    # If query mentions "all" or similar but parser didn't catch it, use a high number
    if any(word in query.lower() for word in ["all", "every", "each", "complete list"]):
        max_results = 100  # Practical maximum
    
    print(f"Query type identified: {query_type}, Max results: {max_results}")
    
    # Handle based on query type
    if query_type.lower() == "count":
        print("Handling as COUNT query - counting matching documents...")
        answer = handle_count_query(query, index=index, metadata=metadata)
    else:  # Default to retrieval
        print(f"Handling as RETRIEVAL query - searching for up to {max_results} relevant documents...")
        answer = handle_retrieval_query(query, index=index, metadata=metadata, max_results=max_results)
    
    # Display answer
    print("\n" + "=" * 80)
    print("ANSWER:")
    print(answer)
    print("=" * 80)
    
    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"Query processed in {processing_time:.2f} seconds.")
    return answer

if __name__ == "__main__":
    main()
