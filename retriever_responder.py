import os
import json
import faiss
import pickle
import numpy as np
import datetime
import dateutil.parser
from dateutil.relativedelta import relativedelta
import re
import logging
import sys
import calendar
import difflib
from typing import Dict, Any, List, Optional, Tuple, Union
from openai import OpenAI
from dotenv import load_dotenv
from dateutil import parser as date_parser
import pandas as pd
import random

# Configure standardized logging
def setup_logging():
    """Set up standardized logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="[RAG] %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("retriever")

# Create logger instance
logger = setup_logging()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")


def has_text_content(review):
    """
    Helper function to check if a review has meaningful text content.
    Returns: 
        - 2 for full text content
        - 1 for minimal metadata (rating, date, location, but no text) 
        - 0 for no useful content
    """
    # If review is None or not a dictionary, it has no text content
    if review is None or not isinstance(review, dict):
        return 0
        
    # First check for text in full_review_json (most reliable source)
    if 'full_review_json' in review:
        json_data = {}
        if isinstance(review['full_review_json'], dict):
            json_data = review['full_review_json']
        elif isinstance(review['full_review_json'], str):
            try:
                json_data = json.loads(review['full_review_json'])
            except:
                pass
                
        if json_data:
            # First check for any text fields that might contain the review text
            for field in ['ratingText', 'reviewText', 'text', 'comment', 'feedback', 'description', 'comments']:
                if field in json_data:
                    text_value = json_data.get(field)
                    if text_value and isinstance(text_value, str) and len(text_value.strip()) > 5 and str(text_value).lower() not in ['nan', 'none', 'null']:
                        logger.debug(f"Found text content in {field}: {text_value[:50]}...")
                        return 2  # Full text content
            
            # Check if it has essential metadata (second priority)
            has_rating = 'ratingValue' in json_data and json_data['ratingValue'] is not None
            has_date = 'date' in json_data and json_data['date'] is not None
            has_location = 'locationId' in json_data and json_data['locationId'] is not None
            
            if has_rating and (has_date or has_location):
                # Has metadata but no text
                return 1
    
    # Next check direct fields in the review dictionary
    for field in ['reviewText', 'ratingText', 'text', 'comment', 'feedback', 'description', 'comments']:
        if field in review:
            text_value = review.get(field)
            if text_value and isinstance(text_value, str) and len(text_value.strip()) > 5 and str(text_value).lower() not in ['nan', 'none', 'null']:
                logger.debug(f"Found direct text content in {field}: {text_value[:50]}...")
                return 2  # Full text content
    
    # Check combined_text if it exists
    if 'combined_text' in review and review['combined_text']:
        combined_text = str(review['combined_text'])
        
        # Extract review text if present
        if 'Review: ' in combined_text:
            parts = combined_text.split('Review: ')
            if len(parts) > 1:
                review_text = parts[1].split('\n')[0].strip()
                # Check if the review text is meaningful
                if (review_text and len(review_text) > 5 and 
                    review_text.lower() not in ['nan', 'none', 'null', 'no review text available', '[no review text available]']):
                    logger.debug(f"Found text in combined_text: {review_text[:50]}...")
                    return 2  # Full text content
        
        # Check for rating and other metadata
        has_rating = 'Rating: ' in combined_text and not ('Rating: N/A' in combined_text)
        has_date = 'Date: ' in combined_text
        has_location = ('Location: ' in combined_text) or ('Location ID: ' in combined_text)
        
        if has_rating and (has_date or has_location):
            return 1  # Minimal content
    
    # Last resort - check basic fields for metadata
    has_rating = any(key in review for key in ['rating', 'ratingValue'])
    has_date = any(key in review for key in ['date', '_parsed_date'])
    has_location = any(key in review for key in ['location', 'locationId', 'locationName'])
    
    if has_rating and (has_date or has_location):
        return 1  # Minimal content with no text
        
    # Nothing useful found
    return 0

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# LLM model to use for parsing queries (faster and cheaper than the main completion model)
QUERY_PARSER_MODEL = "gpt-3.5-turbo"  # Fast model for parsing

# Model for summarization tasks (can use a faster/cheaper model than the main one)
SUMMARIZATION_MODEL = "gpt-3.5-turbo"  # Fast model for summarization

def interpret_relative_date(date_text, current_date=None):
    """
    Convert relative date references to absolute date ranges.
    Intelligently handles queries for last year, last N years, and specific years.
    
    Args:
        date_text: A string containing a relative date reference (e.g., "last year", "previous month")
        current_date: Optional datetime object representing current date (for testing)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects, or (None, None) if can't interpret
    """
    if not date_text or not isinstance(date_text, str):
        return None, None
        
    if current_date is None:
        current_date = datetime.datetime.now()
    
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day
    
    date_text = date_text.lower()
    
    # Debug current date for better insights
    logger.debug(f"Current date reference: {current_date.strftime('%Y-%m-%d')}")
    logger.debug(f"Using current_year value: {current_year}")
    
    # Check for specific calendar year references (e.g., "2021", "in 2022")
    year_pattern = r'\b((?:19|20)\d{2})\b'  # Capture the full 4-digit year
    year_matches = re.findall(year_pattern, date_text)
    
    if year_matches:
        specific_year = int(year_matches[0])
        logger.debug(f"Found specific year reference: {specific_year}")
        start_date = datetime.datetime(specific_year, 1, 1)
        end_date = datetime.datetime(specific_year, 12, 31, 23, 59, 59)
        return start_date, end_date
    
    # Handle relative date formats with rolling windows (e.g., "last 3 months" means 3 months from today)
    
    # Pattern for "last N years" or "past N years" - MUST BE CHECKED BEFORE GENERAL "last year"
    n_years_pattern = r'last\s+(\d+)\s+years|past\s+(\d+)\s+years'
    n_years_match = re.search(n_years_pattern, date_text)
    if n_years_match:
        n = int(n_years_match.group(1) or n_years_match.group(2))
        # Rolling window from N years ago until today
        start_date = current_date - relativedelta(years=n)  # Use relativedelta for proper year calculation
        end_date = current_date
        logger.debug(f"Last {n} years interpreted as rolling window: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
    
    # Pattern for "last N months"
    n_months_pattern = r'last\s+(\d+)\s+months|past\s+(\d+)\s+months'
    n_months_match = re.search(n_months_pattern, date_text)
    if n_months_match:
        n = int(n_months_match.group(1) or n_months_match.group(2))
        # Calculate months back
        target_month = current_month - n
        target_year = current_year
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        start_date = datetime.datetime(target_year, target_month, 1)
        end_date = current_date
        logger.debug(f"Last {n} months interpreted as: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
    
    # Handle common relative date formats (simpler cases)
    if "last year" in date_text or "previous year" in date_text or "past year" in date_text:
        # Last calendar year - from Jan 1 to Dec 31 of previous year
        start_date = datetime.datetime(current_year - 1, 1, 1)
        end_date = datetime.datetime(current_year - 1, 12, 31, 23, 59, 59)
        logger.debug(f"Last year interpreted as: {start_date.date()} to {end_date.date()}")
        logger.debug(f"Calculation details: current_year {current_year} minus 1 = {current_year - 1}")
        return start_date, end_date
        
    elif "this year" in date_text or "current year" in date_text:
        # Current calendar year - from Jan 1 to current date
        start_date = datetime.datetime(current_year, 1, 1)
        end_date = current_date
        logger.debug(f"This year interpreted as: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
        
    elif "last month" in date_text or "previous month" in date_text:
        # Last calendar month - full previous month
        if current_month == 1:  # January
            start_date = datetime.datetime(current_year - 1, 12, 1)
            end_date = datetime.datetime(current_year - 1, 12, 31, 23, 59, 59)
        else:
            start_date = datetime.datetime(current_year, current_month - 1, 1)
            # Calculate last day of previous month correctly
            last_day = calendar.monthrange(current_year, current_month - 1)[1]
            end_date = datetime.datetime(current_year, current_month - 1, last_day, 23, 59, 59)
        logger.debug(f"Last month interpreted as: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
        
    elif "last 2 years" in date_text or "last two years" in date_text or "past 2 years" in date_text:
        # Last two calendar years through current date (NOT just previous 2 years)
        # Calculate actual date from 2 years ago up to today
        start_date = datetime.datetime(current_year - 2, current_month, current_day)
        end_date = current_date
        print(f"DEBUG - Last 2 years interpreted as: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
        
    elif "last 3 years" in date_text or "last three years" in date_text:
        # Last three years through current date
        start_date = datetime.datetime(current_year - 3, current_month, current_day)
        end_date = current_date
        logger.debug(f"Last 3 years interpreted as: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
    
    # Handle special keywords for oldest/newest reviews
    if "oldest" in date_text or "earliest" in date_text:
        # For oldest reviews, we want to set a very early start date and no end date
        # This allows the query to find the oldest reviews in the dataset
        logger.debug(f"'oldest' keyword detected, returning open-ended date range for oldest reviews")
        start_date = datetime.datetime(1900, 1, 1)  # A very early date
        end_date = None  # No end date limit
        return start_date, end_date
    
    elif "newest" in date_text or "latest" in date_text or "recent" in date_text or "most recent" in date_text:
        # For newest reviews, we want to find the most recent reviews
        # We'll use a relatively recent start date and the current date as end
        logger.debug(f"'newest' keyword detected, returning date range for newest reviews")
        # Start from 30 days ago to find recent reviews
        start_date = current_date - datetime.timedelta(days=30)
        end_date = current_date
        return start_date, end_date
        
    # Add more patterns as needed
    
    return None, None


def parse_query_with_llm(query: str) -> Dict[str, Any]:
    """
    Use LLM to extract structured filters and the semantic core of the query.
    Returns a dictionary with the extracted information.
    Handles relative date references and date ranges.
    """
    try:
        # Get current date for relative date understanding
        current_date_dt = datetime.datetime.now()
        current_date = current_date_dt.strftime("%Y-%m-%d")
        logger.debug(f"parse_query_with_llm using current date: {current_date}")
        
        prompt = f"""
        Extract structured filters from the following query.
        TODAY'S DATE IS: {current_date}
        
        Analyze the query: "{query}"
        
        Extract any dates or date ranges mentioned. IMPORTANT: Convert ALL relative dates to absolute dates using today's date.
        Extract rating filters (e.g., "5-star reviews", "ratings above 4").
        Extract sentiment filters (e.g., "positive reviews", "negative feedback").
        Extract count limits (e.g., "show me 5 reviews", "top 10 complaints").
        
        Identify if this is a COUNT query (e.g., "How many reviews..."), a LATEST query (e.g., "Show me the most recent..."), 
        or a RETRIEVAL query (e.g., "Find reviews about...").
        
        Format your response as valid JSON with the following structure:
        {{
            "query_type": "count|latest|retrieval",
            "core_query": "the semantic core of the query",
            "date_start": "YYYY-MM-DD", (if applicable)
            "date_end": "YYYY-MM-DD", (if applicable)
            "rating": number, (if applicable)
            "sentiment": "positive|negative|neutral", (if applicable)
            "max_results": number (if applicable)
        }}
        """
        
        response = client.chat.completions.create(
            model=QUERY_PARSER_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"LLM parsing response: {response_text}")
        logger.debug(f"Raw LLM response for query parsing: {response_text}")
        
        # Try to extract JSON from the response
        try:
            # Find JSON object in the response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                
                # Basic validation and default values
                if "query_type" not in parsed_data:
                    if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]):
                        parsed_data["query_type"] = "count"
                    elif any(word in query.lower() for word in ["latest", "newest", "most recent", "recent"]):
                        parsed_data["query_type"] = "latest"
                    else:
                        parsed_data["query_type"] = "retrieval"
                        
                return parsed_data
            else:
                print("No JSON object found in the LLM response")
                # Basic fallback detection of query type
                if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]):
                    query_type = "count"
                elif any(word in query.lower() for word in ["latest", "newest", "most recent", "recent"]):
                    query_type = "latest"
                else:
                    query_type = "retrieval"
                return {"core_query": query, "query_type": query_type}
        except Exception as e:
            logger.error(f"Error parsing JSON from LLM response: {e}")
            # Basic fallback detection of query type
            query_type = "count" if any(word in query.lower() for word in ["how many", "count", "total number", "number of"]) else "retrieval"
            return {"core_query": query, "query_type": query_type}
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        print("Using enhanced fallback parser...")
        
        # Enhanced fallback parser that extracts more structure from the query
        result = {
            "core_query": query,
            "semantic_query": query,
            "actions": []
        }
        
        # Detect query type with more specific patterns
        query_lower = query.lower()
        
        # 1. Detect query types/actions
        if any(word in query_lower for word in ["how many", "count", "total number", "number of"]):
            result["query_type"] = "count"
            result["actions"] = ["count"]
        elif any(word in query_lower for word in ["latest", "newest", "most recent", "recent"]):
            result["query_type"] = "latest"
            result["actions"] = ["latest"]
        else:
            result["query_type"] = "retrieval"
            result["actions"] = ["retrieval"]
            
        # Check for multi-intent queries
        compound_phrases = [
            "also show", "also tell", "with example", "with some example", 
            "show me some", "tell me some", "and show me", "along with"
        ]
        if any(phrase in query_lower for phrase in compound_phrases):
            if "retrieval" not in result["actions"]:
                result["actions"].append("retrieval")
            result["include_examples"] = True
        
        # 2. Extract date ranges using regex patterns
        # Extract years if present in date text
        year_pattern = r'\b((?:19|20)\d{2})\b'
        month_pattern = r'(?:in|from|during|for)\s+(?:the\s+month\s+of\s+)?([Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)'
        last_n_pattern = r'(?:in|from|during|for|last|past)\s+(\d+)\s+([Dd]ays?|[Ww]eeks?|[Mm]onths?|[Yy]ears?)'
        
        # Extract year
        year_match = re.search(year_pattern, query_lower)
        if year_match:
            year = year_match.group(1)
            # Set approximate date range for the year
            result["date_start"] = f"{year}-01-01"
            result["date_end"] = f"{year}-12-31"
        
        # Extract month
        month_match = re.search(month_pattern, query_lower)
        if month_match:
            month_name = month_match.group(1).lower()
            month_to_num = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            month_num = month_to_num.get(month_name, '01')
            current_year = datetime.datetime.now().year
            # Set approximate date range for the month
            result["date_start"] = f"{current_year}-{month_num}-01"
            
            # Calculate last day of month
            if month_num in ['04', '06', '09', '11']:
                result["date_end"] = f"{current_year}-{month_num}-30"
            elif month_num == '02':
                # Simple leap year approximation
                last_day = '29' if current_year % 4 == 0 else '28'
                result["date_end"] = f"{current_year}-{month_num}-{last_day}"
            else:
                result["date_end"] = f"{current_year}-{month_num}-31"
        
        # Extract last N time period
        last_n_match = re.search(last_n_pattern, query_lower)
        if last_n_match:
            n = int(last_n_match.group(1))
            period = last_n_match.group(2).lower()
            now = datetime.datetime.now()
            
            # Calculate start date based on period
            if 'day' in period:
                start_date = now - datetime.timedelta(days=n)
            elif 'week' in period:
                start_date = now - datetime.timedelta(days=n*7)
            elif 'month' in period:
                # Approximate months as 30 days
                start_date = now - datetime.timedelta(days=n*30)
            elif 'year' in period:
                # Approximate years as 365 days
                start_date = now - datetime.timedelta(days=n*365)
            else:
                start_date = now - datetime.timedelta(days=30)  # Default to 30 days
                
            result["date_start"] = start_date.date().isoformat()
            result["date_end"] = now.date().isoformat()
        
        # 3. Extract rating filters
        rating_patterns = [
            (r'(?:rating|ratings|rated|star|stars)\s+(?:of|=|==|is|are|with)?\s*(\d+)\s*(?:star|stars)?', lambda x: int(x)),
            (r'(\d+)\s*(?:star|stars|rating)', lambda x: int(x)),
            (r'(?:rating|ratings|rated)\s+(?:above|over|more than|>|>=)\s*(\d+)', lambda x: int(x)),
            (r'(?:rating|ratings|rated)\s+(?:below|under|less than|<|<=)\s*(\d+)', lambda x: int(x))
        ]
        
        for pattern, converter in rating_patterns:
            rating_match = re.search(pattern, query_lower)
            if rating_match:
                rating_value = converter(rating_match.group(1))
                result["rating"] = rating_value
                break
        
        # 4. Extract sentiment
        if any(word in query_lower for word in ["positive", "good", "great", "excellent", "happy", "satisfied"]):
            result["sentiment"] = "positive"
        elif any(word in query_lower for word in ["negative", "bad", "poor", "terrible", "unhappy", "dissatisfied"]):
            result["sentiment"] = "negative"
        elif any(word in query_lower for word in ["neutral", "moderate", "average", "ordinary"]):
            result["sentiment"] = "neutral"
        
        # 5. Extract limit (how many results to return)
        limit_patterns = [
            r'(?:show|display|give|tell)\s+(?:me\s+)?(?:the\s+)?(\d+)',
            r'(?:limit|top|first)\s+(\d+)',
            r'(\d+)\s+(?:examples|reviews|results)'
        ]
        
        for pattern in limit_patterns:
            limit_match = re.search(pattern, query_lower)
            if limit_match:
                result["limit"] = int(limit_match.group(1))
                break
        
        logger.debug(f"Enhanced fallback parser result: {json.dumps(result, indent=2)}")
        return result


def parse_query_with_function_calling(query: str) -> Dict[str, Any]:
    """
    Use OpenAI function calling to extract structured filters from user queries.
    Returns a dictionary with the extracted information.
    
    This uses the function calling API to ensure we get properly structured output
    without having to parse free-form text from the LLM.
    """
    # Define the function schema for query parsing
    functions = [
        {
            "name": "parse_query",
            "description": "Extract intent, date range, and filters from user query about reviews",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["count", "latest", "analysis", "retrieve", "time_distribution", "peak_time"]
                        },
                        "description": "List of actions the user wants: count (statistics), latest (recent items), analysis (summary), retrieve (content), time_distribution (show counts by time period), peak_time (show peak time period)"
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["year", "month", "week", "day", "weekday"],
                        "description": "Time granularity to group results by (year, month, day, or weekday)"
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ["distribution", "peak"],
                        "description": "Type of aggregation: total count per group or peak (highest) group"
                    },
                    "complaints": {
                        "type": "boolean",
                        "description": "True if user asks for customer complaints (negative reviews)"
                    },
                    "include_examples": {
                        "type": "boolean",
                        "description": "If true, user wants to see examples of reviews along with count or statistics"
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string", "format": "date"},
                            "end": {"type": "string", "format": "date"}
                        },
                        "description": "Absolute date range to filter reviews by"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Exact rating to filter on (if any)"
                    },
                    "sentiments": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"]
                        },
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": "One or more sentiment filters to apply"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"],
                        "description": "Legacy sentiment filter (single value)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of reviews requested"
                    },
                    "semantic_query": {
                        "type": "string",
                        "description": "What to semantically search for"
                    }
                },
                "required": ["actions", "semantic_query"]
            }
        }
    ]
    
    # Call the API with function calling enabled
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "system", "content": f"TODAY'S DATE IS: {today}"},
            {"role": "user", "content": query}
        ]
        response = client.chat.completions.create(
            model=QUERY_PARSER_MODEL,
            messages=messages,
            functions=functions,
            function_call={"name": "parse_query"}
        )
        logger.debug(f"Function calling using today's date: {today}")
        
        # Extract the function call arguments
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        logger.info(f"LLM function parsing response: {json.dumps(function_args, indent=4)}")
        
        # Convert to our internal format
        result = {
            # Set both core_query and semantic_query to ensure consistency
            "core_query": function_args["semantic_query"],
            "semantic_query": function_args["semantic_query"],
        }
        
        # Handle multiple actions if present
        if "actions" in function_args and isinstance(function_args["actions"], list):
            # Convert actions to our internal query_types
            action_to_type = {
                "count": "count",
                "latest": "latest",
                "analysis": "retrieval",
                "retrieve": "retrieval"
            }
            result["actions"] = [action_to_type.get(action, "retrieval") for action in function_args["actions"]]
            
            # For backward compatibility, set primary query_type to first action
            if result["actions"]:
                result["query_type"] = result["actions"][0]
            else:
                result["query_type"] = "retrieval"
        else:
            # Backward compatibility for single action
            action = function_args.get("action", "retrieve")
            action_to_type = {
                "count": "count",
                "latest": "latest",
                "analysis": "retrieval",
                "retrieve": "retrieval"
            }
            result["query_type"] = action_to_type.get(action, "retrieval")
            result["actions"] = [result["query_type"]]
            
        # Check if examples are requested
        if "include_examples" in function_args:
            result["include_examples"] = function_args["include_examples"]
        
        # Handle date range
        dr = function_args.get("date_range", {})
        if dr and "start" in dr:
            result["date_start"] = dr["start"]
        if dr and "end" in dr:
            result["date_end"] = dr["end"]
        
        # Handle other filters
        if "rating" in function_args:
            result["rating"] = function_args["rating"]
        if "sentiment" in function_args:
            result["sentiment"] = function_args["sentiment"]
        if "limit" in function_args:
            result["max_results"] = function_args["limit"]
        
        # Handle time aggregation parameters
        if "group_by" in function_args:
            result["group_by"] = function_args["group_by"]
        if "aggregate" in function_args:
            result["aggregate"] = function_args["aggregate"]
        if "complaints" in function_args and function_args["complaints"]:
            # If user explicitly asked for complaints, set sentiment to negative
            result["sentiment"] = "negative"
            result["sentiments"] = ["negative"]  # Ensure both fields are set
            result["complaints"] = True
            
        # If LLM has already given us both group_by & aggregate, FORCE a time query
        gb = function_args.get("group_by")
        ag = function_args.get("aggregate")
        if gb and ag:
            # peak_time for "which X has the most…", time_distribution otherwise
            action = "peak_time" if ag == "peak" else "time_distribution"
            result["actions"] = [action]
            result["query_type"] = action
            # ensure we carry them forward
            result["group_by"] = gb
            result["aggregate"] = ag
            # if they asked for "complaints" map it to negative sentiment
            if function_args.get("complaints"):
                result["sentiment"] = "negative"
                result["sentiments"] = ["negative"]
            # done—no more normal count/retrieval for this query
            logger.info(f"Detected time aggregation query with group_by={gb} and aggregate={ag}, setting action={action}")
            return result
        
        # Check for relative date references that might have been missed
        if ("date_start" not in result or "date_end" not in result):
            # Call our existing interpret_relative_date function with explicit current_date
            current_date = datetime.datetime.now()  # Use current date explicitly
            logger.debug(f"Explicitly using current date for relative date parsing: {current_date.strftime('%Y-%m-%d')}")
            start_date, end_date = interpret_relative_date(query, current_date=current_date)
            if start_date and end_date:
                result["date_start"] = start_date.date().isoformat()
                result["date_end"] = end_date.date().isoformat()
                logger.info(f"Detected relative date: {result['date_start']} to {result['date_end']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in function calling parsing: {e}")
        # Fall back to the original parser
        fallback = parse_query_with_llm(query)
        # Ensure semantic_query exists to prevent KeyError
        fallback.setdefault("semantic_query", fallback.get("core_query", query))
        return fallback

# ┌─────────────────────────────────────────────────────────────────────
# │ Chunk 1: add unified parser

def parse_user_query(query: str) -> Dict[str,Any]:
    """
    Standardize query parsing and interpretation.
    
    This function first uses the function-calling parser to extract structured information,
    then applies relative date interpretation as needed.
    
    Returns a dictionary with standard keys.
    """
    # Parse the query
    parsed = parse_query_with_function_calling(query)
    
    # If no explicit date range, try to interpret relative dates
    if not parsed.get("date_start") and not parsed.get("date_end"):
        start_date, end_date = interpret_relative_date(query)
        if start_date and end_date:
            parsed["date_start"] = start_date.date().isoformat()
            parsed["date_end"] = end_date.date().isoformat()
    
    # Return standardized format
    result = {
        "query_type": parsed.get("query_type", "retrieval"),
        "semantic_query": parsed.get("semantic_query", parsed.get("core_query", query)),
        "date_start": parsed.get("date_start"),
        "date_end": parsed.get("date_end"),
        "rating": parsed.get("rating"),
        "sentiment": parsed.get("sentiment"),  # legacy field
        "sentiments": parsed.get("sentiments", []),
        "limit": parsed.get("max_results"),
        # Add the new fields for multi-intent
        "actions": parsed.get("actions", [parsed.get("query_type", "retrieval")]),
        "include_examples": True,  # ALWAYS include examples no matter what
        # Add the new fields for time-based aggregations
        "group_by": parsed.get("group_by"),
        "aggregate": parsed.get("aggregate"),
        "complaints": parsed.get("complaints", False)
    }
    
    # FALLBACK: detect "which <period>" patterns
    m = re.search(r'which\s+(year|month|day|weekday)s?\s+has\s+the\s+most', query.lower())
    if m:
        period = m.group(1)
        logger.info(f"Detected 'which {period} has the most' pattern in query: {query}")
        result.update({
            "actions": ["peak_time"],
            "query_type": "peak_time",
            "group_by": period,
            "aggregate": "peak",
            # If they said "complaints" or "negative", treat as negative reviews
            "sentiment": "negative" if "complaint" in query.lower() or "negative" in query.lower() else 
                        "positive" if "positive" in query.lower() else None
        })
        # If complaints mentioned, set both fields
        if "complaint" in query.lower() or "negative" in query.lower():
            result["sentiments"] = ["negative"]
            result["complaints"] = True
        return result
        
    # FALLBACK: pick up any positive/negative tokens we missed
    sentiments = find_sentiment_tokens(query)
    if sentiments and not result.get("sentiments"):
        result["sentiments"] = sentiments
        result["sentiment"] = sentiments[0]  # legacy single-sentiment field
        logger.info(f"Detected sentiments from fallback: {sentiments}")
    
    # FALLBACK: treat "comparison" as a count+retrieve request
    q = query.lower()
    if "compare" in q or "comparison" in q or "vs" in q or "versus" in q:
        result["query_type"] = "count"
        result["actions"] = ["count", "retrieval"]
        logger.info(f"Detected comparison query, setting actions to count+retrieval")
    
    # Force examples on for any count query
    if result["query_type"] == "count" or "count" in result["actions"]:
        result["include_examples"] = True
    
    # Extract multiple sentiments using typo-tolerant detection
    sentiments = find_sentiment_tokens(query)
    
    # Fall back to singular sentiment if available and no sentiments found
    if not sentiments and result.get("sentiment"):
        # Check if the single sentiment needs typo correction
        single_sentiment = result["sentiment"].lower()
        matches = difflib.get_close_matches(single_sentiment, VALID_SENTIMENTS, cutoff=0.75)
        if matches:
            sentiments = [matches[0]]
        else:
            sentiments = [single_sentiment]
    
    # If still no sentiments, try to infer from rating mentioned in query
    if not sentiments:
        # Try to extract rating from the query
        extracted_rating = extract_numeric_rating(query)
        if extracted_rating is not None:
            inferred_sentiment = rating_to_sentiment(extracted_rating)
            if inferred_sentiment:
                sentiments = [inferred_sentiment]
                logger.debug(f"Inferred sentiment '{inferred_sentiment}' from rating {extracted_rating}")
    
    logger.debug(f"Final detected sentiments: {sentiments}")
    result["sentiments"] = sentiments
    
    # Ensure the legacy sentiment field is always populated if we have sentiments
    if sentiments and not result.get("sentiment"):
        result["sentiment"] = sentiments[0]
        logger.debug(f"Setting legacy sentiment field to '{result['sentiment']}' from sentiments list")
    
    # If no date range was parsed, use the full span of our data
    if result["date_start"] is None and result["date_end"] is None:
        logger.debug("No date range specified in query, applying default date range")
        result["date_start"] = "2011-01-01"  # Adjust based on earliest data
        result["date_end"] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # If no limit was specified:
    if result["limit"] is None:
        #  • For pure count or analysis/statistics queries → leave unlimited
        if result["query_type"] in ("count", "analysis"):
            result["limit"] = None
        #  • Otherwise (retrieval/latest) → cap at 20 to bound LLM context
        else:
            logger.debug("No limit specified for retrieval/latest query, defaulting to 20")
            result["limit"] = 20
    
    # Handle compound queries    # Handle multi-intent queries - look for compound phrases that indicate the user wants examples
    # with their count/analysis
    compound_phrases = [
        "also show", "also tell", "with example", "with some example", 
        "show me some", "tell me some", "and show me", "along with"
    ]
    
    if any(phrase in query.lower() for phrase in compound_phrases):
        if "retrieve" not in result["actions"]:
            result["actions"].append("retrieve")
        result["include_examples"] = True
        
    # Set include_examples to true by default for count queries - users always want examples
    if "count" in result["actions"] and result.get("include_examples") is None:
        result["include_examples"] = True
        logger.info("Automatically enabling examples for count query for better user experience")
    
    return result

# ┌─────────────────────────────────────────────────────────────────────
# │ Time-based aggregation utilities
# └─────────────────────────────────────────────────────────────────────

def metadata_to_df(metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert metadata list to a pandas DataFrame with normalized columns
    for easier aggregation and filtering.
    """
    # Create DataFrame from metadata
    df = pd.DataFrame(metadata)
    
    # Normalize date column
    df["_parsed_date"] = pd.to_datetime(df["_parsed_date"], errors="coerce")
    
    # Ensure sentiment column exists
    if "sentiment" not in df.columns:
        # Extract sentiment from full_review_json or infer from rating
        df["sentiment"] = df.apply(
            lambda row: 
            # First try to get from full_review_json
            (row.get("full_review_json", {}).get("sentimentAnalysis") 
            # Then try to infer from rating
            or rating_to_sentiment(float(row.get("__rating") 
                                    or row.get("rating") 
                                    or row.get("full_review_json", {}).get("ratingValue") 
                                    or 0))),
            axis=1
        )
    
    # Create derived time columns for aggregation
    df["year"] = df["_parsed_date"].dt.year
    df["month"] = df["_parsed_date"].dt.month
    df["day"] = df["_parsed_date"].dt.day
    df["weekday"] = df["_parsed_date"].dt.day_name()
    df["yearmonth"] = df["_parsed_date"].dt.strftime("%Y-%m")
    df["date_str"] = df["_parsed_date"].dt.strftime("%Y-%m-%d")
    
    logger.info(f"Converted {len(df)} metadata records to DataFrame with columns: {df.columns.tolist()}")
    return df

def time_distribution(df: pd.DataFrame, group_by: str = "year", sentiment: Optional[str] = None) -> pd.Series:
    """
    Return counts of reviews indexed by year/month/day/weekday.
    
    Args:
        df: DataFrame with normalized columns
        group_by: Time granularity (year, month, day, weekday, yearmonth)
        sentiment: Optional filter for sentiment (positive, negative, neutral)
    
    Returns:
        Series with counts by time period
    """
    # Filter by sentiment if specified
    filtered_df = df.copy()
    if sentiment:
        # Case-insensitive comparison
        filtered_df = filtered_df[filtered_df["sentiment"].str.lower() == sentiment.lower()]
        logger.info(f"Filtered to {len(filtered_df)} reviews with sentiment: {sentiment}")
    
    # Group by the specified time granularity
    if group_by in ["year", "month", "day", "weekday", "yearmonth", "date_str"]:
        # Handle different formatting requirements for each grouping
        if group_by == "yearmonth":
            # Format as YYYY-MM
            dist = filtered_df.groupby("yearmonth").size().sort_index()
        elif group_by == "date_str":
            # Format as YYYY-MM-DD
            dist = filtered_df.groupby("date_str").size().sort_index()
        elif group_by == "month":
            # Convert numeric months to month names
            month_names = {i: calendar.month_name[i] for i in range(1, 13)}
            counts = filtered_df.groupby("month").size()
            dist = pd.Series([counts.get(i, 0) for i in range(1, 13)], 
                              index=[month_names[i] for i in range(1, 13)])
        elif group_by == "weekday":
            # Ensure weekdays are in correct order (Monday first)
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            counts = filtered_df.groupby("weekday").size()
            dist = pd.Series([counts.get(day, 0) for day in days], index=days)
        else:
            # Simple grouping for year and day
            dist = filtered_df.groupby(group_by).size().sort_index()
    else:
        raise ValueError(f"Invalid group_by: {group_by}. Must be one of: year, month, day, weekday, yearmonth, date_str")
    
    logger.info(f"Generated time distribution by {group_by} with {len(dist)} groups")
    return dist

# Map human period → pandas frequency
_FREQ = {
    "day":   "D",
    "week":  "W-MON",  # weeks starting on Monday
    "month": "M",
    "year":  "A",
    "weekday": None  # Special case handled differently
}

def peak_time_by_period(df: pd.DataFrame, date_field: str = "_parsed_date", period: str = "year", 
                       sentiment: str = None, start_date = None, end_date = None):
    """Ultra-efficient pandas-based peak time detection.
    
    Args:
      - df: DataFrame with review data, having datetime values in date_field
      - date_field: column name that holds datetime values
      - period: one of "day", "week", "month", "year", "weekday"
      - sentiment: optional sentiment filter (positive, negative, neutral)
      - start_date: optional start date for filtering (pd.Timestamp or string)
      - end_date: optional end date for filtering (pd.Timestamp or string)
    
    Returns:
      (peak_label, peak_count, full_series)
    """
    # Make sure dates are parsed properly
    df = df.copy()
    df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
    df = df.dropna(subset=[date_field])
    
    # Exit early if no data
    if len(df) == 0:
        return "No data", 0, pd.Series()
    
    # Apply sentiment filtering BEFORE indexing for proper complaint handling
    if sentiment:
        if 'sentiment' in df.columns:
            logger.info(f"Filtering by sentiment: '{sentiment}' before time aggregation")
            df = df[df['sentiment'].str.lower() == sentiment.lower()]
            if len(df) == 0:
                logger.warning(f"No reviews found with sentiment '{sentiment}'")
                return "No data", 0, pd.Series()
        else:
            logger.warning("Sentiment column not found in DataFrame")
    
    # Set date index for super-fast slicing and grouping
    df = df.set_index(date_field).sort_index()
    
    # Apply date range filtering if specified
    if start_date is not None or end_date is not None:
        df = df.loc[start_date:end_date]
        if len(df) == 0:
            logger.warning("No reviews found in the specified date range")
            return "No data", 0, pd.Series()
    
    # Special handling for weekday
    if period == "weekday":
        weekday_counts = df.index.day_name().value_counts().sort_index()
        if len(weekday_counts) == 0:
            return "No data", 0, pd.Series()
        peak_label = weekday_counts.idxmax()
        peak_count = int(weekday_counts.max())
        return peak_label, peak_count, weekday_counts
    
    # Check if period is valid
    if period not in _FREQ:
        logger.warning(f"Invalid period: '{period}'. Defaulting to 'year'")
        period = "year"
    
    # Group by the chosen period using pandas Grouper (C-optimized)
    counts = df.groupby(pd.Grouper(freq=_FREQ[period])).size()
    
    if len(counts) == 0:
        logger.warning("No data available for peak time analysis")
        return "No data", 0, pd.Series()
    
    # Find the peak in one pass
    peak_idx = counts.idxmax()
    peak_count = int(counts.max())
    
    # Format the period label beautifully
    if period == "year":
        peak_label = str(peak_idx.year)
    elif period == "month":
        peak_label = peak_idx.strftime("%Y-%m")
    elif period == "week":
        peak_label = peak_idx.strftime("%Y-W%W")  # Year-Week format
    else:  # day
        peak_label = peak_idx.strftime("%Y-%m-%d")
    
    return peak_label, peak_count, counts

def peak_time(dist: pd.Series) -> Tuple[Any, int]:
    """Find the peak period in the time distribution.
    
    Args:
        dist: Series with counts by time period
        
    Returns:
        Tuple of (peak_label, peak_count)
    """
    if len(dist) == 0:
        return None, 0
        
    peak_label = dist.idxmax()
    peak_count = int(dist.loc[peak_label])
    
    logger.debug(f"Peak time identified: {peak_label} with {peak_count} reviews")
    return peak_label, peak_count

# └─────────────────────────────────────────────────────────────────────

# File paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(CURRENT_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "reviews_faiss_index.index")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "reviews_metadata.pkl")

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o-mini"  # Can be changed to gpt-4o or other preferred model

def normalize_metadata(metadata):
    """
    Normalize metadata by attaching standard fields to all reviews.
    This ensures every review has _parsed_date, __text, and __rating fields.
    """
    logger.info(f"Normalizing metadata for {len(metadata)} reviews")
    for r in metadata:
        # 1) Parse date
        # Try full_review_json, then original_row, then top-level
        raw = r.get("full_review_json") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        
        date_str = raw.get("date") \
                or r.get("original_row", {}).get("date") \
                or r.get("date")
        
        try:
            r["_parsed_date"] = date_parser.parse(date_str) if date_str else None
        except (ValueError, TypeError):
            r["_parsed_date"] = None
            
        # 2) Extract full text
        r["__text"] = raw.get("ratingText") \
                    or raw.get("text") \
                    or r.get("original_row", {}).get("reviewText") \
                    or r.get("text") \
                    or ""
        
        # 3) Extract rating
        r["__rating"] = raw.get("ratingValue") \
                        or r.get("rating") \
                        or r.get("original_row", {}).get("ratingValue") \
                        or "N/A"
    
    return metadata

def load_faiss_index():
    """Load the FAISS index and metadata from disk"""
    try:
        # Load the FAISS index and metadata
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors.")
        
        # Normalize metadata right after loading
        metadata = normalize_metadata(metadata)
        
        return index, metadata
    except FileNotFoundError:
        logger.error(f"Index files not found. Please run embedding_generator.py first.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading index: {e}")
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
        logger.error(f"Error generating embedding: {e}")
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
                logger.warning(f"Failed to parse date '{date_str}': {e}")
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

def retrieve_relevant_documents(query: str, index, metadata: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve top_k most relevant documents for the query using hybrid retrieval with LLM parsing."""
    # Use LLM to extract structured filters and the core semantic query
    parsed_query = parse_user_query(query)
    
    # Extract filters from parsed query
    date_start = parsed_query.get("date_start")
    date_end = parsed_query.get("date_end")
    rating_filter = parsed_query.get("rating")
    sentiment = parsed_query.get("sentiment")  # Legacy single sentiment
    sentiments = parsed_query.get("sentiments", [])  # List of sentiments
    
    # We'll handle the 'with text' filter along with other filters in a centralized way below
    # Set with_text flag based on both explicit flag and query text
    if "with text" in query.lower() and not parsed_query.get("with_text", False):
        parsed_query["with_text"] = True
    semantic_query = parsed_query["semantic_query"]
    
    # Log detected filters for debugging
    date_start = parsed_query.get("date_start")
    date_end = parsed_query.get("date_end")
    rating_filter = parsed_query.get("rating")
    with_text = parsed_query.get("with_text", False)
    
    if date_start or date_end:
        logger.info(f"Detected date range filter: {date_start} to {date_end}")
    if rating_filter is not None:
        logger.info(f"Detected rating filter: {rating_filter}")
    if sentiment or sentiments:
        logger.info(f"Detected sentiment filter: {sentiment or sentiments}")
    if with_text:
        logger.info("Query specifies 'with text' - will filter for reviews with text content")
    
    # 1) First do semantic search to get initial candidates
    normalized_query = normalize_for_search(semantic_query)
    query_embedding = embed_text(normalized_query)
    initial_results = []
    
    if query_embedding is not None:
        # Get more results than needed for filtering
        search_top_k = min(top_k * 5, 100)
        D, I = index.search(np.array([query_embedding]), search_top_k)
        
        # Convert to result objects with normalized fields
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
                
            review = metadata[idx]
            initial_results.append({
                "text": review.get("__text", ""),
                "date": review.get("_parsed_date"),
                "rating": review.get("__rating"),
                "distance": float(dist),
                "original_index": idx,
                # Keep original data for context building
                "full_review_json": review.get("full_review_json", {}),
                "original_row": review.get("original_row", {}),
                # Copy any other fields we might need
                "locationId": review.get("locationId"),
                "reviewer": review.get("reviewerTitle", "Unknown"),
                "combined_text": review.get("combined_text")
            })
    
    # 2) Apply filters on normalized fields
    filtered_results = []
    for result in initial_results:
        # Date filter
        if date_start and result["date"] and result["date"] < date_parser.parse(date_start):
            continue
        if date_end and result["date"] and result["date"] > date_parser.parse(date_end):
            continue
            
        # Rating filter
        if rating_filter is not None and str(result["rating"]) != str(rating_filter):
            continue
            
        # Sentiment filter
        if sentiment or sentiments:
            # Define helper function to check if a review matches the sentiment criteria
            def matches_sentiment(review):
                # 1. Try to get explicit sentimentAnalysis from full_review_json
                s = None
                if isinstance(review.get("full_review_json"), dict):
                    s = review["full_review_json"].get("sentimentAnalysis")
                    
                # 2. If not found, infer from rating
                if not s:
                    rv = None
                    # Try multiple places for rating value
                    if isinstance(review.get("full_review_json"), dict):
                        rv = review["full_review_json"].get("ratingValue")
                    if rv is None:
                        rv = review.get("rating") or review.get("__rating")
                    
                    # Convert rating to sentiment if possible
                    try:
                        if rv is not None:
                            s = rating_to_sentiment(float(rv))
                    except:
                        logger.debug(f"Failed to convert rating {rv} to sentiment")
                        
                # If we couldn't determine sentiment, this doesn't match
                if not s:
                    return False
                    
                # Check if it matches any requested sentiment
                if sentiments:
                    return s.lower() in [sent.lower() for sent in sentiments]
                elif sentiment:
                    return s.lower() == sentiment.lower()
                return True
            
            # Apply the sentiment filter
            if not matches_sentiment(result):
                logger.debug(f"Filtering out review that didn't match sentiment criteria: {result.get('rating')}")
                continue
            
        # Text filter if requested
        if with_text and (not result["text"] or len(result["text"].strip()) < 5):
            continue
            
        filtered_results.append(result)
    
    # 3) Sort by relevance and return top-k
    filtered_results.sort(key=lambda x: x.get("distance", float('inf')))
    final_results = filtered_results[:top_k]
    
    logger.info(f"Retrieved {len(initial_results)} initial results, {len(filtered_results)} after filtering, returning {len(final_results)} final results")
    return final_results

# No need for wrapper - this is the single consolidated retrieval function

def handle_user_query(query, index=None, metadata=None, max_results=15):
    """
    Main dispatcher for handling review queries.
    
    Parses the query, routes to appropriate handler, and returns results.
    
    Args:
        query: Raw user query string
        index: FAISS index for vector search
        metadata: Review metadata list
        max_results: Maximum results to return
    
    Returns:
        Response string with results or error message
    """
    # Load index and metadata if not provided
    if index is None or metadata is None:
        try:
            index, metadata = load_faiss_index()
            logger.info(f"Loaded index and {len(metadata)} metadata records")
        except Exception as e:
            error_message = f"Error loading index: {e}"
            logger.error(error_message)
            return error_message
    
    try:
        # Parse the query to determine intent and extract filters
        parsed_query = parse_user_query(query)
        actions = parsed_query.get("actions", [parsed_query.get("query_type", "retrieval")])
        
        # Log the parsed query and actions for debugging
        logger.info(f"Parsed query parameters: {json.dumps({k: v for k, v in parsed_query.items() if v is not None}, indent=2)}")
        logger.info(f"Will execute {len(actions)} actions: {actions}")
        
        responses = []
        
        # Execute the requested actions in order
        for action in actions:
            if action == "count":
                result = handle_count_query(query, index, metadata, max_results)
                responses.append(result)
                
            elif action == "latest":
                result = handle_latest_query(query, index, metadata, max_results)
                responses.append(result)
            
            elif action == "peak_time" or action == "time_distribution":
                # Route to time-based aggregation handler
                result = handle_time_query(query, index, metadata, max_results)
                responses.append(result)
                
            elif action == "time_aggregation":
                # Route to time-based aggregation handler
                result = handle_time_query(query, index, metadata, max_results)
                responses.append(result)
                
            else:  # Default is retrieval-based query
                result = handle_retrieval_query(query, index, metadata, max_results)
                responses.append(result)
        
        # Combine responses if multiple actions were performed
        if len(responses) == 1:
            return responses[0]
        else:
            # Join multiple responses with clear section headers
            combined = ""
            for i, (action, response) in enumerate(zip(actions, responses)):
                # Add a horizontal rule between sections except for the first one
                if i > 0:
                    combined += "\n\n---\n\n"
                
                # Add section header based on action type
                if action == "count":
                    combined += "## Review Statistics\n\n"
                elif action == "latest":
                    combined += "## Latest Reviews\n\n"
                elif action == "peak_time":
                    combined += "## Peak Time Analysis\n\n"
                elif action == "time_distribution":
                    combined += "## Time Distribution Analysis\n\n"
                elif action == "time_aggregation":
                    combined += "## Time Aggregation Analysis\n\n"
                else:  # retrieval
                    combined += "## Relevant Reviews\n\n"
                
                combined += response
            
            return combined
        
    except Exception as e:
        error_message = f"Error handling query: {str(e)}"
        logger.error(f"{error_message}\nTraceback: {traceback.format_exc()}")
        return error_message


def handle_latest_query(query: str, index=None, metadata=None, max_results: int = 5) -> str:
    """
    Handle a query about the latest/recent reviews.
    
    Args:
        query: The user query
        index: The FAISS index for vector search
        metadata: The metadata associated with the index
        max_results: Maximum number of results to return
        
    Returns:
        A string response to the user's query
    """
    try:
        logger.info(f"Handling query for latest reviews...")
        
        # Ensure max_results has a default value if None
        if max_results is None:
            max_results = 5  # Use default value if not specified
            
        # Parse the query - use the unified parser
        parsed_query = parse_user_query(query)
        logger.info(f"[LATEST QUERY] Parsed query: {json.dumps(parsed_query, indent=2)}")
        
        # Initialize empty list for items to return
        all_items = []
        
        # Apply date/rating/sentiment filters if present
        date_start = parsed_query.get("date_start")
        date_end = parsed_query.get("date_end")
        rating = parsed_query.get("rating")
        sentiment = parsed_query.get("sentiment")
        
        # Special handling for "with text" queries
        with_text = "with text" in query.lower()
        if with_text:
            logger.debug("Special handling for 'with text' in latest query")
        
        # First pass: Parse all dates in metadata once
        logger.info(f"[LATEST QUERY] Processing {len(metadata)} total reviews...")
        parsed_metadata = []
        for item in metadata:
            # Try to extract date from multiple possible locations
            date_str = None
            
            # First try full_review_json
            if "full_review_json" in item:
                try:
                    if isinstance(item["full_review_json"], dict):
                        review_json = item["full_review_json"]
                    else:
                        review_json = json.loads(item["full_review_json"])
                    date_str = review_json.get("date")
                except:
                    pass
            
            # Then try original_row
            if not date_str and "original_row" in item and "date" in item["original_row"]:
                date_str = item["original_row"]["date"]
            
            # Finally try direct date field
            if not date_str and "date" in item:
                date_str = item["date"]
            
            if date_str:
                try:
                    parsed_date = date_parser.parse(date_str)
                    item["_parsed_date"] = parsed_date
                    parsed_metadata.append(item)
                except Exception as e:
                    # Skip items with unparseable dates
                    continue
        
        # Second pass: Apply filters to pre-parsed items
        for item in parsed_metadata:
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
                all_items.append(item)
        
        logger.info(f"[LATEST QUERY] Found {len(all_items)} reviews matching criteria")
        
        # If "with text" is specified, filter out reviews without text
        if with_text:
            text_reviews = [item for item in all_items if has_text_content(item) == 2]
            logger.info(f"[LATEST QUERY] Found {len(text_reviews)} reviews with text out of {len(all_items)} matching reviews")
            
            # If we found reviews with text, use those. Otherwise, fall back to the original filtered reviews
            if text_reviews:
                all_items = text_reviews
        
        # Check if we're looking for oldest or newest reviews
        is_oldest_query = False
        if date_start and isinstance(date_start, str) and date_start.lower() == 'oldest':
            is_oldest_query = True
        elif date_end and isinstance(date_end, str) and date_end.lower() == 'oldest':
            is_oldest_query = True
        elif 'oldest' in query.lower() or 'first' in query.lower() or 'early' in query.lower():
            is_oldest_query = True
        
        # Sort by date - newest first or oldest first based on query type
        sorted_items = []
        if all_items:
            sorted_items = sorted(all_items, key=lambda x: x.get("_parsed_date", datetime.datetime.min), 
                             reverse=not is_oldest_query)  # reverse=False for oldest first
            
            # Log the appropriate message based on query type
            if is_oldest_query:
                logger.info(f"[LATEST QUERY] Found {len(sorted_items)} reviews, showing oldest {min(max_results, len(sorted_items))}")
            else:
                logger.info(f"[LATEST QUERY] Found {len(sorted_items)} reviews, showing latest {min(max_results, len(sorted_items))}")
        
            # Take the top N reviews based on the requested max_results
            latest_items = sorted_items[:max_results]
            
            # Debug: Check what we're sending to prepare_context
            sample_item = latest_items[0] if latest_items else None
            if sample_item:
                logger.debug(f"[LATEST QUERY] Sample item keys before context prep: {list(sample_item.keys())}")
                if "full_review_json" in sample_item:
                    try:
                        if isinstance(sample_item["full_review_json"], dict):
                            logger.debug(f"[LATEST QUERY] Sample review has text: {'ratingText' in sample_item['full_review_json']}")
                        else:
                            review_json = json.loads(sample_item["full_review_json"])
                            logger.debug(f"[LATEST QUERY] Sample review has text: {'ratingText' in review_json}")
                    except:
                        pass
            
            # Prepare context from sorted items
            context = prepare_context(latest_items, max_reviews=max_results, query=query)
            
            # Add explicit guidance for the LLM to include review text
            context_prefix = f"Found {len(latest_items)} {'oldest' if is_oldest_query else 'latest'} reviews. Be sure to include the review TEXT in your response:\n\n"
            context = context_prefix + context
            
            # Generate response
            response = refine_with_llm(query, context)
            return response
        else:
            return "There are no reviews available that match your criteria."
    except Exception as e:
        logger.error(f"Error in handle_latest_query: {e}")

def handle_time_query(query: str, index, metadata: List[Dict[str, Any]], max_results: int = None) -> str:
    """
    Handle time-based aggregation queries to find patterns in review distribution over time.
    
    This function handles queries like:
    - "In which year were most complaints posted?"
    - "What month has the most positive reviews?"
    - "On which day of the week do we get most negative feedback?"
    """
    # Parse the query using our unified parser
    parsed_query = parse_user_query(query)
    logger.info(f"[TIME QUERY] Processing with parameters: {json.dumps({k: v for k, v in parsed_query.items() if v is not None}, indent=2)}")
    
    # Extract query parameters
    # IMPORTANT: Ensure we ALWAYS have a valid group_by - default to "year" if None or invalid
    group_by = parsed_query.get("group_by")
    if group_by not in ["year", "month", "day", "week", "weekday"]:
        group_by = "year"  # More robust default
        logger.info(f"Missing or invalid group_by, defaulting to: {group_by}")
    
    aggregate = parsed_query.get("aggregate", "peak")  # Default to peak if not specified
    is_complaints = parsed_query.get("complaints", False)
    
    # Handle sentiment - if complaints specified, override to negative
    sentiment = "negative" if is_complaints else parsed_query.get("sentiment")
    if parsed_query.get("sentiments") and not sentiment:
        sentiment = parsed_query["sentiments"][0] if parsed_query["sentiments"] else None
        
    # Double-check complaint handling - IMPORTANT!
    if parsed_query.get("complaints", False) == True:
        # Force override to negative for complaints queries regardless of other settings
        sentiment = "negative"
        logger.info(f"Complaints query detected - forcing sentiment to '{sentiment}'")
    
    # Extract date range parameters
    start_date = None
    end_date = None
    
    if parsed_query.get("date_start"):
        # Handle timezone compatibility
        start_date = pd.to_datetime(parsed_query["date_start"]).tz_localize("UTC")
        logger.info(f"Using start date filter: {start_date}")
        
    if parsed_query.get("date_end"):
        # Handle timezone compatibility
        end_date = pd.to_datetime(parsed_query["date_end"]).tz_localize("UTC")
        logger.info(f"Using end date filter: {end_date}")
    
    # Convert metadata to DataFrame with optimized indexing
    df = metadata_to_df(metadata)
    sentiment_label = "complaints" if is_complaints else f"{sentiment if sentiment else 'all'} reviews"
    
    # Use the optimized O(n) peak_time_by_period function that handles all filtering in one pass
    if aggregate == "peak":
        # Find the peak time period using our optimized function with direct parameter passing
        peak_label, peak_count, time_series = peak_time_by_period(
            df=df, 
            period=group_by,
            sentiment=sentiment,
            start_date=start_date,
            end_date=end_date
        )
        
        if peak_label == "No data" or peak_count == 0:
            return "No reviews found matching your criteria."
        
        # Format the response
        response = f"The {group_by} with the most {sentiment_label} is **{peak_label}** with **{peak_count}** reviews."
        
        # Add examples if requested
        if parsed_query.get("include_examples", True):
            # Get example reviews matching the peak period and sentiment
            # Use datetime indexing for optimal filtering
            
            # First, create a filter mask based on the period
            # Note: We already have the peak time period from the time_series result
            peak_timestamp = None
            for idx, count in time_series.items():
                if count == peak_count:
                    peak_timestamp = idx
                    break
            
            if peak_timestamp is None:
                logger.warning("Could not find peak timestamp in time_series.")
                return response
            
            # Use efficient datetime-indexed filtering for the peak period
            if group_by == "year":
                # Filter reviews from the peak year
                start_of_period = pd.Timestamp(f"{peak_timestamp.year}-01-01")
                end_of_period = pd.Timestamp(f"{peak_timestamp.year}-12-31 23:59:59")
            elif group_by == "month":
                # Filter reviews from the peak month
                start_of_period = peak_timestamp.replace(day=1, hour=0, minute=0, second=0)
                # Last day of month (accounting for different month lengths)
                next_month = start_of_period + pd.DateOffset(months=1)
                end_of_period = next_month - pd.DateOffset(seconds=1)
            elif group_by == "week":
                # Filter reviews from the peak week
                start_of_period = peak_timestamp
                end_of_period = peak_timestamp + pd.DateOffset(days=6, hours=23, minutes=59, seconds=59)
            elif group_by == "weekday":
                # Special handling for weekday - filter all reviews on that weekday
                day_of_week = peak_timestamp.dayofweek
                mask = df["_parsed_date"].dt.dayofweek == day_of_week
                peak_df = df[mask]
            else:  # day
                # Filter reviews from the peak day
                start_of_period = peak_timestamp.replace(hour=0, minute=0, second=0)
                end_of_period = peak_timestamp.replace(hour=23, minute=59, second=59)
            
            # Create DataFrame with examples, using optimized filtering
            if group_by != "weekday":  # For all except weekday
                # Use efficient date-range filtering
                mask = (df["_parsed_date"] >= start_of_period) & (df["_parsed_date"] <= end_of_period)
                peak_df = df[mask]
                
            # Make sure we're working with the properly filtered DataFrame
            # Apply sentiment filter again for examples to ensure consistency
            if sentiment:
                # Explicitly filter for the requested sentiment
                logger.info(f"Explicitly filtering examples for sentiment: {sentiment}")
                peak_df = peak_df[peak_df["sentiment"].str.lower() == sentiment.lower()]
                if len(peak_df) == 0:
                    logger.warning(f"No examples with sentiment '{sentiment}' found in the peak period")
                    response += f"\n\nNo example reviews with {sentiment} sentiment found in this time period."
                    return response
            
            # Get up to 5 examples, sorted by date (most recent first) and verify they match sentiment
            examples_count = min(5, len(peak_df))
            if examples_count > 0:
                peak_df = peak_df.sort_values(by="_parsed_date", ascending=False).head(examples_count)
                
                # Verify sentiment of examples as a final check
                if sentiment:
                    logger.info(f"Examples sentiment verification - expected: {sentiment}")
                    for _, row in peak_df.iterrows():
                        actual_sentiment = row.get("sentiment")
                        logger.info(f"Example sentiment check: {actual_sentiment}")
                
                response += "\n\nHere are some example reviews from this period:\n\n"
                
                for i, (_, row) in enumerate(peak_df.iterrows(), 1):
                    # Extract review info with vectorized operations where possible
                    text = row.get("__text", "No text available")
                    date_str = row["_parsed_date"].strftime("%Y-%m-%d") if "_parsed_date" in row else "Unknown date"
                    rating = row.get("__rating", "Unknown")
                    
                    response += f"{i}. **Date:** {date_str}  \n"
                    response += f"   **Rating:** {rating}  \n"
                    response += f"   **Review:** {text[:200]}{'...' if len(text) > 200 else ''}  \n\n"
    else:  # aggregate == "distribution" 
        # Generate time distribution using the optimized function with direct parameter passing
        _, _, time_series = peak_time_by_period(
            df=df, 
            period=group_by,
            sentiment=sentiment,
            start_date=start_date,
            end_date=end_date
        )
        
        if len(time_series) == 0:
            return "No reviews found matching your criteria."
            
        # Show the distribution as a table
        sentiment_desc = f" {sentiment}" if sentiment else ""
        response = f"Distribution of{sentiment_desc} reviews by {group_by}:\n\n"
        
        # Format as a markdown table
        response += f"| {group_by.capitalize()} | Count |\n"
        response += "| --- | --- |\n"
        
        # Add rows for each time period
        for period, count in time_series.items():
            if group_by == "year":
                period_label = str(period.year)
            elif group_by == "month":
                period_label = period.strftime("%Y-%m")
            elif group_by == "week":
                period_label = period.strftime("%Y-W%W")
            elif group_by == "day":
                period_label = period.date().isoformat()
            else:
                period_label = str(period)
                
            response += f"| {period_label} | {count} |\n"
            
        # Add total
        total_reviews = time_series.sum()
        response += f"| **Total** | **{total_reviews}** |\n"
        
    # Add examples if requested
    if parsed_query.get("include_examples", True):
        # First apply the proper sentiment filter if specified
        if sentiment:
            logger.info(f"Filtering example reviews for time distribution by sentiment: {sentiment}")
            filtered_df = df[df["sentiment"].str.lower() == sentiment.lower()]
            if len(filtered_df) == 0:
                logger.warning(f"No examples with sentiment '{sentiment}' found in the specified period")
                response += f"\n\nNo example reviews with {sentiment} sentiment found in this time period."
                return response
        else:
            filtered_df = df
            
        # Get up to 5 examples (sorted by date, newest first) from the properly filtered DataFrame
        examples_df = filtered_df.sort_values(by="_parsed_date", ascending=False).head(5)
        
        # Debug logging to verify sentiment filtering worked
        if sentiment:
            logger.info(f"Selected {len(examples_df)} example reviews with sentiment '{sentiment}'")
            for _, row in examples_df.iterrows():
                logger.info(f"Example review sentiment: {row.get('sentiment')}, rating: {row.get('__rating')}")
        
        if len(examples_df) > 0:
            response += "\n\nHere are some example reviews:\n\n"
            
            for i, (_, row) in enumerate(examples_df.iterrows(), 1):
                # Extract review text and metadata with fallbacks
                text = row.get("__text", "No text available")
                date_str = row["_parsed_date"].strftime("%Y-%m-%d") if "_parsed_date" in row else "Unknown date"
                rating = row.get("__rating", "Unknown")
                
                response += f"{i}. **Date:** {date_str}  \n"
                response += f"   **Rating:** {rating}  \n"
                response += f"   **Review:** {text[:200]}{'...' if len(text) > 200 else ''}  \n\n"
    
    return response

def handle_count_query(query: str, index=None, metadata=None, max_results: int = 15) -> str:
    """
    Handle a count-type query.
    
    Args:
        query: The user query string
        index: The FAISS index for vector search
        metadata: The metadata associated with the index
        
    Returns:
        String response with the count information
    """
    # Parse the query using our unified parser
    parsed_query = parse_user_query(query)
    logger.info(f"[COUNT QUERY] Parsed query: {json.dumps(parsed_query, indent=2)}")
    
    # Extract date range and rating filters
    date_start = parsed_query.get("date_start")
    date_end = parsed_query.get("date_end")
    rating = parsed_query.get("rating")
    
    # Get the list of sentiments (may be multiple for multi-sentiment queries)
    sentiments = parsed_query.get("sentiments", [])
    single_sentiment = parsed_query.get("sentiment")
    
    # If we have an old-style single sentiment but no sentiments list, use it
    if not sentiments and single_sentiment:
        sentiments = [single_sentiment]
        
    # If we still have no sentiments, try to infer from rating if available
    if not sentiments and parsed_query.get("rating") is not None:
        inferred_sentiment = rating_to_sentiment(parsed_query["rating"])
        if inferred_sentiment:
            sentiments = [inferred_sentiment]
            logger.info(f"Inferred sentiment '{inferred_sentiment}' from rating {parsed_query['rating']}")
    
    logger.info(f"Handling count query with sentiments: {sentiments}")
    
    # Get whether examples should be included
    include_examples = parsed_query.get("include_examples", True)  # Default to showing examples
    logger.info(f"Include examples in count response: {include_examples}")
    
    # For multi-sentiment queries, prepare structures to track counts and examples per sentiment
    if len(sentiments) > 1:
        # 1) Compute counts for each sentiment
        counts = {}
        for sentiment in sentiments:
            filters = {
                'date_start': date_start,
                'date_end': date_end,
                'rating': rating,
                'sentiment': sentiment
            }
            count = count_matching_documents(metadata, filters)
            counts[sentiment] = count
            logger.info(f"Count for sentiment '{sentiment}': {count}")
        
        # 2) Find examples for each sentiment (up to 5 per sentiment if examples requested)
        examples = {}
        if include_examples:
            for sentiment in sentiments:
                # Filter metadata for this sentiment
                filtered_items = []
                for item in metadata:
                    date_match = True
                    if date_start or date_end:
                        date_match = filter_by_date(item, date_start, date_end)
                    
                    rating_match = filter_by_rating(item, rating) if rating else True
                    sentiment_match = filter_by_sentiment(item, sentiment)
                    
                    if date_match and rating_match and sentiment_match:
                        filtered_items.append(item)
                
                # Find items with text content (priority 2)
                text_items = [item for item in filtered_items if has_text_content(item) == 2]
                # Take up to 5 examples with text instead of just 3
                examples[sentiment] = text_items[:5]
            
        # 3) Format the multi-sentiment response
        resp = []
        date_range_text = f"**Counts from {date_start} to {date_end}:**"
        resp.append(date_range_text)
        
        for sentiment, count in counts.items():
            resp.append(f"- **{sentiment.title()} reviews:** {count}")
        
        # Only include examples if requested or by default
        if include_examples:
            resp.append("\n**Examples:**")
            
            for sentiment, items in examples.items():
                resp.append(f"\n__{sentiment.title()} reviews__:")
                if not items:
                    resp.append("No text-based examples found.")
                else:
                    for item in items:
                        # Format the example with date and truncated text
                        text = item.get("__text", item.get("ratingText", "[no text]"))
                        
                        # Get the date from _parsed_date or fall back to raw date
                        date_str = "[unknown date]"
                        if "_parsed_date" in item and item["_parsed_date"]:
                            date_str = item["_parsed_date"].date().isoformat()
                        elif "date" in item:
                            date_str = item["date"]
                        
                        # Add truncated text example
                        trunc_text = text[:150] + ("..." if len(text) > 150 else "")
                        resp.append(f"- ({date_str}) {trunc_text}")
        
        return "\n".join(resp)
    
    # Single sentiment path (original implementation)
    else:
        # Use the first sentiment if available, otherwise None
        sentiment = sentiments[0] if sentiments else None
        
        # Extract filters
        filters = {
            'date_start': date_start,
            'date_end': date_end,
            'rating': rating,
            'sentiment': sentiment
        }
        
        # Count matching documents
        count = count_matching_documents(metadata, filters)
        logger.info(f"Final count of matching documents: {count}")
        
        # Get samples of matching documents for context (showing more examples by default)
        matching_docs = []
        sample_size = min(10, count) if include_examples else 0  # Get up to 10 examples if requested
        logger.info(f"Will include up to {sample_size} examples in count response")
        sample_count = 0
        
        if count > 0 and sample_size > 0:
            logger.debug(f"Finding {sample_size} sample documents for count query context...")
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
            # Extract rating using normalized field or nested path
            rating = (
                doc.get("__rating")
                or doc.get("full_review_json", {}).get("ratingValue")
                or "unknown rating"
            )
            
            # Extract date using normalized field or nested path
            date_str = "unknown date"
            if doc.get("_parsed_date"):
                date_str = doc["_parsed_date"].date().isoformat()
            elif isinstance(doc.get("full_review_json"), dict):
                date_str = doc["full_review_json"].get("date", "unknown date")
            
            # Extract review text using normalized field or nested paths
            review_text = (
                doc.get("__text")
                or (isinstance(doc.get("full_review_json"), dict) and doc["full_review_json"].get("ratingText"))
                or (isinstance(doc.get("full_review_json"), dict) and doc["full_review_json"].get("text"))
                or "No review text available"
            )
            
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
        logger.info("Making direct call to OpenAI API for count query response...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a capable model for summarization
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5  # Lower temperature for more factual responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating count response with OpenAI API: {e}")
        # Fallback to basic response
        response = f"Found {count} reviews that match your criteria."
        if count > 0 and matching_docs:
            response += f"\n\nExample: {matching_docs[0].get('ratingText', 'No text available')}"
            
    return response

def handle_retrieval_query(query: str, index=None, metadata=None, max_results: int = 10) -> str:
    """
    Handle a retrieval-type query:
      1. Filter by date/rating/sentiment
      2. Iteratively pull more candidates from FAISS until we have
         enough reviews with actual text, or exhaust the pool.
      3. Fall back to minimal-content reviews if needed.
    
    Args:
        query: The user query
        index: The FAISS index for vector search
        metadata: The metadata associated with the index
        max_results: Maximum number of results to return
        
    Returns:
        A string response to the user's query
    """
    # Debug flag - set to True to print detailed information about filtered results
    DEBUG_REVIEWS = True
    
    # Parse query to get structured filters
    parsed = parse_user_query(query)
    print(f"[RETRIEVAL QUERY] Parsed query: {json.dumps(parsed, indent=2)}")
    
    core_query = parsed.get("core_query", query)
    semantic_query = parsed.get("semantic_query", core_query)
    date_start, date_end = parsed.get("date_start"), parsed.get("date_end")
    rating, sentiment = parsed.get("rating"), parsed.get("sentiment")
    limit = parsed.get("limit") or max_results
    
    # ------------------------------------------------------------------------
    # Special handling for "with text" queries (standalone or compound)
    # ------------------------------------------------------------------------
    if "with text" in query.lower():
        # Start with all reviews or apply date/rating/sentiment filters first
        filtered_items = []
        
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
        
        # Then filter for text content
        text_reviews = [
            r for r in filtered_items
            if isinstance(r.get("full_review_json"), dict)
            and r["full_review_json"].get("ratingText") is not None
            and r["full_review_json"].get("ratingText", "").strip()
        ]
        
        logger.info(f"Found {len(text_reviews)} reviews with text content after filtering")
        
        # If no reviews with text found, fall back to any that match filters
        if not text_reviews and filtered_items:
            print("No reviews with text found, falling back to any reviews matching filters")
            text_reviews = filtered_items[:limit]
        
        # If still nothing, return appropriate message
        if not text_reviews:
            if date_start or date_end:
                date_range = f"from {date_start}" if date_start else ""
                date_range += f" to {date_end}" if date_end else ""
                return f"Sorry, I couldn't find any reviews {date_range} with text."
            else:
                return "Sorry, I couldn't find any reviews that contain text."
                
        # prepare context and return response
        ctx = prepare_context(text_reviews, max_reviews=limit, query=query)
        return refine_with_llm(query, ctx)
    
    # ------------------------------------------------------------------------
    # Regular retrieval logic for non-"with text" queries
    # ------------------------------------------------------------------------
    print(f"[RETRIEVAL QUERY] Parsed query: {json.dumps(parsed, indent=2)}")
    
    core_query = parsed.get("core_query", query)
    semantic_query = parsed.get("semantic_query", core_query)
    date_start, date_end = parsed.get("date_start"), parsed.get("date_end")
    rating, sentiment = parsed.get("rating"), parsed.get("sentiment")
    limit = parsed.get("limit") or max_results
    
    # Track date handling information for better debugging
    date_handling_info = {
        "source": "unified_parser",
        "original_query": query,
        "parser_dates": {"start": date_start, "end": date_end},
    }
    
    # Note: No need for additional date interpretation as parse_user_query already handles this
    
    # Phase 1: Get all documents that match the filters
    logger.info(f"Phase 1: Filtering all documents based on criteria...")
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
    
    logger.info(f"Found {len(filtered_items)} documents matching all filters")
    
    # If no filters were applied or few results, consider all documents
    if not (date_start or date_end or rating or sentiment) and len(filtered_items) == 0:
        print("No filters applied and no results found, using all documents")
        filtered_items = metadata
        filtered_ids = list(range(len(metadata)))
    
    # If we don't have enough matching documents, return what we have
    if len(filtered_items) == 0:
        context = "No documents found matching the specified filters."
        return refine_with_llm(query, context)
    
    # Debug inspection of filtered items to understand their structure
    if DEBUG_REVIEWS and len(filtered_items) > 0:
        # Print structure of the first filtered item to inspect its fields
        print("\n=== DEBUG: INSPECTING FILTERED ITEM STRUCTURE ===")
        sample_item = filtered_items[0]
        print(f"Sample item keys: {list(sample_item.keys())}")
        
        # Check if there's review text in standard locations
        if 'text' in sample_item:
            print(f"Text field exists: '{sample_item['text'][:100]}...'")
        elif 'original_row' in sample_item and 'text' in sample_item['original_row']:
            print(f"Original row text field exists: '{sample_item['original_row']['text'][:100]}...'")
        
        # Check for full_review_json structure if it exists
        if 'full_review_json' in sample_item:
            print(f"full_review_json type: {type(sample_item['full_review_json'])}")
            if isinstance(sample_item['full_review_json'], dict):
                print(f"full_review_json keys: {list(sample_item['full_review_json'].keys())}")
                
                # Check specifically for ratingText
                if 'ratingText' in sample_item['full_review_json']:
                    rating_text = sample_item['full_review_json']['ratingText']
                    if rating_text:
                        print(f"full_review_json ratingText: '{rating_text[:100]}...'")
                    else:
                        print("full_review_json ratingText exists but is empty")
                else:
                    print("full_review_json does not have ratingText field")
                    
                # Check rating value
                if 'ratingValue' in sample_item['full_review_json']:
                    print(f"full_review_json ratingValue: {sample_item['full_review_json']['ratingValue']}")
                else:
                    print("full_review_json does not have ratingValue field")
                
                if 'text' in sample_item['full_review_json']:
                    print(f"full_review_json text: '{sample_item['full_review_json']['text'][:100]}...'")
            elif isinstance(sample_item['full_review_json'], str):
                try:
                    json_data = json.loads(sample_item['full_review_json'])
                    print(f"Parsed full_review_json keys: {list(json_data.keys())}")
                except:
                    print("Could not parse full_review_json as JSON string")
                    
        # Check for combined_text
        if 'combined_text' in sample_item:
            if sample_item['combined_text']:
                print(f"combined_text exists: '{sample_item['combined_text'][:100]}...'")
            else:
                print("combined_text exists but is empty")
    
    # Check if this is an analysis query to determine how to handle the results
    is_analysis_query = False
    if 'analysis' in query.lower() or 'analyze' in query.lower() or 'statistics' in query.lower():
        is_analysis_query = True
    
    # For analysis queries, we first compute statistics on ALL filtered items
    analysis_stats = {}
    if is_analysis_query and filtered_items:
        analysis_stats = compute_review_statistics(filtered_items)
        print(f"Computed statistics over all {len(filtered_items)} filtered reviews")
    
    # Phase 2: Apply semantic search on filtered results for relevant examples
    print(f"Phase 2: Ranking {len(filtered_items)} filtered documents by semantic relevance...")
    filtered_results = []
    
    try:
        # Create embeddings for the semantic query (optimized for search)
        query_embedding = embed_text(semantic_query)
        print(f"Using semantic query for embedding: {semantic_query}")
        
        # Make sure query embedding is properly shaped for FAISS
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D array with one row
        
        # For analysis queries, we need a larger sample size for better representation
        search_max_results = 30 if is_analysis_query else max_results
        
        if len(filtered_ids) > 0:
            # Option 1: If we're working with a subset, create a temporary index
            if len(filtered_ids) < len(metadata):
                # Extract vectors for filtered items
                filtered_vectors = np.vstack([index.reconstruct(idx) for idx in filtered_ids])
                
                # Create a temporary index for the filtered subset
                temp_index = faiss.IndexFlatL2(filtered_vectors.shape[1])
                temp_index.add(filtered_vectors)
                
                # Search the temporary index
                D, I = temp_index.search(query_embedding, min(search_max_results, len(filtered_items)))
                
                # Map temporary indices back to the filtered items
                for i, tmp_idx in enumerate(I[0]):
                    if D[0][i] != -1 and tmp_idx < len(filtered_items):
                        item = filtered_items[tmp_idx]
                        item["distance"] = float(D[0][i])
                        filtered_results.append(item)
            else:
                # Option 2: If all docs are filtered in, just search the main index
                D, I = index.search(query_embedding, min(search_max_results, len(filtered_items)))
                
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
            filtered_results = filtered_items[:search_max_results]
    except Exception as e:
        print(f"Error during semantic ranking: {e}")
        # Fallback: Use the first few filtered items without semantic ranking
        filtered_results = filtered_items[:max_results]
    
    print(f"Final result count after semantic ranking: {len(filtered_results)}")
    
    # Sort reviews based on content level and relevance
    def sort_key(review):
        # Get content level: 2 (full text), 1 (minimal content), 0 (no content)
        content_level = has_text_content(review)
        
        # Relevance score (negative because we want lower distance = higher priority)
        relevance = -1.0 * float(review.get('distance', float('inf')))
        
        # Return tuple for sorting (content_level first, then relevance)
        return (content_level, relevance)
    
    # Sort the filtered results by content level and relevance
    filtered_results.sort(key=sort_key, reverse=True)  # reverse=True to put highest scores first
    
    # Debug info about text content
    full_text = sum(1 for r in filtered_results if has_text_content(r) == 2)
    minimal = sum(1 for r in filtered_results if has_text_content(r) == 1)
    no_content = sum(1 for r in filtered_results if has_text_content(r) == 0)
    print(f"DEBUG - Reviews with text: {full_text}, with minimal content: {minimal}, with no content: {no_content}, total: {len(filtered_results)}")
    
    # For analysis queries, intelligently sample reviews for better representativeness
    if is_analysis_query:
        # For large sets, ensure good statistical representation
        if len(filtered_results) > 50:  # Arbitrary threshold for "large"
            print(f"Large analysis result set ({len(filtered_results)} reviews) - using map-reduce approach...")
            context = map_reduce_reviews(filtered_results, query, max_chunks=5)
        else:
            # For smaller sets, use intelligent sampling to ensure diversity
            print(f"Using intelligent sampling for analysis of {len(filtered_results)} reviews...")
            # Get a representative sample ensuring diversity in sentiment, rating, and presence of text
            sampled_reviews = sample_reviews_for_analysis(filtered_results, max_samples=max(20, max_results))
            context = prepare_context(sampled_reviews, max_reviews=max_results)
    else:
        # For non-analysis queries, use simpler handling
        if len(filtered_results) > 50:  # Arbitrary threshold for "large"
            print(f"Large result set ({len(filtered_results)} reviews) - using map-reduce approach...")
            context = map_reduce_reviews(filtered_results, query, max_chunks=5)
        else:
            # For small result sets, use the traditional approach
            print(f"Using traditional context preparation for {len(filtered_results)} reviews...")
            context = prepare_context(filtered_results, max_reviews=max_results, query=query)
        
        # Debug: Check what context is actually being sent to the LLM
        if DEBUG_REVIEWS:
            print(f"\n=== DEBUG: CONTEXT BEING SENT TO LLM ===\n{context}\n=== END DEBUG CONTEXT PREVIEW ===\n")
        
        # Special handling for empty context despite having filtered results
        if context.strip() == "No relevant reviews found." or context.strip() == "Reviews were found matching your date criteria, but none contained substantial review text or rating information.":
            context = f"Found {len(filtered_items)} reviews matching the date criteria from {date_start} to {date_end}, but they don't contain sufficient review text for analysis."
    
    # Handle case where we found reviews but need to provide clearer guidance to the LLM
    if len(filtered_results) > 0:
        # For analysis queries, include the full statistics in the context
        if is_analysis_query and analysis_stats:
            total_reviews = len(filtered_items)
            
            # Create stats prefix with all computed statistics
            stats_prefix = f"STATISTICAL ANALYSIS OF ALL {total_reviews} REVIEWS FROM {date_start or 'all time'} TO {date_end or 'present'}:\n\n"
            stats_prefix += format_review_statistics(analysis_stats)
            stats_prefix += f"\n\nBELOW ARE {len(filtered_results)} SAMPLE REVIEWS FOR QUALITATIVE ANALYSIS:\n\n"
            
            # Add stats to the beginning of the context
            context = stats_prefix + context
        
        # Add more explicit guidance for relative date queries to ensure LLM understands the context
        elif "last year" in query.lower():
            review_count = len(filtered_items)  # Use total filtered count, not just sampled
            context_prefix = f"I found {review_count} reviews from {date_start} to {date_end} (last year). The following contains analysis-worthy review content:\n\n"
            context = context_prefix + context
        elif date_start and date_end:  # For any date-filtered query, add clarity
            review_count = len(filtered_items)  # Use total filtered count, not just sampled
            context_prefix = f"I found {review_count} reviews from {date_start} to {date_end}. Here is the review content for analysis:\n\n"
            context = context_prefix + context
    else:
        # If we have no results after semantic ranking but had filtered items, provide that info
        if len(filtered_items) > 0:
            context = f"Found {len(filtered_items)} reviews matching your criteria, but none were semantically relevant to your query about '{core_query}'."
        else:
            context = "No documents found matching the specified filters."
    
    # Estimate token count
    token_count = estimate_token_count(query + context)
    print(f"Estimated token count for LLM request: {token_count}")
    
    # Generate the final response
    return refine_with_llm(query, context)

def compute_review_statistics(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics over a set of reviews
    
    Args:
        reviews: List of review items to analyze
        
    Returns:
        Dictionary containing various statistics about the reviews
    """
    try:
        total_reviews = len(reviews)
        if total_reviews == 0:
            return {"total_count": 0}
            
        # Initialize counters
        ratings = []
        rating_counts = {}
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
        dates = []
        has_text_count = 0
        
        # Collect data from all reviews
        for review in reviews:
            # Extract rating
            rating = None
            if 'full_review_json' in review:
                if isinstance(review['full_review_json'], dict) and 'ratingValue' in review['full_review_json']:
                    rating = review['full_review_json']['ratingValue']
                elif isinstance(review['full_review_json'], str):
                    try:
                        json_data = json.loads(review['full_review_json'])
                        if 'ratingValue' in json_data:
                            rating = json_data['ratingValue']
                    except:
                        pass
            elif 'original_row' in review and 'rating' in review['original_row']:
                rating = review['original_row']['rating']
            elif 'rating' in review:
                rating = review['rating']
                
            # Process ratings
            if rating is not None:
                try:
                    rating = float(rating)
                    ratings.append(rating)
                    rating_key = str(rating)
                    if rating_key in rating_counts:
                        rating_counts[rating_key] += 1
                    else:
                        rating_counts[rating_key] = 1
                except:
                    pass
                    
            # Process sentiment
            sentiment = "unknown"
            if 'sentimentAnalysis' in review:
                sentiment = review['sentimentAnalysis'].lower() if review['sentimentAnalysis'] and isinstance(review['sentimentAnalysis'], str) else "unknown"
            elif 'full_review_json' in review and isinstance(review['full_review_json'], dict) and 'sentimentAnalysis' in review['full_review_json']:
                sentiment_value = review['full_review_json']['sentimentAnalysis']
                sentiment = sentiment_value.lower() if sentiment_value and isinstance(sentiment_value, str) else "unknown"
                
            # Normalize sentiment value
            if sentiment in ["positive", "neutral", "negative"]:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts["unknown"] += 1
            
            # Check for text content using our text content detection function
            content_level = has_text_content(review)
            if content_level == 2:  # Has full text content
                has_text_count += 1
                
            # Extract date
            if '_parsed_date' in review:
                dates.append(review['_parsed_date'])
        
        # Calculate statistics
        stats = {
            "total_count": total_reviews,
            "rating_counts": rating_counts,
            "sentiment_counts": sentiment_counts,
            "has_text_count": has_text_count,
            "no_text_count": total_reviews - has_text_count,
        }
        
        # Rating statistics
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            stats["average_rating"] = round(avg_rating, 2)
            stats["rating_count"] = len(ratings)
            stats["min_rating"] = min(ratings)
            stats["max_rating"] = max(ratings)
        else:
            stats["rating_count"] = 0
        
        # Date statistics
        if dates:
            stats["oldest_date"] = min(dates)
            stats["newest_date"] = max(dates)
            stats["date_count"] = len(dates)
        else:
            stats["date_count"] = 0
        
        # Calculate rating distribution percentages
        if ratings:
            stats["rating_distribution"] = {}
            for rating_key, count in rating_counts.items():
                percentage = (count / len(ratings)) * 100
                stats["rating_distribution"][rating_key] = round(percentage, 1)
        
        # Calculate sentiment distribution percentages
        total_with_sentiment = sum(sentiment_counts.values())
        if total_with_sentiment > 0:
            stats["sentiment_distribution"] = {}
            for sentiment, count in sentiment_counts.items():
                if count > 0:  # Only include non-zero sentiments
                    percentage = (count / total_with_sentiment) * 100
                    stats["sentiment_distribution"][sentiment] = round(percentage, 1)
        
        return stats
    except Exception as e:
        print(f"Error computing review statistics: {e}")
        return {"total_count": len(reviews), "error": str(e)}

def sample_reviews_for_analysis(reviews: List[Dict[str, Any]], max_samples: int = 20, query: str = None) -> List[Dict[str, Any]]:
    """
    Select a representative sample of reviews for qualitative analysis.
    Ensures diversity across ratings, sentiments, dates, and text content while
    being context-aware to the specific query type (complaints, sentiment, etc.).
    
    Args:
        reviews: List of all reviews matching filters
        max_samples: Maximum number of reviews to include in the sample
        query: The original user query to customize sampling strategy
        
    Returns:
        A representative subset of reviews for qualitative analysis
    """
    if len(reviews) <= max_samples:
        return reviews
    
    # Group reviews by categories for balanced sampling
    reviews_by_rating = {}
    reviews_by_sentiment = {"positive": [], "neutral": [], "negative": [], "unknown": []}
    reviews_by_text = {2: [], 1: [], 0: []}  # Full text, minimal, none
    reviews_by_year = {}
    reviews_by_quarter = {}  # For temporal distribution
    
    # Analyze query to determine sampling strategy
    query_focus = "balanced"
    if query:
        query_lower = query.lower()
        if "complaint" in query_lower or "negative" in query_lower:
            query_focus = "negative"
        elif "positive" in query_lower or "good" in query_lower or "great" in query_lower:
            query_focus = "positive"
        elif "newest" in query_lower or "latest" in query_lower or "recent" in query_lower:
            query_focus = "recent"
        elif "oldest" in query_lower:
            query_focus = "oldest"
    
    print(f"DEBUG: Sampling strategy focus: {query_focus}")
    
    # First classify all reviews
    for review in reviews:
        # Get text content level
        content_level = has_text_content(review)
        reviews_by_text.setdefault(content_level, []).append(review)
        
        # Group by rating
        rating = None
        if 'rating' in review:
            rating = review['rating']
        elif 'ratingValue' in review:
            rating = review['ratingValue']
        elif 'full_review_json' in review and isinstance(review['full_review_json'], dict):
            if 'ratingValue' in review['full_review_json']:
                rating = review['full_review_json']['ratingValue']
        
        if rating is not None:
            try:
                rating_key = str(float(rating))
                reviews_by_rating.setdefault(rating_key, []).append(review)
            except:
                pass
        
        # Group by sentiment
        sentiment = "unknown"
        if 'sentimentAnalysis' in review:
            sentiment = review['sentimentAnalysis'].lower() if review['sentimentAnalysis'] and isinstance(review['sentimentAnalysis'], str) else "unknown"
        elif 'full_review_json' in review and isinstance(review['full_review_json'], dict):
            if 'sentimentAnalysis' in review['full_review_json']:
                sentiment_value = review['full_review_json']['sentimentAnalysis']
                sentiment = sentiment_value.lower() if sentiment_value and isinstance(sentiment_value, str) else "unknown"
        
        if sentiment in ["positive", "neutral", "negative"]:
            reviews_by_sentiment[sentiment].append(review)
        else:
            reviews_by_sentiment["unknown"].append(review)
            
        # Group by time periods
        review_date = None
        try:
            # Extract date from review
            if 'full_review_json' in review and review['full_review_json']:
                if isinstance(review['full_review_json'], dict) and 'date' in review['full_review_json']:
                    date_str = review['full_review_json']['date']
                    review_date = date_parser.parse(date_str) if date_str else None
                elif isinstance(review['full_review_json'], str):
                    try:
                        json_data = json.loads(review['full_review_json'])
                        if 'date' in json_data:
                            review_date = date_parser.parse(json_data['date'])
                    except:
                        pass
            
            if review_date:
                year = review_date.year
                quarter = f"{year}-Q{(review_date.month-1)//3+1}"
                reviews_by_year.setdefault(year, []).append(review)
                reviews_by_quarter.setdefault(quarter, []).append(review)
        except Exception as e:
            # Skip date parsing errors
            pass
    
    # Start with reviews that have full text content
    sampled_reviews = []
    
    # Adjust sampling strategy based on query focus
    if query_focus == "negative":
        sentiments = ["negative", "neutral", "positive"]
        target_per_sentiment = max(1, max_samples // 2)  # 50% negative
    elif query_focus == "positive":
        sentiments = ["positive", "neutral", "negative"]
        target_per_sentiment = max(1, max_samples // 2)  # 50% positive
    else:
        sentiments = ["negative", "positive", "neutral"]
        target_per_sentiment = max(1, max_samples // 3)  # Balanced
    
    print(f"DEBUG: Sampling strategy: ~{target_per_sentiment} reviews per sentiment category, focus={query_focus}")
    
    # Set up temporal strategy if needed
    temporal_strategy = None
    if query_focus in ["recent", "oldest"]:
        # Get all years with reviews
        years = sorted(list(reviews_by_year.keys()))
        if years:
            if query_focus == "recent":
                temporal_strategy = "newest"
                # Focus on newest 2 years if available
                target_years = years[-2:] if len(years) >= 2 else years
            else:  # oldest
                temporal_strategy = "oldest"
                # Focus on oldest 2 years if available
                target_years = years[:2] if len(years) >= 2 else years
                
            print(f"DEBUG: Temporal focus on {target_years} years")
    
    # First prioritize reviews with text content for each sentiment
    for sentiment in sentiments:
        candidates = [r for r in reviews_by_sentiment[sentiment] if has_text_content(r) == 2]
        
        # Apply temporal filter if needed
        if temporal_strategy and candidates:
            filtered_candidates = []
            for review in candidates:
                # Extract year from review date
                review_date = None
                try:
                    if 'full_review_json' in review and review['full_review_json']:
                        if isinstance(review['full_review_json'], dict) and 'date' in review['full_review_json']:
                            date_str = review['full_review_json']['date']
                            if date_str:
                                review_date = date_parser.parse(date_str)
                except:
                    pass
                
                # Only include reviews from target years
                if review_date and review_date.year in target_years:
                    filtered_candidates.append(review)
            
            # Use temporally filtered candidates if any exist
            if filtered_candidates:
                candidates = filtered_candidates
        
        if candidates:
            # Take a sample ensuring diversity in ratings if possible
            if len(candidates) > target_per_sentiment:
                # Sort by rating and then take samples at regular intervals
                candidates.sort(key=lambda x: float(x.get('rating', 0)) if x.get('rating') is not None else 0)
                step = max(1, len(candidates) // target_per_sentiment)
                for i in range(0, min(len(candidates), target_per_sentiment * step), step):
                    if len(sampled_reviews) < max_samples:
                        sampled_reviews.append(candidates[i])
            else:
                # Take all if fewer than target
                sampled_reviews.extend(candidates[:target_per_sentiment])
    
    # If we haven't filled our quota, try to add temporal diversity
    if len(sampled_reviews) < max_samples and len(reviews_by_quarter) > 1:
        # Get quarters not represented in our sample
        sampled_quarters = set()
        for review in sampled_reviews:
            try:
                if 'full_review_json' in review and review['full_review_json']:
                    if isinstance(review['full_review_json'], dict) and 'date' in review['full_review_json']:
                        date_str = review['full_review_json']['date']
                        if date_str:
                            review_date = date_parser.parse(date_str)
                            quarter = f"{review_date.year}-Q{(review_date.month-1)//3+1}"
                            sampled_quarters.add(quarter)
            except:
                pass
        
        # Try to include reviews from unrepresented quarters
        remaining = max_samples - len(sampled_reviews)
        missing_quarters = [q for q in sorted(reviews_by_quarter.keys()) if q not in sampled_quarters]
        
        # Prioritize missing quarters with text content
        for quarter in missing_quarters:
            if remaining <= 0:
                break
                
            # Find reviews from this quarter with text
            text_reviews = [r for r in reviews_by_quarter[quarter] if has_text_content(r) == 2]
            if text_reviews:
                # Add one review from this quarter
                sampled_reviews.append(text_reviews[0])
                remaining -= 1
    
    # If we haven't filled our quota, add reviews with minimal content
    if len(sampled_reviews) < max_samples and reviews_by_text.get(1):
        remaining = max_samples - len(sampled_reviews)
        sampled_reviews.extend(reviews_by_text[1][:remaining])
    
    # If still not enough, add reviews with no text content
    if len(sampled_reviews) < max_samples and reviews_by_text.get(0):
        remaining = max_samples - len(sampled_reviews)
        sampled_reviews.extend(reviews_by_text[0][:remaining])
    
    print(f"DEBUG: Sampled {len(sampled_reviews)} reviews for analysis from {len(reviews)} total")
    return sampled_reviews


def format_review_statistics(stats: Dict[str, Any]) -> str:
    """
    Format review statistics into a readable string for LLM context
    
    Args:
        stats: Dictionary of statistics from compute_review_statistics
        
    Returns:
        Formatted string containing statistical analysis
    """
    lines = []
    
    # Total count
    lines.append(f"Total Reviews: {stats['total_count']}")
    
    # Date range if available
    if stats.get('date_count', 0) > 0:
        oldest = stats['oldest_date'].strftime('%Y-%m-%d') if stats.get('oldest_date') else "unknown"
        newest = stats['newest_date'].strftime('%Y-%m-%d') if stats.get('newest_date') else "unknown"
        lines.append(f"Date Range: {oldest} to {newest}")
    
    # Rating statistics
    if stats.get('rating_count', 0) > 0:
        lines.append(f"\nRating Statistics:")
        lines.append(f"- Average Rating: {stats.get('average_rating', 'N/A')}")
        lines.append(f"- Rating Range: {stats.get('min_rating', 'N/A')} to {stats.get('max_rating', 'N/A')}")
        
        # Rating counts and distribution
        if 'rating_counts' in stats and stats['rating_counts']:
            lines.append("\nRating Distribution:")
            sorted_ratings = sorted([(float(k), v) for k, v in stats['rating_counts'].items()])
            for rating, count in sorted_ratings:
                percentage = stats.get('rating_distribution', {}).get(str(rating), 0)
                lines.append(f"- {rating}: {count} reviews ({percentage}%)")
    else:
        lines.append("No rating information available")
    
    # Sentiment statistics
    if 'sentiment_counts' in stats:
        # Check if we have any sentiment data other than unknown
        has_sentiment = False
        for sentiment, count in stats['sentiment_counts'].items():
            if sentiment != 'unknown' and count > 0:
                has_sentiment = True
                break
                
        if has_sentiment:
            lines.append("\nSentiment Analysis:")
            if 'sentiment_distribution' in stats:
                for sentiment, percentage in stats['sentiment_distribution'].items():
                    if sentiment != 'unknown':
                        count = stats['sentiment_counts'][sentiment]
                        lines.append(f"- {sentiment.capitalize()}: {count} reviews ({percentage}%)")
        else:
            lines.append("\nNo sentiment data available")
    
    # Text content statistics
    if 'has_text_count' in stats:
        lines.append(f"\nText Content:")
        has_text = stats['has_text_count']
        no_text = stats['no_text_count']
        total = has_text + no_text
        has_text_pct = (has_text / total) * 100 if total > 0 else 0
        lines.append(f"- Reviews with text: {has_text} ({round(has_text_pct, 1)}%)")
        lines.append(f"- Reviews without text: {no_text} ({round(100 - has_text_pct, 1)}%)")
    
    return "\n".join(lines)

def filter_by_date(result: Dict[str, Any], date_start: Optional[str], date_end: Optional[str]) -> bool:
    """Filter a result by date range."""
    # If no date filters specified, all results pass the filter
    if date_start is None and date_end is None:
        return True
    
    # Debug info - only show if filters are actually applied
    logger.debug(f"Filter by date - Date range: {date_start} to {date_end}")
    
    # Initialize date_interpretation_info
    date_interpretation_info = {"interpretation_source": "direct"}
    
    # Check if either date_start or date_end might contain relative date references
    relative_start, relative_end = None, None
    
    # Try to interpret date_start as a relative reference
    if date_start and isinstance(date_start, str):
        if any(term in date_start.lower() for term in ["last", "previous", "past", "this", "current"]):
            relative_start, _ = interpret_relative_date(date_start)
            if relative_start:
                logger.debug(f"Interpreted '{date_start}' as {relative_start.date()} (relative date reference)")
                # Store the original input for clarity in error messages
                date_start_original = date_start
                date_interpretation_info["interpretation_source"] = "relative_start_date"
    
    # Try to interpret date_end as a relative reference
    if date_end and isinstance(date_end, str):
        if any(term in date_end.lower() for term in ["last", "previous", "past", "this", "current"]):
            _, relative_end = interpret_relative_date(date_end)
            if relative_end:
                logger.debug(f"Interpreted '{date_end}' as {relative_end.date()} (relative date reference)")
                # Store the original input for clarity in error messages
                date_end_original = date_end
                date_interpretation_info["interpretation_source"] = "relative_end_date"
    
    # Check if this is possibly a relative date range like "last year"
    if date_start and isinstance(date_start, str) and date_end is None:
        rel_start, rel_end = interpret_relative_date(date_start) # Use default current time
        if rel_start and rel_end:
            relative_start, relative_end = rel_start, rel_end
            logger.debug(f"Interpreted '{date_start}' as date range: {rel_start.date()} to {rel_end.date()} (relative date range)")
            # Store the input information for error clarity
            date_range_original = date_start
            date_interpretation_info["interpretation_source"] = "relative_date_range"
    
    # Extract date from the result
    result_date = None
    
    # Try to extract from 'date' field
    if "date" in result:
        try:
            result_date = date_parser.parse(result["date"])
            logger.debug(f"Found date in 'date' field: {result_date}")
        except Exception as e:
            logger.debug(f"Error parsing date from 'date' field: {e}")
    
    # If not found, try to extract from 'full_review_json'
    if result_date is None and "full_review_json" in result:
        try:
            # Handle both dict and string JSON formats
            if isinstance(result["full_review_json"], dict):
                if "date" in result["full_review_json"]:
                    result_date = date_parser.parse(result["full_review_json"]["date"])
                    logger.debug(f"Found date in full_review_json dict: {result_date}")
            else:  # Assume it's a JSON string
                try:
                    json_data = json.loads(result["full_review_json"]) \
                        if isinstance(result["full_review_json"], str) \
                        else result["full_review_json"]
                    if "date" in json_data:
                        result_date = date_parser.parse(json_data["date"])
                        logger.debug(f"Found date in full_review_json JSON: {result_date}")
                except:
                    pass  # Fail silently if JSON parsing fails
        except Exception as e:
            logger.debug(f"Error parsing date from full_review_json: {e}")
    
    # Try alternative field names
    if result_date is None and "reviewDate" in result:
        try:
            result_date = date_parser.parse(result["reviewDate"])
            logger.debug(f"Found date in 'reviewDate' field: {result_date}")
        except Exception as e:
            logger.debug(f"Error parsing date from 'reviewDate' field: {e}")
    
    # If we couldn't find a date, the result doesn't match the filter
    if result_date is None:
        logger.debug("No date found in result, filtering out")
        return False
    # Apply date filters - use interpreted relative dates if available
    try:
        # Check for special keywords
        if date_start and isinstance(date_start, str) and date_start.lower() == 'oldest':
            print("  DEBUG - 'oldest' keyword detected - will keep all results (handled later)")
            return True  # We'll sort by date ascending later
        
        if date_end and isinstance(date_end, str) and date_end.lower() == 'oldest':
            print("  DEBUG - 'oldest' keyword detected - will keep all results (handled later)")
            return True  # We'll sort by date ascending later
            
        # Record the final date range we'll use for filtering
        date_interpretation_info["final_date_range"] = {
            "start": relative_start.strftime('%Y-%m-%d') if relative_start else date_start,
            "end": relative_end.strftime('%Y-%m-%d') if relative_end else date_end
        }
        
        # Use relative dates if available, otherwise parse the provided strings
        start_date = relative_start if relative_start else \
            (date_parser.parse(date_start) if date_start and date_start.lower() != 'oldest' else None)
        end_date = relative_end if relative_end else \
            (date_parser.parse(date_end) if date_end and date_end.lower() != 'oldest' else None)
            
        # Make all dates timezone-aware for proper comparison
        # Convert result_date to date-only to avoid timezone comparison issues
        result_date_date = result_date.date()  # Extract date part only, removing timezone
        date_interpretation_info["result_date"] = result_date_date.isoformat() if result_date_date else None
        
        # Create a strict filter that stores information about the date query
        # This will be used by handle_retrieval_query to detect and handle cases where
        # too few reviews match the requested date range
        
        # Apply start date filter if specified
        if start_date:
            start_date_date = start_date.date()  # Extract date part only
            #print(f"  DEBUG - Start date filter: {start_date_date} for query interpretation {date_interpretation_info['interpretation_source']}")
            if result_date_date < start_date_date:
                #print(f"  DEBUG - Review date {result_date_date} is before start date {start_date_date}, filtering out")
                # Add information to the global counter for matched reviews by date range
                return False
        
        # Apply end date filter if specified
        if end_date:
            end_date_date = end_date.date()  # Extract date part only
            #print(f"  DEBUG - End date filter: {end_date_date} for query interpretation {date_interpretation_info['interpretation_source']}")
            if result_date_date > end_date_date:
                #print(f"  DEBUG - Review date {result_date_date} is after end date {end_date_date}, filtering out")
                return False
        
        # If we got here, the result is within the date range
        #print(f"  DEBUG - Review date {result_date_date} is within range, keeping")
        # Increment the successful date matches counter
        return True
    
    except Exception as e:
        print(f"  DEBUG - Error in date filtering: {e}")
        # If there's an error in date comparison, better to include than exclude
        return True

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

# Constants for sentiment analysis
VALID_SENTIMENTS = ["positive", "negative", "neutral"]

def extract_numeric_rating(text: str) -> Optional[float]:
    """
    Extract a numeric rating value from text.
    Looks for patterns like "5-star", "4 stars", "rating of 3", etc.
    
    Args:
        text: The text to search for rating mentions
        
    Returns:
        Float rating value if found, None otherwise
    """
    if not text:
        return None
        
    # Common patterns for rating mentions
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:star|stars)',  # "5 stars", "4.5 star"
        r'(\d+(?:\.\d+)?)\-star',            # "5-star"
        r'rating\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)',  # "rating of 4", "rating: 3"
        r'rated\s*(?:as)?\s*(\d+(?:\.\d+)?)',      # "rated as 5", "rated 3"
        r'(\d+(?:\.\d+)?)\s*out\s*of\s*5',   # "4 out of 5"
        r'score\s*(?:of|:)?\s*(\d+(?:\.\d+)?)' # "score of 2", "score: 1"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                rating = float(match.group(1))
                # Validate the rating is in a reasonable range
                if 0 <= rating <= 5:
                    return rating
            except (ValueError, IndexError):
                continue
    
    return None

def rating_to_sentiment(rating: float) -> str:
    """
    Map a numeric rating to a sentiment category.
    
    Args:
        rating: Numeric rating (typically 1-5)
        
    Returns:
        String sentiment category: 'negative' (1-2), 'neutral' (3), or 'positive' (4-5)
    """
    try:
        rating_float = float(rating)
        if rating_float <= 2.0:
            return "negative"
        elif rating_float == 3.0:
            return "neutral"
        else:  # 4.0 or 5.0
            return "positive"
    except (ValueError, TypeError):
        return None  # Unable to convert to float or None value

def find_sentiment_tokens(text):
    """
    Extract sentiment tokens from text with typo tolerance using difflib.
    
    Args:
        text: The text to search for sentiment mentions
        
    Returns:
        List of normalized sentiment values with duplicates removed
    """
    tokens = text.lower().split()
    sentiments = []
    
    # Check each word against valid sentiments with fuzzy matching
    for word in tokens:
        matches = difflib.get_close_matches(word, VALID_SENTIMENTS, cutoff=0.75)
        if matches:
            sentiments.append(matches[0])  # Use the best match
            
    # Remove duplicates while preserving order
    return list(dict.fromkeys(sentiments))

def filter_by_sentiment(result, sentiment_filter: str) -> bool:
    """Filter a result based on sentiment."""
    if not sentiment_filter:
        return True
        
    # Try to get sentiment from the result
    sentiment = None
    
    # First check if sentiment is already in the document
    if "sentiment" in result:
        sentiment = result["sentiment"]
    elif "original_row" in result and "sentiment" in result["original_row"]:
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
    
    # Infer sentiment from rating using our helper function
    try:
        rating = float(rating_str)
        inferred_sentiment = rating_to_sentiment(rating)
        if not inferred_sentiment:
            return False
            
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

def prepare_context(results: List[Dict[str, Any]], max_reviews: int = 10, query: str = None) -> str:
    """Prepare context for the LLM from metadata results with improved handling of textless reviews.
    
    Args:
        results: List of review data
        max_reviews: Maximum number of reviews to include in context
        query: Original user query to tailor sampling strategy
    """
    # Get total counts BEFORE limiting results
    total_review_count = len(results)
    total_with_text = sum(1 for r in results if has_text_content(r) == 2)
    
    # Limit to the specified number of reviews for display
    limited_results = results[:max_reviews] if max_reviews else results

    context_lines = []
    context_parts = []

    # General statistics for the limited set
    review_count = len(limited_results)
    # Initialize counter for reviews with text in the limited set
    reviews_with_text = 0

    # If no results, return a message
    if total_review_count == 0:
        return "No matching reviews found."

    # Always add a header with TOTAL stats first, then clarify how many we're showing
    context_lines.append(f"Found {total_review_count} total matching reviews. Showing details for {review_count}.")
    
    # Additional note if there's a limit applied
    if max_reviews is not None and max_reviews > 0 and total_review_count > max_reviews:
        logger.info(f"Limiting context to {max_reviews} reviews out of {total_review_count} total")
        
    # results already limited above - now use the limited_results variable
    results = limited_results
    
    for i, result in enumerate(results, 1):
        context_part = f"---\nREVIEW #{i}:\n"
        review_text_found = False  # Flag to track if we found review text
        review_text = None
        rating = 'N/A'
        reviewer = 'Unknown'
        reply = None
        date = None
        location_id = None
        
        # Ensure we have at least a location ID if available
        if 'locationId' in result:
            location_id = result.get('locationId')
        elif 'full_review_json' in result and isinstance(result['full_review_json'], dict):
            if 'locationId' in result['full_review_json']:
                location_id = result['full_review_json'].get('locationId')
        
        # Check for combined_text (pre-formatted text)
        if "combined_text" in result and result["combined_text"]:
            # Don't use combined_text directly anymore - we'll format everything consistently
            combined_text = result["combined_text"]
            # Extract reviewer from combined text if not found elsewhere
            if reviewer == 'Unknown' and 'Reviewer:' in combined_text:
                reviewer_line = [line for line in combined_text.split('\n') if 'Reviewer:' in line]
                if reviewer_line:
                    reviewer = reviewer_line[0].replace('Reviewer:', '').strip()
        
        # Direct access to full_review_json as it contains the most reliable data
        if "full_review_json" in result:
            json_data = {}
            if isinstance(result["full_review_json"], dict):
                json_data = result["full_review_json"]
            else:  # Try to parse as JSON string
                try:
                    if isinstance(result["full_review_json"], str):
                        json_data = json.loads(result["full_review_json"])
                except Exception as e:
                    print(f"Error parsing full_review_json: {e}")
            
            # Extract key fields from full_review_json
            if json_data:
                # Get reviewer title
                if 'reviewerTitle' in json_data and json_data['reviewerTitle']:
                    reviewer = json_data['reviewerTitle']
                
                # Get rating value
                if 'ratingValue' in json_data and json_data['ratingValue'] is not None:
                    rating = json_data['ratingValue']
                
                # Get review text
                if 'ratingText' in json_data and json_data['ratingText'] and str(json_data['ratingText']).lower() not in ['nan', 'none', 'null']:
                    review_text = json_data['ratingText']
                    review_text_found = True
                
                # Get reply
                if 'reviewReply' in json_data and json_data['reviewReply'] and str(json_data['reviewReply']).lower() not in ['nan', 'none', 'null']:
                    reply = json_data['reviewReply']
                
                # Get date
                if 'date' in json_data and json_data['date']:
                    date = json_data['date']
        
        # If full_review_json didn't contain what we need, try original_row
        if "original_row" in result:
            orig = result["original_row"]
            
            if reviewer == 'Unknown' and 'reviewerTitle' in orig:
                reviewer = orig['reviewerTitle']
            
            if rating == 'N/A' and 'ratingValue' in orig and orig['ratingValue'] is not None:
                rating = orig['ratingValue']
                
            if not review_text_found:
                for field in ['ratingText', 'text', 'reviewText', 'review']:
                    if field in orig and orig[field] and str(orig[field]).lower() not in ['nan', 'none', 'null']:
                        review_text = orig[field]
                        review_text_found = True
                        break
            
            if not reply and 'reviewReply' in orig and orig['reviewReply'] and str(orig['reviewReply']).lower() not in ['nan', 'none', 'null']:
                reply = orig['reviewReply']
                
            if not date and 'date' in orig:
                date = orig['date']
        
        # As a last resort, look directly in the result dictionary
        if reviewer == 'Unknown' and 'reviewer' in result:
            reviewer = result['reviewer']
            
        if rating == 'N/A' and 'rating' in result and result['rating'] is not None:
            rating = result['rating']
            
        if not review_text_found and 'text' in result and result['text'] and str(result['text']).lower() not in ['nan', 'none', 'null']:
            review_text = result['text']
            review_text_found = True
            
        # Track reviews with actual text content
        if review_text_found:
            reviews_with_text += 1
            
        if not reply and 'reply' in result and result['reply'] and str(result['reply']).lower() not in ['nan', 'none', 'null']:
            reply = result['reply']
            
        if not date and 'date' in result:
            date = result['date']
        
        # FALLBACK: if nothing else gave us a date, use the parsed one
        if date is None and "_parsed_date" in result:
            date = result["_parsed_date"].date().isoformat()
            
        # Now build the context part with all the information we've collected
        if reviewer != 'Unknown':
            context_part += f"Reviewer: {reviewer}\n"
        
        if rating != 'N/A':
            context_part += f"Rating: {rating}\n"
            
        location_info = ""
        if location_id:
            location_info = f"Location ID: {location_id}\n"
            
        # Handle review text more robustly
        if review_text and str(review_text).lower() not in ['nan', 'none', 'null']:
            # Truncate very long reviews to a reasonable length
            if len(str(review_text)) > 500:
                review_text = str(review_text)[:497] + "..."
            context_part += f"Review: {review_text}\n"
            review_text_found = True
        else:
            # If this is a simple query about "give me reviews from 2024"
            # we should make it clear there is a review even without text
            context_part += "Review: [No review text available]\n"
                
        if reply and str(reply).lower() not in ['nan', 'none', 'null']:
            if len(str(reply)) > 300:
                reply = str(reply)[:297] + "..."
            context_part += f"Reply: {reply}\n"
            
        if date:
            context_part += f"Date: {date}\n"
            
        if location_id:
            context_part += f"{location_info}"
        
        # Add match score
        context_part += f"Relevance Score: {1.0 / (1.0 + result.get('distance', 0)):.4f}\n"
        
        # Always include the review - even without text content - for simple queries like "give me reviews from 2024"
        # For complex/specific queries, we might want to be more selective, but basic queries should show all matches
        context_parts.append(context_part)
    
    # Debug information about reviews with text
    print(f"Debug: Including {reviews_with_text} reviews with actual text content out of {review_count} total reviews")
    
    # Even if no reviews have text, if we found some reviews, include them
    if not context_parts:
        return "Reviews were found matching your criteria, but none contained valid review information."
    
    # Add a header indicating BOTH the total count and limited count to help the LLM understand
    header = f"=== Found {total_review_count} TOTAL reviews matching criteria. Showing {review_count} reviews ({reviews_with_text} with text). {total_with_text} out of {total_review_count} total contain review text. ===\n\n"
    
    return header + "\n".join(context_parts)

def estimate_token_count(text: str) -> int:
    """Estimate the token count of a text. This is a rough estimate, not exact."""
    # A rough estimate: 1 token ~= 4 characters for English text
    return len(text) // 4

def refine_with_llm(query: str, context: str, model: str = LLM_MODEL) -> str:
    """Generate a response based on the query and context using the LLM."""
    # Extract review count from the context using a more robust approach
    # 1) Try to pull the real count from the header first (most accurate)
    m = re.search(r"=== Found (\d+) reviews", context)
    if m:
        review_count = int(m.group(1))
    else:
        # 2) Fallback: try MAP-REDUCE summaries
        if "SUMMARY OF REVIEWS CHUNK" in context:
            review_count = context.count("SUMMARY OF REVIEWS CHUNK")
        else:
            # 3) Final fallback: count raw reviews by "REVIEW #" markers
            review_count = context.count("REVIEW #")
    
    # Create a system prompt that generates conversational, natural-sounding responses
    system_prompt = """You are Sarah, a friendly and insightful customer experience specialist who analyzes customer reviews with a personal touch.
    
    When responding to queries about review data:
    - Write in a warm, conversational tone as if you're talking directly to a friend or colleague
    - Use natural language with occasional contractions (I've, we're, there's) and conversational phrases
    - Avoid corporate or robotic language - no "per your request" or "as per the data"
    - Start responses with friendly intros like "I looked into those reviews for you" or "Here's what I found about..."
    - Include thoughtful transitions between sections ("Interestingly," "What stood out to me was," "I noticed that...")
    - Express genuine enthusiasm where appropriate ("Wow, the feedback in May was really positive!")
    - When presenting examples, introduce them naturally: "Here's a telling review from March:"
    
    Always maintain accuracy:
    1. Base your analysis ONLY on the provided review data in the CONTEXT section
    2. Present comprehensive information with all relevant statistics and examples
    3. Don't skip or omit any important reviews or data points from your analysis
    4. Format information in an easy-to-read way with appropriate spacing and organization
    5. Never reference how you obtained the information (no mentions of databases, filters, etc.)
    
    Remember: Your goal is to sound like a knowledgeable human giving a thoughtful, complete analysis in a natural conversation.
    """
    
    # Determine if we're working with raw reviews or summarized chunks
    if "SUMMARY OF REVIEWS CHUNK" in context:
        # For summarized content (map-reduce approach)
        user_prompt = f"""QUERY: {query}

CONTEXT FROM ANALYSIS OF MULTIPLE REVIEW CHUNKS:
{context}

IMPORTANT: Synthesize the information from all review summaries to answer the query.
Based solely on the above context, please answer the query comprehensively."""
    else:
        # For raw review content (traditional approach)
        user_prompt = f"""QUERY: {query}

CONTEXT FROM RETRIEVED REVIEWS ({review_count} reviews total):
{context}

IMPORTANT: Include information from ALL {review_count} reviews in your answer. Do not skip any reviews.
Based solely on the above context, please answer the query comprehensively."""

    # Estimate token count to avoid hitting limits
    estimated_tokens = estimate_token_count(system_prompt) + estimate_token_count(user_prompt)
    
    print(f"Estimated token count for LLM request: {estimated_tokens}")
    
    # If the total tokens might exceed limits, switch to a more structured summary approach
    if estimated_tokens > 120000:  # Setting a safety margin below the model's max context
        print("WARNING: Context likely exceeds token limit. Shortening content...")
        if "SUMMARY OF REVIEWS CHUNK" in context:
            # We're already using summaries but still too long
            # Extract just the first and last paragraph from each chunk summary
            shortened_context = ""
            for chunk in context.split("SUMMARY OF REVIEWS CHUNK"):
                if not chunk.strip():
                    continue
                    
                chunk_parts = chunk.split("\n\n")
                if len(chunk_parts) > 2:
                    shortened_context += "SUMMARY OF REVIEWS CHUNK" + chunk_parts[0] + "\n\n" + chunk_parts[-1] + "\n\n"
                else:
                    shortened_context += "SUMMARY OF REVIEWS CHUNK" + chunk + "\n\n"
            
            context = shortened_context
            user_prompt = f"""QUERY: {query}

CONTEXT FROM ANALYSIS OF MULTIPLE REVIEW CHUNKS (CONDENSED TO AVOID TOKEN LIMITS):
{context}

IMPORTANT: The context has been condensed to fit within token limits. Synthesize the available information to answer the query.
Based solely on the above context, please answer the query comprehensively."""
        else:
            # If we haven't already chunked and summarized, do so now
            print("Context too large - attempting emergency summarization")
            emergency_summary = f"ERROR: The review set is too large to process directly. Please modify your query to be more specific or apply additional filters to reduce the number of reviews."
            return emergency_summary

    try:
        print(f"Sending query to {model} for final response generation...")
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
    index, metadata = load_faiss_index()
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
    logger.info("Processing query...")
    start_time = datetime.datetime.now()
    
    # Parse the query using our unified parser for consistent, reliable parsing
    parsed_query = parse_user_query(query)
    logger.info(f"[RETRIEVAL QUERY] Parsed query: {json.dumps(parsed_query, indent=2)}")
    
    # Get the primary query type and actions list
    query_type = parsed_query.get("query_type", "retrieval")
    actions = parsed_query.get("actions", [query_type]) 
    include_examples = parsed_query.get("include_examples", False)
    
    # Always force with_text and include_examples for all queries to maximize data
    parsed_query["with_text"] = True
    parsed_query["include_examples"] = True
    logger.info("Forced with_text=True and include_examples=True to maximize data for all queries")
    
    # ────────────── Inject count for analysis ──────────────
    if "analysis" in query.lower() or "analyze" in query.lower():
        # Make sure we do a count *and* a retrieval/analysis
        if "count" not in actions:
            actions.insert(0, "count")
        if "retrieval" not in actions:
            actions.append("retrieval")
        logger.info(f"Analysis query detected, expanded actions: {actions}")
    
    # Get limit/max_results if specified, otherwise use default values
    # If parsed limit is None, let handlers decide (analysis will use whole set;
    # retrieval/latest will fall back to their own defaults)
    max_results = parsed_query.get("limit")
    if max_results is None:
        max_results = 15 if "count" in actions else 10
    
    # Load FAISS index and metadata if not provided
    if index is None or metadata is None:
        index, metadata = load_faiss_index()
    
    # Store all responses for multiple intents
    responses = []
    
    # Process according to query type and actions
    if "count" in actions:
        # Handle multi-sentiment counting specially
        if len(parsed_query.get("sentiments", [])) > 1:
            logger.info(f"Detected multi-sentiment count query with sentiments: {parsed_query['sentiments']}")
            responses.append(handle_count_query(query, index, metadata, max_results))
        else:
            responses.append(handle_count_query(query, index, metadata, max_results))
    
    # Handle time-based aggregation queries
    if "time_distribution" in actions or "peak_time" in actions:
        logger.info(f"Detected time-based query with group_by={parsed_query.get('group_by')} and aggregate={parsed_query.get('aggregate')}")
        time_answer = handle_time_query(query, index, metadata, max_results)
        responses.append(time_answer)
    
    # Handle standard retrieval actions
    if "latest" in actions:
        latest_answer = handle_latest_query(query, index=index, metadata=metadata, max_results=max_results)
        responses.append(latest_answer)
    
    # Always run retrieval if specifically requested or if include_examples is True
    if "retrieval" in actions or include_examples:
        # Determine the appropriate limit based on the query intent
        if actions == ["retrieval"]:
            # Pure retrieval query: give exactly what the user asked for
            example_limit = max_results
            logger.info(f"Pure retrieval query detected - using user's full limit: {max_results}")
        else:
            # Count/analysis or multi-intent query: only show up to 5 examples
            example_limit = min(5, max_results)
            logger.info(f"Secondary retrieval or multi-intent query - limiting examples to: {example_limit}")
            
        retrieval_answer = handle_retrieval_query(query, index=index, metadata=metadata, max_results=example_limit)
        responses.append(retrieval_answer)
    
    # If no specific actions were recognized, default to retrieval
    if not responses:
        retrieval_answer = handle_retrieval_query(query, index=index, metadata=metadata, max_results=max_results)
        responses.append(retrieval_answer)
    
    # Combine multiple responses in the order the parser gave us
    if len(responses) == 1:
        answer = responses[0]
    else:
        # Define section titles for each action type
        section_titles = {
            "count": "Count Information",
            "latest": "Latest Reviews",
            "retrieval": "Example Reviews",
            "time_distribution": "Time Distribution Analysis",
            "peak_time": "Peak Time Analysis"
        }
        
        # Build response sections in the same order as actions
        chunks = []
        for action, resp in zip(actions, responses):
            title = section_titles.get(action, action.title())
            chunks.append(f"### {title}\n\n{resp}")
        
        # Join all sections together
        answer = "\n\n".join(chunks)
    
    # Display answer
    logger.info("\n" + "=" * 80)
    logger.info("ANSWER:")
    logger.info(answer)
    logger.info("=" * 80)
    
    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    logger.info(f"Query processed in {processing_time:.2f} seconds.")
    return answer

def chunk_results(results: List[Dict[str, Any]], chunk_size: int = 20) -> List[List[Dict[str, Any]]]:
    """
    Break up a large set of review results into smaller chunks for processing.
    
    Args:
        results: List of review metadata dictionaries
        chunk_size: Maximum number of reviews per chunk
        
    Returns:
        List of chunks, where each chunk is a list of review dictionaries
    """
    return [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]


def summarize_chunk(chunk: List[Dict[str, Any]], query: str) -> str:
    """
    Summarize a chunk of reviews to create a condensed representation.
    
    Args:
        chunk: A list of review dictionaries to summarize
        query: The original user query to focus the summarization
        
    Returns:
        A string containing the summarized content
    """
    # Format the chunk for summarization
    chunk_text = "\n\n".join([f"REVIEW {i+1}:\n" + 
                         f"Date: {item.get('original_row', {}).get('date', 'Unknown')}\n" +
                         f"Rating: {item.get('original_row', {}).get('ratingValue', 'Unknown')}\n" +
                         f"Text: {item.get('original_row', {}).get('ratingText', 'No text')}\n" +
                         f"Sentiment: {item.get('original_row', {}).get('sentimentAnalysis', 'Unknown')}"
                        for i, item in enumerate(chunk)])
    
    # Create a summarization prompt
    system_prompt = "You are an expert summarizer. Condense multiple reviews into a concise summary that captures key insights, patterns, and trends. Focus on information relevant to the query."
    
    user_prompt = f"""I need you to summarize the following set of {len(chunk)} reviews, focusing on information relevant to this query: '{query}'

{chunk_text}

Provide a concise summary that highlights patterns, common themes, and key insights about these reviews. Focus especially on aspects relevant to the query."""
    
    try:
        response = client.chat.completions.create(
            model=SUMMARIZATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual summaries
            max_tokens=500   # Keep summaries concise
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Fallback to a basic summary if API call fails
        return f"Summary of {len(chunk)} reviews (average rating: {sum(float(item.get('original_row', {}).get('ratingValue', 0)) for item in chunk) / len(chunk):.1f}/5)"


def map_reduce_reviews(results: List[Dict[str, Any]], query: str, max_chunks: int = 5) -> str:
    """
    Process large review sets using a map-reduce approach:
    1. Map: Break reviews into chunks and summarize each chunk
    2. Reduce: Combine chunk summaries into a final context
    
    Args:
        results: List of review dictionaries
        query: The original user query
        max_chunks: Maximum number of chunks to process
        
    Returns:
        A condensed context string to feed to the LLM
    """
    # If the result set is small enough, just use traditional context preparation
    if len(results) <= 20:  # Small enough to process directly
        return prepare_context(results, max_reviews=20)  # Limit to 20 reviews for consistency with chunking
    
    # For larger sets, use the chunking approach
    chunks = chunk_results(results, chunk_size=20)
    
    # Limit to max_chunks to avoid processing too many
    if len(chunks) > max_chunks:
        print(f"Limiting analysis to {max_chunks} chunks ({max_chunks * 20} reviews) out of {len(chunks)} chunks")
        chunks = chunks[:max_chunks]
    
    # Map: Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)} ({len(chunk)} reviews)...")
        summary = summarize_chunk(chunk, query)
        chunk_summaries.append(f"SUMMARY OF REVIEWS CHUNK {i+1}/{len(chunks)}:\n{summary}")
    
    # Reduce: Join the summaries with some metadata about the process
    total_reviews = sum(len(chunk) for chunk in chunks)
    full_context = f"ANALYSIS OF {total_reviews} REVIEWS (from {len(chunks)} chunks):\n\n"
    full_context += "\n\n".join(chunk_summaries)
    
    # Add a note about any reviews that were excluded
    if len(results) > total_reviews:
        excluded = len(results) - total_reviews
        full_context += f"\n\nNote: {excluded} additional reviews matched but were not included in the detailed analysis due to volume limitations."
    
    return full_context



if __name__ == "__main__":
    main()
