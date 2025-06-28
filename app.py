"""
Reviews Info - Flask Server for the RAG Review Analysis System
This serves the web UI and provides API endpoints to process queries using the RAG system.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import logging

# Import the RAG system functionality
from retriever_responder import parse_user_query, load_faiss_index, process_query


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded index and metadata
index = None
metadata = None

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', path)

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests and return RAG responses"""
    global index, metadata
    data = request.get_json()
    query = data.get('query', '')
    
    # Log the incoming query
    logger.info(f"Received query: {query}")
    
    if not query.strip():
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Process the query using the retriever_responder module
        response = process_query(query, index=index, metadata=metadata)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': f'Failed to process query: {str(e)}'}), 500

# Initialize the index and metadata at startup
def initialize_app():
    """Load the FAISS index and metadata"""
    global index, metadata
    try:
        logger.info("Loading FAISS index and metadata...")
        index, metadata = load_faiss_index()
        logger.info(f"Successfully loaded index with {index.ntotal} vectors")
        return True
    except Exception as e:
        logger.error(f"Failed to load index and metadata: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    # Determine port
    port = int(os.environ.get('PORT', 5001))
    
    # Check if the FAISS index and metadata files exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(current_dir, "embeddings")
    faiss_index_path = os.path.join(embeddings_dir, "reviews_faiss_index.pkl")
    metadata_path = os.path.join(embeddings_dir, "reviews_metadata.pkl")
    
    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        logger.error("FAISS index or metadata file not found. Please run embedding_generator.py first.")
        sys.exit(1)
    
    # Initialize the app before starting the server
    logger.info("Initializing RAG system...")
    if not initialize_app():
        logger.error("Failed to initialize the RAG system. Please check the logs for details.")
        sys.exit(1)
        
    # Start the Flask app
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
