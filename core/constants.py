"""
Global constants for the loan-navigator-suite project.
"""
from pathlib import Path

# Base directory configuration
BASE_DIR = Path(__file__).parent.parent  # Go up one level to reach project root

# File paths and directories
DATASET_DIR = BASE_DIR / "dataset"
VECTOR_STORE_DIR = BASE_DIR / "database/vector_store"
PROMPTS_DIR = BASE_DIR / "prompts"

# Vector store settings
COLLECTION_NAME = "loan_documents"
COLLECTION_METADATA = {"use_type": "PRODUCTION"}

# Document processing settings
MIN_DOCUMENT_LENGTH = 100
CHUNK_SIZE = 3500
CHUNK_OVERLAP = 500

# Azure OpenAI settings
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_NAME = "gpt4o"  # Using GPT-4 for complex reasoning and calculations

# Retrieval settings
SEARCH_K = 2
TEST_QUERY = "What are the loan policies?"
