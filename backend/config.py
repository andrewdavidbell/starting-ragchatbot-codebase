import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 10        # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


def validate_config():
    """
    Validate configuration settings and raise errors for invalid values.
    Call this on application startup to fail fast with clear error messages.
    """
    errors = []

    # Critical: MAX_RESULTS must be greater than 0
    if Config.MAX_RESULTS <= 0:
        errors.append(
            f"❌ MAX_RESULTS is {Config.MAX_RESULTS}. "
            "This will cause all searches to return 0 results! "
            "Set to at least 5 for proper functionality."
        )

    # Warn if MAX_RESULTS is too low
    if 0 < Config.MAX_RESULTS < 3:
        errors.append(
            f"⚠️  MAX_RESULTS={Config.MAX_RESULTS} is very low. "
            "Search quality may be poor. Recommended: 5-10"
        )

    # Check CHUNK_SIZE is reasonable
    if Config.CHUNK_SIZE <= 0:
        errors.append(f"❌ CHUNK_SIZE must be positive, got {Config.CHUNK_SIZE}")

    if Config.CHUNK_SIZE < 100:
        errors.append(
            f"⚠️  CHUNK_SIZE={Config.CHUNK_SIZE} is very small. "
            "Chunks may not contain enough context. Recommended: 500-1000"
        )

    # Check CHUNK_OVERLAP is valid
    if Config.CHUNK_OVERLAP < 0:
        errors.append(f"❌ CHUNK_OVERLAP must be non-negative, got {Config.CHUNK_OVERLAP}")

    if Config.CHUNK_OVERLAP >= Config.CHUNK_SIZE:
        errors.append(
            f"❌ CHUNK_OVERLAP ({Config.CHUNK_OVERLAP}) must be less than "
            f"CHUNK_SIZE ({Config.CHUNK_SIZE})"
        )

    # Check MAX_HISTORY is reasonable
    if Config.MAX_HISTORY <= 0:
        errors.append(f"❌ MAX_HISTORY must be positive, got {Config.MAX_HISTORY}")

    if Config.MAX_HISTORY < 3:
        errors.append(
            f"⚠️  MAX_HISTORY={Config.MAX_HISTORY} is very low. "
            "Conversation context will be limited. Recommended: 5-10"
        )

    # Check API key is set
    if not Config.ANTHROPIC_API_KEY:
        errors.append(
            "❌ ANTHROPIC_API_KEY is not set. "
            "Please add it to your .env file. "
            "Get your API key from https://console.anthropic.com/"
        )

    # If there are errors, raise an exception
    if errors:
        error_message = "\n\n⚠️  Configuration Validation Failed:\n" + "\n".join(f"  {e}" for e in errors)
        raise ValueError(error_message)

    print("✅ Configuration validation passed")


