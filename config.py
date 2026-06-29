import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Confidence score thresholds — asymmetric to protect against false positives
HUMAN_THRESHOLD = 0.35   # below this → high-confidence human
AI_THRESHOLD = 0.70      # above this → high-confidence AI

# Signal weights (sum to 1.0)
LLM_WEIGHT = 0.60
STYLOMETRIC_WEIGHT = 0.40

# Rate limiting
RATE_LIMIT = "10 per hour"

# Content limits
MAX_CONTENT_CHARS = 10_000
MIN_CONTENT_WORDS = 10
