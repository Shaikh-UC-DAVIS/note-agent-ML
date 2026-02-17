import os
from dotenv import load_dotenv

load_dotenv()

config = {
    'groq_api_key': os.getenv('GROQ_API_KEY'),
    'model': os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
    'temperature': float(os.getenv('GROQ_TEMPERATURE', 0.2)),
    'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', 2048)),
    'timeout': int(os.getenv('GROQ_TIMEOUT', 30))
}
