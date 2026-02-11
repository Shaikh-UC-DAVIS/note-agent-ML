import os
from typing import List, Dict, Optional
import tiktoken
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start_char_idx: int
    end_char_idx: int
    token_count: int
    metadata: Dict

class TextIngestion:
    def __init__(self, model_name: str = "cl100k_base"):
        """
        Initialize the TextIngestion pipeline.
        
        Args:
            model_name: The encoding model to use for token counting. 
                       Default is 'cl100k_base' (used by GPT-4).
        """
        self.tokenizer = tiktoken.get_encoding(model_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str, window_size: int = 800, overlap: int = 80) -> List[Chunk]:
        """
        Splits text into chunks with a specified window size and overlap.
        
        Args:
            text: The input text to chunk.
            window_size: The maximum number of tokens per chunk.
            overlap: The number of tokens to overlap between chunks.
            
        Returns:
            A list of Chunk objects.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []

        chunks = []
        step = window_size - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + window_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate character indices (approximate reconstruction for demo, accurate would require mapping)
            # For a real robust implementation, we'd need to map token indices back to char indices deeply.
            # Here we'll do a simple find, but beware of duplicates. 
            # A better approach for exact char/span mapping usually involves keeping offsets.
            # For this MVP, we will rely on the decoded text.
            
            chunk_obj = Chunk(
                text=chunk_text,
                start_char_idx=-1, # Todo: implement precise char mapping if needed
                end_char_idx=-1,   # Todo: implement precise char mapping if needed
                token_count=len(chunk_tokens),
                metadata={}
            )
            chunks.append(chunk_obj)
            
        return chunks

    def read_file(self, file_path: str) -> str:
        """Reads a file from the filesystem."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def process_file(self, file_path: str) -> List[Chunk]:
        """Reads a file and returns its chunks."""
        text = self.read_file(file_path)
        return self.chunk_text(text)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: The SentenceTransformer model to use.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Please install it with `pip install sentence-transformers`.")

    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generates embeddings for a list of chunks.
        
        Args:
            chunks: A list of Chunk objects.
            
        Returns:
            A list of embedding vectors (lists of floats).
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
