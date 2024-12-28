from typing import List, Optional
import re

class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using specified separators."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try different separators to find the best split point
            split_point = end
            for separator in self.separators:
                last_separator = text.rfind(separator, start, end)
                if last_separator != -1:
                    split_point = last_separator + len(separator)
                    break
            
            # Add the chunk
            chunks.append(text[start:split_point])
            
            # Move start point for next chunk, considering overlap
            start = split_point - self.chunk_overlap
            
            # Ensure we're not stuck at the same position
            if start < split_point - self.chunk_size:
                start = split_point
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing newlines."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Strip whitespace
        text = text.strip()
        return text