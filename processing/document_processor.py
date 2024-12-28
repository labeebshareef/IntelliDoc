import os
from typing import List, Dict, Optional
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings

from .utils.text_splitter import TextSplitter
from .utils.embeddings import EmbeddingsManager

class DocumentProcessor:
    def __init__(
        self,
        storage_path: str = "./storage",
        collection_name: str = "documents"
    ):
        self.storage_path = storage_path
        self.chroma_path = os.path.join(storage_path, "chroma")
        os.makedirs(self.chroma_path, exist_ok=True)
        
        # Initialize components
        self.text_splitter = TextSplitter()
        self.embeddings = EmbeddingsManager()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def process_document(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Process a document and store its chunks in the vector store."""
        # Clean and split the text
        cleaned_text = self.text_splitter.clean_text(content)
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Generate embeddings
        embeddings = self.embeddings.get_embeddings(chunks)
        
        # Prepare metadata
        doc_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "chunk_count": len(chunks)
        }
        if metadata:
            doc_metadata.update(metadata)
        
        # Generate IDs for chunks
        doc_id = document_id or f"doc_{datetime.utcnow().timestamp()}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=chunk_ids,
            metadatas=[doc_metadata] * len(chunks)
        )
        
        return doc_id
    
    def search_similar(
        self,
        query: str,
        n_results: int = 3,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar chunks in the vector store."""
        # Generate query embedding
        query_embedding = self.embeddings.get_embedding(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=metadata_filter
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def delete_document(self, document_id: str) -> None:
        """Delete a document and its chunks from the vector store."""
        # Find all chunks for this document
        chunk_ids = self.collection.get(
            where={"document_id": document_id}
        )['ids']
        
        # Delete the chunks
        self.collection.delete(ids=chunk_ids)
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict]:
        """Get metadata for a specific document."""
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                limit=1
            )
            if result['metadatas']:
                return result['metadatas'][0]
            return None
        except:
            return None