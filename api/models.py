from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    
class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of chat messages")
    stream: bool = Field(default=False, description="Whether to stream the response")
    
class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    context_used: int = Field(..., description="Number of context chunks used")
    
class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    date: Optional[datetime] = Field(None, description="Document date")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    
class ProcessedDocument(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    chunk_count: int = Field(..., description="Number of chunks created")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query")
    filters: Optional[Dict] = Field(None, description="Metadata filters")
    limit: int = Field(default=3, description="Number of results to return")
    
class SearchResult(BaseModel):
    content: str = Field(..., description="Content of the chunk")
    metadata: Dict = Field(..., description="Chunk metadata")
    score: float = Field(..., description="Similarity score")