from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import List, Optional
import os
from datetime import datetime
import chardet

from ..models import ProcessedDocument, DocumentMetadata, SearchQuery, SearchResult
from ..dependencies import get_document_processor
from processing import DocumentProcessor

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=ProcessedDocument)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    tags: Optional[str] = None,
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    try:
        # Create temporary file path
        upload_dir = os.path.join("storage", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        # Read file content
        content = await file.read()
        
        # Detect the file encoding
        result = chardet.detect(content)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
        try:
            # Try to decode with detected encoding
            text_content = content.decode(encoding)
        except UnicodeDecodeError:
            # If that fails, try some common encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for enc in encodings:
                try:
                    text_content = content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode file content. Please ensure it's a valid text file."
                )
        
        # Prepare metadata
        metadata = DocumentMetadata(
            title=title or file.filename,
            date=datetime.utcnow(),
            tags=tags.split(",") if tags else None
        )
        
        # Process document
        doc_id = doc_processor.process_document(
            content=text_content,
            metadata=metadata.dict()
        )
        
        return ProcessedDocument(
            document_id=doc_id,
            chunk_count=len(text_content.split()) // 512 + 1,  # Rough estimate
            metadata=metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=List[SearchResult])
async def search_documents(
    query: SearchQuery,
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    try:
        results = doc_processor.search_similar(
            query=query.query,
            n_results=query.limit,
            metadata_filter=query.filters
        )
        
        return [
            SearchResult(
                content=result["content"],
                metadata=result["metadata"],
                score=1 - result["distance"]  # Convert distance to similarity score
            )
            for result in results
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    try:
        doc_processor.delete_document(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))