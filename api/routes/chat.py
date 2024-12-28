from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from typing import List, Optional
import asyncio

from ..models import ChatRequest, ChatResponse, Message
from ..dependencies import get_chat_engine, get_document_processor
from processing import ChatEngine, DocumentProcessor

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_engine: ChatEngine = Depends(get_chat_engine),
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    try:
        # Get the latest question
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        question = request.messages[-1].content
        
        # Search for relevant context
        context = doc_processor.search_similar(question)
        
        # Format previous messages for history
        history = []
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                history.append({
                    "user": request.messages[i].content,
                    "assistant": request.messages[i + 1].content
                })
        
        # Generate response
        response = await chat_engine.generate_response(
            question=question,
            context=context,
            chat_history=history,
            stream=False
        )
        
        return ChatResponse(
            response=response,
            context_used=len(context)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    chat_engine: ChatEngine = Depends(get_chat_engine),
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    async def event_generator():
        try:
            # Get context and prepare response
            question = request.messages[-1].content
            context = doc_processor.search_similar(question)
            
            async for chunk in chat_engine.generate_response(
                question=question,
                context=context,
                stream=True
            ):
                if chunk:
                    yield {
                        "event": "message",
                        "data": chunk
                    }
            
            yield {
                "event": "done",
                "data": ""
            }
        
        except Exception as e:
            yield {
                "event": "error",
                "data": str(e)
            }
    
    return EventSourceResponse(event_generator())