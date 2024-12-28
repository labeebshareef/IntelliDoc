from typing import List, Dict, Optional, Union, AsyncGenerator
import json
import httpx
from datetime import datetime

class ChatEngine:
    def __init__(
        self,
        model_name: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        context_window: int = 4096
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.context_window = context_window
        self.system_prompt = """You are a helpful assistant answering questions based on the provided context.
Please be concise and accurate. If the context doesn't contain relevant information, say so."""
    
    async def generate_response(
        self,
        question: str,
        context: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        # Prepare context string
        context_str = "\n\n".join([
            f"[Content {i+1}]: {item['content']}"
            for i, item in enumerate(context)
        ])
        
        # Prepare chat history
        history_str = ""
        if chat_history:
            history_str = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in chat_history[-3:]  # Include last 3 exchanges
            ])
        
        # Construct the prompt
        prompt = f"{self.system_prompt}\n\nContext:\n{context_str}\n"
        if history_str:
            prompt += f"\nPrevious conversation:\n{history_str}\n"
        prompt += f"\nUser: {question}\nAssistant: "
        
        # Prepare the request
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": self.context_window
            }
        }
        
        async with httpx.AsyncClient() as client:
            if not stream:
                response = await client.post(url, json=payload)
                response_data = response.json()
                return response_data["response"]
            else:
                async def stream_response():
                    async with client.stream("POST", url, json=payload) as response:
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    yield data.get("response", "")
                                except json.JSONDecodeError:
                                    continue
                return stream_response()
    
    def prepare_context(
        self,
        context_chunks: List[Dict],
        max_tokens: int = 2048
    ) -> List[Dict]:
        """Prepare context by selecting most relevant chunks within token limit."""
        # Sort chunks by relevance (distance)
        sorted_chunks = sorted(context_chunks, key=lambda x: x['distance'])
        
        selected_chunks = []
        total_length = 0
        avg_tokens_per_char = 0.25  # Rough estimate
        
        for chunk in sorted_chunks:
            chunk_length = len(chunk['content']) * avg_tokens_per_char
            if total_length + chunk_length > max_tokens:
                break
            
            selected_chunks.append(chunk)
            total_length += chunk_length
        
        return selected_chunks
    
    def format_chat_history(
        self,
        messages: List[Dict],
        max_messages: int = 5
    ) -> List[Dict]:
        """Format and limit chat history."""
        # Keep only the last few messages
        recent_messages = messages[-max_messages:]
        
        # Format messages
        formatted_history = []
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                formatted_history.append({
                    "user": recent_messages[i]["content"],
                    "assistant": recent_messages[i + 1]["content"]
                })
        
        return formatted_history