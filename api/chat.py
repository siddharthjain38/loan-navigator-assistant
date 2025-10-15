"""
Chat endpoints for handling policy queries.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.supervisor import Supervisor

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

# Initialize simple supervisor - all logic in one place!
supervisor = Supervisor()

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str

class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str

@router.post("/")
def process_chat(request: ChatRequest):
    """Process a chat message and return a response."""
    try:
        print(f"Received chat request: {request.message}")
        agent_response = supervisor.process(request.message)
        print(f"Got response: {agent_response.answer}")
        return ChatResponse(
            answer=agent_response.answer
        )
    except Exception as e:
        print(f"Error in process_chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )