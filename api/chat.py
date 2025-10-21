"""
Chat endpoints for handling policy queries with conversation memory.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from agents.supervisor import Supervisor
from core.factory import get_workflow_engine

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize supervisor and workflow engine
supervisor = Supervisor()
workflow_engine = get_workflow_engine()

# In-memory conversation storage (use Redis/database for production)
conversations: Dict[str, List[Dict]] = {}


class Message(BaseModel):
    """Single message in conversation."""

    role: str
    content: str
    timestamp: str


class ChatRequest(BaseModel):
    """Chat request model with session support."""

    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model with session and history."""

    answer: str
    session_id: str
    conversation_history: List[Message]


def get_conversation_context(session_id: str, limit: int = 5) -> str:
    """Get recent conversation history as context."""
    if session_id not in conversations:
        return ""

    recent_messages = conversations[session_id][-limit:]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    return context


@router.post("/")
def process_chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message with conversation memory."""
    try:
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Initialize conversation if new session
        if session_id not in conversations:
            conversations[session_id] = []

        print(f"Received chat request: {request.message} (session: {session_id})")

        # Add user message to history
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat(),
        }
        conversations[session_id].append(user_message)

        # Get conversation context for better responses
        context = get_conversation_context(session_id)

        # Process with supervisor (pass context and session_id)
        agent_response = supervisor.process(
            request.message, context=context if context else None, session_id=session_id
        )

        # Add assistant response to history
        assistant_message = {
            "role": "assistant",
            "content": agent_response.answer,
            "timestamp": datetime.now().isoformat(),
        }
        conversations[session_id].append(assistant_message)

        print(f"Got response: {agent_response.answer[:100]}...")

        # Return response with full conversation history
        return ChatResponse(
            answer=agent_response.answer,
            session_id=session_id,
            conversation_history=[Message(**msg) for msg in conversations[session_id]],
        )

    except Exception as e:
        print(f"Error in process_chat: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/history/{session_id}")
def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in conversations:
        return {"history": []}
    return {"history": conversations[session_id]}


@router.delete("/history/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    if session_id in conversations:
        del conversations[session_id]
    # Also clear workflow engine context
    workflow_engine.clear_session_context(session_id)
    return {"message": "Conversation cleared"}
