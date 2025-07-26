# backend/main.py (Updated orchestration logic)
import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Import agent creation functions
from agents.triage_agent import create_triage_agent
from agents.tech_agent import create_tech_agent
from agents.billing_agent import create_billing_agent

# Import the *direct* function for orchestration, not the tool object
from tools.knowledge_base_tools import direct_escalate_to_human

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:52330",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for handling email collection (simple for demo, not multi-user safe)
_waiting_for_email: bool = False
_escalation_summary_context: Optional[str] = None
_original_query_context: Optional[str] = None

# Initialize the LLM with proper configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Replace with a valid model name
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True,  # Important for Gemini
)

try:
    # Initialize agents
    triage_agent_executor = create_triage_agent(llm)
    tech_agent_executor = create_tech_agent(llm)
    billing_agent_executor = create_billing_agent(llm)
except Exception as e:
    print(f"Error initializing agents: {str(e)}")
    raise


# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []


# Helper function to convert chat history to LangChain message format
def format_chat_history(history: List[Dict[str, str]]) -> List[Any]:
    formatted_history = []
    for entry in history:
        if entry["role"] == "user":
            formatted_history.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "ai":
            formatted_history.append(AIMessage(content=entry["content"]))
    return formatted_history


# Helper to extract email
def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else None


def _extract_context_from_history(chat_history: List[Dict[str, str]]) -> str:
    """Extract relevant context from chat history for escalation."""
    # Get the last 3 exchanges (or all if less than 3)
    recent_exchanges = chat_history[-3:] if len(chat_history) > 3 else chat_history

    # Build context string from recent exchanges
    context_parts = []
    for msg in recent_exchanges:
        role = "User" if msg["role"] == "user" else "Agent"
        content = msg["content"]
        context_parts.append(f"{role}: {content}")

    return " | ".join(context_parts)


# The core logic for handling customer queries, now with email collection
async def handle_customer_query_backend(
    query: str, raw_chat_history: List[Dict[str, str]]
) -> str:
    global _waiting_for_email, _escalation_summary_context, _original_query_context

    formatted_history = format_chat_history(raw_chat_history)
    response = ""
    extracted_email = extract_email(query)  # Try to extract email from current turn

    # --- Direct Human Escalation Request ---
    if any(
        phrase in query.lower()
        for phrase in [
            "connect me to human",
            "connect with human",
            "talk to human",
            "speak with human",
        ]
    ):
        if extracted_email:
            # If email is provided in the same message as escalation request
            context = _extract_context_from_history(raw_chat_history)
            final_summary = f"Customer requested direct escalation. Context: {context}"
            response = direct_escalate_to_human(
                summary=final_summary, user_email=extracted_email
            )
            return response
        else:
            _waiting_for_email = True
            _escalation_summary_context = _extract_context_from_history(
                raw_chat_history
            )
            return "I'll help you connect with a human agent. Could you please provide your email address so we can create a support ticket?"

    # --- State Management for Email Collection ---
    if _waiting_for_email:
        if extracted_email:
            # Email received, proceed with final escalation using the direct function
            print(
                f"Orchestrator: Email '{extracted_email}' received. Finalizing escalation."
            )
            final_summary = (
                _escalation_summary_context
                if _escalation_summary_context
                else "Issue requiring human attention."
            )
            response = direct_escalate_to_human(
                summary=final_summary, user_email=extracted_email
            )
            _waiting_for_email = False  # Reset state
            _escalation_summary_context = None
            _original_query_context = None
            return response
        else:
            # Still waiting for email, user didn't provide one this turn
            return "I'm still waiting for your email address to escalate this. Could you please provide it?"

    # --- Initial Query Processing (Triage) ---
    print("Triage Agent: Analyzing query intent...")
    triage_result = triage_agent_executor.invoke(
        {"input": query, "chat_history": formatted_history}
    )
    triage_output = triage_result["output"].strip()
    print(f"Triage Agent Output: {triage_output}")

    # If FAQ resolves the query, return the answer
    if not triage_output.startswith("ROUTE_TECH") and not triage_output.startswith(
        "ROUTE_BILLING"
    ):
        return triage_output

    # --- Routing if FAQ is Insufficient ---
    if triage_output.startswith("ROUTE_TECH"):
        print("Orchestrator: Routing to Technical Support Agent.")
        tech_result = tech_agent_executor.invoke(
            {"input": query, "chat_history": formatted_history}
        )
        agent_output = tech_result["output"].strip()
        if agent_output.startswith("NEED_EMAIL_FOR_ESCALATION:"):
            _waiting_for_email = True
            _escalation_summary_context = agent_output.replace(
                "NEED_EMAIL_FOR_ESCALATION:", ""
            ).strip()
            _original_query_context = query
            response = "I need your email address to escalate this technical issue. Could you please provide it?"
        else:
            response = agent_output
    elif triage_output.startswith("ROUTE_BILLING"):
        print("Orchestrator: Routing to Billing Agent.")
        billing_result = billing_agent_executor.invoke(
            {"input": query, "chat_history": formatted_history}
        )
        agent_output = billing_result["output"].strip()
        if agent_output.startswith("NEED_EMAIL_FOR_ESCALATION:"):
            _waiting_for_email = True
            _escalation_summary_context = agent_output.replace(
                "NEED_EMAIL_FOR_ESCALATION:", ""
            ).strip()
            _original_query_context = query
            response = "I need your email address to escalate this billing issue. Could you please provide it?"
        else:
            response = agent_output

    return response


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        agent_response = await handle_customer_query_backend(
            request.message, request.chat_history
        )
        return {"response": agent_response}
    except Exception as e:
        print(f"Error processing chat request: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
async def root():
    return {"message": "AI Customer Support Backend is running!"}
    return {"message": "AI Customer Support Backend is running!"}
