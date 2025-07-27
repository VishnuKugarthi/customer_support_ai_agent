# backend/main.py (Updated orchestration logic)
import os
import re
import time
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Session management class
class UserSession:
    def __init__(self):
        self.waiting_for_email: bool = False
        self.escalation_summary: Optional[str] = None
        self.original_query: Optional[str] = None
        self.last_interaction: float = time.time()


# Session storage
_user_sessions: Dict[str, UserSession] = {}
_SESSION_TIMEOUT = 3600  # 1 hour timeout


def get_user_session(session_id: str) -> UserSession:
    """Get or create a user session with automatic cleanup."""
    current_time = time.time()

    # Cleanup old sessions
    expired_sessions = [
        sid
        for sid, session in _user_sessions.items()
        if current_time - session.last_interaction > _SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        del _user_sessions[sid]

    # Get or create session
    if session_id not in _user_sessions:
        _user_sessions[session_id] = UserSession()

    _user_sessions[session_id].last_interaction = current_time
    return _user_sessions[session_id]


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
    "http://localhost:3000",
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


# Helper function to handle billing queries
async def handle_billing_query(
    query: str, formatted_history: List[Any], customer_id: Optional[str] = None
) -> str:
    """Handle billing-related queries with optional customer ID."""
    enhanced_query = query
    if customer_id:
        enhanced_query = f"Process this billing query for {customer_id}: {query}"
    else:
        # Try to extract customer ID from query if not provided
        match = re.search(r"customer[_ ](\d+)", query.lower())
        if match:
            customer_id = f"customer_{match.group(1)}"
            enhanced_query = f"Process this billing query for {customer_id}: {query}"

    print(f"Processing billing query: {enhanced_query}")
    billing_result = billing_agent_executor.invoke(
        {"input": enhanced_query, "chat_history": formatted_history}
    )
    return billing_result["output"].strip()


# The core logic for handling customer queries, with session-based state management
async def handle_customer_query_backend(
    query: str, raw_chat_history: List[Dict[str, str]], session: UserSession
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

    # Process triage output and determine routing
    cleaned_triage_output = triage_output.strip()

    # Check for customer ID pattern in the query first
    customer_id_match = re.search(r"customer[_ ](\d+)", query.lower())
    if customer_id_match:
        print("Direct billing route: Customer ID detected")
        # Normalize customer ID format
        customer_id = f"customer_{customer_id_match.group(1)}"
        # Route directly to billing agent
        return await handle_billing_query(query, formatted_history, customer_id)

    # Then check triage output
    if "ROUTE_TECH:" in cleaned_triage_output:
        cleaned_triage_output = cleaned_triage_output.split("ROUTE_TECH:")[1].strip()
        route_to = "TECH"
    elif "ROUTE_BILLING:" in cleaned_triage_output:
        cleaned_triage_output = cleaned_triage_output.split("ROUTE_BILLING:")[1].strip()
        route_to = "BILLING"
    else:
        # Check for billing keywords before defaulting to FAQ
        billing_keywords = [
            "balance",
            "payment",
            "bill",
            "charge",
            "account",
            "plan",
            "$",
        ]
        if any(keyword in query.lower() for keyword in billing_keywords):
            print("Billing route: Keywords detected")
            route_to = "BILLING"
        else:
            return triage_output  # FAQ response

    # Store the cleaned context for routing
    context = cleaned_triage_output

    print(f"Routing Context: {context}")

    def clean_agent_response(response: str) -> str:
        """Clean any routing or internal tags from agent responses."""
        # List of all internal tags/patterns to remove
        tags_to_remove = [
            "ROUTE_TECH:",
            "ROUTE_TECH",
            "ROUTE_BILLING:",
            "ROUTE_BILLING",
            "NEED_EMAIL_FOR_ESCALATION:",
            "Invoking: `get_tech_solution`",
            "> Entering new AgentExecutor chain...",
            "> Finished chain.",
            "with `{",
            "}`",
        ]

        # First remove any lines containing these patterns
        cleaned_lines = [
            line
            for line in response.split("\n")
            if not any(tag in line for tag in tags_to_remove)
            and not line.startswith(">")
            and not line.startswith("Invoking:")
        ]

        # Join remaining lines and clean up any leftover tags
        cleaned = "\n".join(cleaned_lines)
        for tag in tags_to_remove:
            cleaned = cleaned.replace(tag, "")

        # Clean up extra whitespace and newlines
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        return cleaned.strip()

    # --- Routing if FAQ is Insufficient ---
    if "ROUTE_TECH" in triage_output:
        print("Orchestrator: Routing to Technical Support Agent.")
        # Add context to the query for the technical agent
        enhanced_query = f"{query}\nContext: {context}" if context else query
        tech_result = tech_agent_executor.invoke(
            {"input": enhanced_query, "chat_history": formatted_history}
        )
        technical_response = tech_result["output"].strip()

        if technical_response.startswith("NEED_EMAIL_FOR_ESCALATION:"):
            _waiting_for_email = True
            _escalation_summary_context = clean_agent_response(
                technical_response.replace("NEED_EMAIL_FOR_ESCALATION:", "")
            )
            _original_query_context = query
            response = "I'll need to connect you with our technical specialist for this issue. Could you please provide your email address for follow-up?"
        else:
            response = clean_agent_response(technical_response)

    elif "ROUTE_BILLING" in triage_output:
        print("Orchestrator: Routing to Billing Agent.")
        # Add context to the query for the billing agent
        enhanced_query = f"{query}\nContext: {context}" if context else query
        billing_result = billing_agent_executor.invoke(
            {"input": enhanced_query, "chat_history": formatted_history}
        )
        billing_response = billing_result["output"].strip()

        if billing_response.startswith("NEED_EMAIL_FOR_ESCALATION:"):
            _waiting_for_email = True
            _escalation_summary_context = clean_agent_response(
                billing_response.replace("NEED_EMAIL_FOR_ESCALATION:", "")
            )
            _original_query_context = query
            response = "I'll need to connect you with our billing specialist for this. Could you please provide your email address for follow-up?"
        else:
            response = clean_agent_response(billing_response)

    return response


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle incoming chat requests with session management."""
    try:
        # Generate a session ID if not provided
        session_id = getattr(request, "session_id", None) or str(uuid4())
        session = get_user_session(session_id)

        # Process the query
        agent_response = await handle_customer_query_backend(
            query=request.message,
            raw_chat_history=request.chat_history,
            session=session,
        )

        # Log successful interaction
        print(f"Chat processed successfully - Session: {session_id[:8]}")

        return {
            "response": agent_response,
            "session_id": session_id,
            "requires_action": session.waiting_for_email,
            "action_type": "provide_email" if session.waiting_for_email else None,
        }
    except Exception as e:
        print(f"Error processing chat request: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again.",
        )


@app.get("/")
async def root():
    return {"message": "AI Customer Support Backend is running!"}
    return {"message": "AI Customer Support Backend is running!"}
