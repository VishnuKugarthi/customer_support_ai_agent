# backend/tools/knowledge_base_tools.py
import json
import os
import uuid
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import (
    tool as langchain_tool,
)  # Import the tool decorator and alias it


# --- Knowledge Base Data Loading ---
def load_json_data(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Data file not found at {filepath}. Returning empty data.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filepath}. Returning empty data.")
        return {}


FAQ_KB = load_json_data(
    os.path.join(os.path.dirname(__file__), "../data/faq_knowledge_base.json")
)
TECH_KB = load_json_data(
    os.path.join(os.path.dirname(__file__), "../data/tech_kb.json")
)
BILLING_DB = load_json_data(
    os.path.join(os.path.dirname(__file__), "../data/billing_db.json")
)

# --- Tool Definitions (Raw Python Functions) ---


def get_faq_answer(query: str) -> str:
    """
    Looks up an answer to a common customer question in the FAQ knowledge base.
    Use this for general inquiries like 'What are your hours?' or 'How do I reset my password?'.
    """
    for q, a in FAQ_KB.items():
        if q.lower() in query.lower():
            return a
    return "I could not find an answer to your question in the FAQ. Please try rephrasing or ask for human assistance."


def get_tech_solution(issue: str) -> str:
    """
    Retrieves a technical solution from the knowledge base for common tech issues.
    Use this for problems like 'internet not working' or 'app crashing'.
    """
    for tech_issue, solution in TECH_KB.items():
        if tech_issue.lower() in issue.lower():
            return solution
    return "I could not find a specific solution for this technical issue in our knowledge base. It might require further investigation or escalation."


def get_billing_info(customer_id: str) -> str:
    """
    Retrieves billing information for a specific customer ID from the billing database.
    Use this for queries like 'What's my bill for customer_101?' or 'Check payment status for customer_555'.
    """
    info = BILLING_DB.get(customer_id)
    if info:
        return (
            f"Customer ID: {customer_id}, Name: {info['name']}, "
            f"Balance: {info['balance']}, Last Payment: {info['last_payment_date']}, "
            f"Plan: {info['plan']}."
        )
    return f"Could not find billing information for customer ID: {customer_id}. Please verify the ID."


# Define the Pydantic model for the tool's arguments
class EscalateToHumanArgs(BaseModel):
    summary: str = Field(
        ...,
        description="A concise summary of the user's issue and what steps have already been attempted by the AI.",
    )
    user_email: Optional[str] = Field(
        None,
        description="Optional: The user's email address for contact. Defaults to 'customer@example.com' if not provided.",
    )


# --- Raw Function for Escalation Logic (for direct calls in main.py) ---
# This is the actual Python function that contains the escalation logic.
def _raw_escalate_to_human_logic(summary: str, user_email: Optional[str] = None) -> str:
    """
    (Internal) Contains the core logic for escalating to a human agent.
    """
    ticket_id = str(uuid.uuid4()).replace("-", "")[:8].upper()
    final_email = user_email if user_email else "customer@example.com"

    print(f"\n--- Escalating to Human Support ---")
    print(f"Summary for human agent: {summary}")
    print(f"Generated Ticket ID: {ticket_id}")
    print(
        f"Simulating email to {final_email}: Your issue has been escalated. Your ticket number is {ticket_id}. A human agent will contact you shortly."
    )
    print(f"-----------------------------------\n")

    return (
        f"The issue has been escalated to a human support agent. Your ticket number is {ticket_id}. "
        f"A confirmation has been sent to {final_email}."
    )


# --- LangChain Tool for Agents (using the decorator) ---
# This is the version agents will "call" using LangChain's tool mechanism.
@langchain_tool(args_schema=EscalateToHumanArgs)
def escalate_to_human_tool(summary: str, user_email: Optional[str] = None) -> str:
    """
    Escalates the current customer issue to a human support agent.
    Provide a summary of the issue and any relevant customer contact information like email.
    """
    return _raw_escalate_to_human_logic(summary, user_email)


# --- Expose the raw function for direct use in orchestration ---
# This is the function main.py will import when it needs to call escalation directly.
direct_escalate_to_human = _raw_escalate_to_human_logic
