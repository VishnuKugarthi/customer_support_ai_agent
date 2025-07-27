# backend/tools/knowledge_base_tools.py
import json
import os
import uuid
from typing import Optional
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from pydantic import BaseModel, Field
from langchain_core.tools import (
    tool as langchain_tool,
    tool,  # Import the tool decorator
)


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


@tool
def get_faq_answer(query: str) -> str:
    """
    Looks up an answer to a common customer question in the FAQ knowledge base.
    Use this for general inquiries like 'What are your hours?' or 'How do I reset my password?'.
    """
    query_words = set(query.lower().split())

    # First try exact phrase matching
    for faq_q, faq_a in FAQ_KB.items():
        if faq_q.lower() in query.lower():
            return faq_a

    # Then try keyword matching
    best_match = None
    max_word_match = 0

    for faq_q, faq_a in FAQ_KB.items():
        faq_words = set(faq_q.lower().split())
        matching_words = query_words.intersection(faq_words)

        # Check if this is a better match than what we've seen
        if len(matching_words) > max_word_match:
            max_word_match = len(matching_words)
            best_match = faq_a

        # If we match most of the words in either the query or the FAQ question
        if len(matching_words) >= min(len(query_words), len(faq_words)) * 0.7:
            return faq_a

    # If we found a decent partial match, use it
    if best_match and max_word_match >= 2:
        return best_match

    return "I could not find an answer to your question in the FAQ. Please try rephrasing or ask for human assistance."


@tool
def get_tech_solution(issue: str) -> str:
    """
    Returns a solution from the tech knowledge base using keyword matching.
    """
    issue_lower = issue.lower()
    # Exact match first
    if issue_lower in TECH_KB:
        return TECH_KB[issue_lower]

    # Fuzzy/keyword match
    for kb_key in TECH_KB:
        if kb_key in issue_lower or issue_lower in kb_key:
            return TECH_KB[kb_key]

    # Partial word match
    for kb_key in TECH_KB:
        if any(word in kb_key for word in issue_lower.split()):
            return TECH_KB[kb_key]

    return "Sorry, I couldn't find a solution for your issue. Please provide more details or contact support."


@tool
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
    ticket_id = str(uuid.uuid4()).replace("-", "")[:8].upper()
    final_email = user_email if user_email else "customer@example.com"

    subject = f"Support Ticket #{ticket_id} Created"
    body = f"""Dear Customer,\n\nYour support request has been received and escalated to our support team.\n\nTicket Details:\n- Ticket ID: {ticket_id}\n- Status: Open\n- Summary: {summary}\n\nA support representative will contact you shortly to assist you with your issue.\n\nPlease keep this ticket number for your reference: {ticket_id}\n\nIf you need to follow up on this ticket, please reply to this email or contact our support team with your ticket number.\n\nBest regards,\nCustomer Support Team,\nVishnu."""
    send_email(final_email, subject, body)

    print(f"\n--- Escalating to Human Support ---")
    print(f"Summary for human agent: {summary}")
    print(f"Generated Ticket ID: {ticket_id}")
    print(f"Email sent to {final_email}")
    print(f"-----------------------------------\n")

    return (
        f"The issue has been escalated to a human support agent. Your ticket number is {ticket_id}. "
        f"A confirmation has been sent to {final_email}."
    )


def send_email(to_email: str, subject: str, body: str):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM_EMAIL")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(from_email, [to_email], msg.as_string())
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


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
