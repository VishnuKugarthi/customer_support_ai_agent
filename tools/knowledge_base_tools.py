import json
import uuid  # Import uuid for generating unique IDs
from langchain_core.tools import tool

# Load simulated data (in a real app, this would be database queries)
try:
    with open("data/faq_knowledge_base.json", "r") as f:
        FAQ_KB = json.load(f)
except FileNotFoundError:
    FAQ_KB = {}
    print("Warning: faq_knowledge_base.json not found. FAQ tool will be empty.")

try:
    with open("data/tech_kb.json", "r") as f:
        TECH_KB = json.load(f)
except FileNotFoundError:
    TECH_KB = {}
    print("Warning: tech_kb.json not found. Tech support tool will be empty.")

try:
    with open("data/billing_db.json", "r") as f:
        BILLING_DB = json.load(f)
except FileNotFoundError:
    BILLING_DB = {}
    print("Warning: billing_db.json not found. Billing tool will be empty.")


@tool
def get_faq_answer(query: str) -> str:
    """
    Searches the FAQ knowledge base for an answer to a common question.
    Input should be a concise question or keywords.
    """
    query_lower = query.lower()
    for q, a in FAQ_KB.items():
        if query_lower in q.lower():
            return a
    return "I could not find a direct answer in the FAQs. Please try rephrasing or consider if it's a technical or billing issue."


@tool
def get_tech_solution(issue: str) -> str:
    """
    Provides a solution for common technical issues.
    Input should be a clear description of the technical problem.
    """
    issue_lower = issue.lower()
    for i, s in TECH_KB.items():
        if issue_lower in i.lower():
            return s
    return "I could not find a specific solution for this technical issue in our knowledge base. It might require further investigation or escalation."


@tool
def get_billing_info(customer_id: str) -> str:
    """
    Retrieves billing information for a given customer ID.
    Input should be the customer's ID (e.g., 'customer_101').
    """
    info = BILLING_DB.get(customer_id)
    if info:
        return (
            f"Customer ID: {customer_id}, Name: {info['name']}, "
            f"Balance: {info['balance']}, Last Payment: {info['last_payment_date']}, "
            f"Plan: {info['plan']}"
        )
    return f"No billing information found for customer ID: {customer_id}. Please verify the ID."


@tool
def escalate_to_human(summary: str, user_email: str = "customer@example.com") -> str:
    """
    Simulates escalating the issue to a human support agent.
    This tool should be called when an AI agent cannot resolve the issue.
    Input should be a concise summary of the problem and previous attempts to resolve it.
    Optionally, provide the user's email address if known.
    """
    ticket_id = str(uuid.uuid4())[:8].upper()  # Generate a short unique ID
    print(f"\n--- Escalating to Human Support ---")
    print(f"Summary for human agent: {summary}")
    print(f"Generated Ticket ID: {ticket_id}")
    print(
        f"Simulating email to {user_email}: Your issue has been escalated. Your ticket number is {ticket_id}. A human agent will contact you shortly."
    )
    print(f"-----------------------------------\n")
    return f"The issue has been escalated to a human support agent. Your ticket number is {ticket_id}. A confirmation has been sent to {user_email}."
