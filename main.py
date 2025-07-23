import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Import agent creation functions
from agents.triage_agent import create_triage_agent
from agents.tech_agent import create_tech_agent
from agents.billing_agent import create_billing_agent

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")

# Initialize the LLM (Google Gemini Pro)
# temperature=0.0 makes the model more deterministic and factual, good for support
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0
)

# Create instances of each specialized agent
triage_agent_executor = create_triage_agent(llm)
tech_agent_executor = create_tech_agent(llm)
billing_agent_executor = create_billing_agent(llm)


def handle_customer_query(query: str, chat_history: list) -> str:
    """
    Main function to handle customer queries by routing them through the appropriate agents.
    """
    print(f"\n--- User Query: {query} ---")
    response = ""

    # 1. Triage the query using the Triage Agent
    print("Triage Agent: Analyzing query intent...")
    # The triage agent's output will either be a direct answer,
    # a routing instruction (e.g., "ROUTE_TECH"), or an escalation message.
    triage_result = triage_agent_executor.invoke(
        {"input": query, "chat_history": chat_history}
    )
    triage_output = triage_result["output"].strip()
    print(f"Triage Agent Output: {triage_output}")

    # 2. Orchestrator decides next step based on Triage Agent's output
    if triage_output.startswith("ROUTE_TECH"):
        print("Orchestrator: Routing to Technical Support Agent.")
        tech_result = tech_agent_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
        response = tech_result["output"]
    elif triage_output.startswith("ROUTE_BILLING"):
        print("Orchestrator: Routing to Billing Agent.")
        billing_result = billing_agent_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
        response = billing_result["output"]
    else:
        # If Triage Agent resolved it directly (e.g., FAQ answer) or escalated itself
        response = triage_output

    return response


def main():
    """
    Runs the main conversation loop for the customer support system.
    """
    print("Welcome to the Multi-Agent Customer Support System!")
    print("Type 'exit' to end the conversation.")
    print(
        "Try queries like: 'How do I reset my password?', 'My internet is not working.', 'What's my bill for customer_101?', 'I need to speak to a human, my email is test@example.com'."
    )

    chat_history = []  # Stores the conversation history for context

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Thank you for using our support system. Goodbye!")
            break

        # Handle the user's query through the agent system
        response = handle_customer_query(user_input, chat_history)
        print(f"Agent: {response}")

        # Update chat history for context in subsequent turns
        # LangChain's agents often manage their own internal history for tool use,
        # but maintaining an external chat_history list is good for longer conversations
        # and for passing to the top-level agent.
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
