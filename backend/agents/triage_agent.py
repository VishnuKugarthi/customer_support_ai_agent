# backend/agents/triage_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler

from tools.knowledge_base_tools import get_faq_answer


def create_triage_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    triage_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer support triage agent. Your primary goal is to understand the user's problem "
                "and either answer it directly using the provided FAQ tool, or determine if it's a technical "
                "or billing-related issue that needs to be routed to a specialized agent. "
                "Your response should be concise and directly actionable by the system orchestrator. "
                "If you need to ask a clarifying question to determine the type of issue, do so. "
                "However, once you are certain of the issue type, your FINAL response must be a routing instruction or an FAQ answer. "
                "Do NOT engage in further conversation once a route is determined or an FAQ is answered. "
                "Do NOT call `escalate_to_human` directly; this is handled by specialized agents if they cannot resolve. \n\n"
                "**Instructions for Final Response:**\n"
                "- If you can answer the question using the `get_faq_answer` tool, provide the answer directly. Example: 'Our support hours are Monday to Friday, 9 AM to 5 PM EST.'\n"
                "- If the query is clearly about a technical problem (e.g., 'internet not working', 'app crashing', 'software installation'), "
                "your FINAL response MUST be exactly 'ROUTE_TECH'.\n"
                "- If the query is clearly about billing, payments, invoices, or account balance (e.g., 'my bill', 'payment due', 'invoice'), "
                "your FINAL response MUST be exactly 'ROUTE_BILLING'.\n"
                "- If you need more information to classify or answer, ask a clarifying question. Example: 'Could you please clarify what kind of issue you are experiencing?'",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    triage_tools = [get_faq_answer]

    triage_agent = create_tool_calling_agent(llm, triage_tools, triage_prompt)

    # Initialize AgentExecutor with updated configuration
    triage_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=triage_agent,
        tools=triage_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    return triage_agent_executor
