# agents/triage_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools.knowledge_base_tools import (
    get_faq_answer,
)  # Removed escalate_to_human from import if not used directly by Triage Agent


def create_triage_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    triage_prompt_template = """
    You are a customer support triage agent. Your primary goal is to understand the user's problem
    and either answer it directly using the provided FAQ tool, or determine if it's a technical
    or billing-related issue that needs to be routed to a specialized agent.

    If you can answer the question using the `get_faq_answer` tool, provide the answer directly as your Final Answer.
    If the query is clearly about a technical problem (e.g., "internet not working", "app crashing", "software installation"),
    your Final Answer MUST be exactly "ROUTE_TECH".
    If the query is clearly about billing, payments, invoices, or account balance (e.g., "my bill", "payment due", "invoice"),
    your Final Answer MUST be exactly "ROUTE_BILLING".
    Do NOT call `escalate_to_human` directly. This is handled by specialized agents if they cannot resolve.

    Tools available to you:
    {tools}

    Use the following format for your thinking process:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, which must be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer or routing instruction
    Final Answer: the final answer to the original input question (e.g., FAQ answer, ROUTE_TECH, or ROUTE_BILLING)

    Current conversation:
    {chat_history}
    User query: {input}
    {agent_scratchpad}
    """

    triage_prompt = PromptTemplate.from_template(triage_prompt_template)
    # Triage agent only needs get_faq_answer. escalate_to_human is for specialist agents.
    triage_tools = [get_faq_answer]  # Removed escalate_to_human from tools list

    triage_agent = create_react_agent(llm, triage_tools, triage_prompt)
    triage_agent_executor = AgentExecutor(
        agent=triage_agent, tools=triage_tools, verbose=True, handle_parsing_errors=True
    )
    return triage_agent_executor
