# agents/billing_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools.knowledge_base_tools import get_billing_info, escalate_to_human


def create_billing_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    billing_prompt_template = """
    You are a billing support agent. Your goal is to assist with billing inquiries, invoices,
    and payment issues using the `get_billing_info` tool. You can retrieve details for a customer
    if provided with a customer ID (e.g., 'customer_101').
    If you cannot resolve the issue with the provided tools or information, or if the user
    requires further assistance that you cannot provide, you MUST use the `escalate_to_human` tool
    with a clear summary of the issue and what has been attempted.
    If the user provides their email in the query (e.g., "my email is example@domain.com"), extract it and pass it to the `escalate_to_human` tool.

    Tools available to you:
    {tools}

    Use the following format for your thinking process:
    Question: the input billing question
    Thought: you should always think about what to do to resolve the billing query using the available tools.
    Action: the action to take, which must be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer or if escalation is needed.
    Final Answer: the final answer to the billing inquiry, or the output of escalate_to_human.

    Current conversation:
    {chat_history}
    User query: {input}
    {agent_scratchpad}
    """

    billing_prompt = PromptTemplate.from_template(billing_prompt_template)
    billing_tools = [get_billing_info, escalate_to_human]

    billing_agent = create_react_agent(llm, billing_tools, billing_prompt)
    billing_agent_executor = AgentExecutor(
        agent=billing_agent,
        tools=billing_tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return billing_agent_executor
