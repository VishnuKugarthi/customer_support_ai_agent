# backend/agents/billing_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.knowledge_base_tools import get_billing_info, escalate_to_human_tool


def create_billing_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    billing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a billing support agent. Your goal is to assist with billing inquiries, invoices, "
                "and payment issues using the `get_billing_info` tool. You can retrieve details for a customer "
                "if provided with a customer ID (e.g., 'customer_101'). "
                "If you cannot resolve the issue with the provided tools or information, you MUST escalate. "
                "When escalating, first check if the user's email is present in the current query or chat history. "
                "If an email is found (e.g., 'my email is example@domain.com'), use the `escalate_to_human_tool` tool with the extracted email. "
                "If NO email is found, your FINAL response MUST be exactly 'NEED_EMAIL_FOR_ESCALATION: [concise summary of issue]'. "
                "Do NOT call `escalate_to_human_tool` if you don't have an email. "
                "If you need more information to provide a solution, ask a clarifying question.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    billing_tools = [get_billing_info, escalate_to_human_tool]

    billing_agent = create_tool_calling_agent(llm, billing_tools, billing_prompt)
    # REMOVE handle_parsing_errors=True
    billing_agent_executor = AgentExecutor(
        agent=billing_agent, tools=billing_tools, verbose=True
    )
    return billing_agent_executor
