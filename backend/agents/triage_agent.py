# backend/agents/triage_agent.py
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.knowledge_base_tools import get_faq_answer


def create_triage_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    tools = [get_faq_answer]

    triage_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a customer support triage agent. Your primary goal is to help customers by:
            1. Finding answers in the FAQ knowledge base using the get_faq_answer tool
            2. Routing complex issues to specialized agents

            IMPORTANT: FAQ HANDLING
            - ALWAYS use the get_faq_answer tool first for ANY user question
            - If the tool returns an answer, provide it to the user exactly as received
            - Only route to specialized agents if the FAQ doesn't have an answer
            - For general questions about policies, hours, contact info, etc., use the FAQ

            Instructions for routing (only if FAQ has no answer):
            1. For technical issues (internet, app, login problems):
               Respond with 'ROUTE_TECH:' + brief description
            2. For billing issues (balance, payments, subscriptions):
               Respond with 'ROUTE_BILLING:' + brief description
            
            Example responses:
            - FAQ answer: Return exact answer from FAQ tool
            - Technical: "ROUTE_TECH: User having internet connectivity issues"
            - Billing: "ROUTE_BILLING: User needs help with payment or billing processing"
            
            Remember: 
            - FAQ answers should be returned exactly as provided by the tool
            - Don't modify or interpret FAQ answers
            - Route to agents only if FAQ tool doesn't have an answer""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    try:
        agent = create_tool_calling_agent(llm, tools, triage_prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, max_iterations=3
        )

        return agent_executor

    except Exception as e:
        print(f"Error in create_triage_agent: {str(e)}")
        raise
