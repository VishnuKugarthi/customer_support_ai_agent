# backend/agents/triage_agent.py
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain

from tools.knowledge_base_tools import get_faq_answer


def create_triage_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    tools = [get_faq_answer]
    
    prefix = """You are a customer support triage agent. Your primary goal is to understand the user's problem 
    and either answer it directly using the provided FAQ tool, or determine if it's a technical or billing-related issue."""
    
    suffix = """Begin! Remember to use tools when necessary and be helpful.

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )

    try:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.__name__ for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=3
        )
        
        return agent_executor
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        raise
            llm=llm, tools=triage_tools, system_message=triage_prompt, verbose=True
        )

        # Initialize AgentExecutor with proper configuration
        triage_agent_executor = AgentExecutor(
            agent=agent,
            tools=triage_tools,
            max_iterations=3,
            early_stopping_method="generate",
            verbose=True,
        )

        return triage_agent_executor

    except Exception as e:
        print(f"Error in create_triage_agent: {str(e)}")
        raise
