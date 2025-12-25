from typing import List

from langchain_classic.agents import create_react_agent, AgentExecutor
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS






from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch




class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(..., description="The URL to the source")

class AgentResponse(BaseModel):
    """Schema for a response for agent with answer and sources"""
    answer:str = Field(..., description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="A list of sources to generate the answer")



llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
tools = [
    TavilySearch(),
]
react_prompt = hub.pull("hwchase17/react")
#agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"]

).partial(format_instructions="")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm

def main():
    result = chain.invoke(
        input={
            "input": "your query"
        }
    )
    print(result)



if __name__ == "__main__":
    main()
