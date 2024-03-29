from typing import List, Union, Any, Iterator

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from langchain_core.runnables import RunnableGenerator
from langchain_core.utils.function_calling import format_tool_to_openai_tool
from langchain_openai import ChatOpenAI
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from langserve.pydantic_v1 import BaseModel, Field

load_dotenv()

# Setup the FastAPI web service
app = FastAPI(
    title="Simple OpenAI Tools Agent",
    version="1.0",
    description="A simple OpenAI Tools agent that uses a chat model to interact with the user and a search tool to provide information.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


search_tool = DuckDuckGoSearchRun()


# Pre-defined prompt with react logic
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

tools = [search_tool]
prompt = hub.pull("hwchase17/openai-tools-agent")


llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


class Input(BaseModel):
    input: str
    # T he field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )


class Output(BaseModel):
    output: Any


async def gen(input: Iterator[Any]) -> Iterator[int]:
    async for x in input:
        yield AIMessage(content=x['output'])


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output)
                  .with_config({"run_name": "hello-world-agent"}),
    path="/openai-tools-agent",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
