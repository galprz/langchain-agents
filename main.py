
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from web_agent import init, graph
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI()

add_routes(
    app,
    graph,
    path="/talk",
)

if __name__ == "__main__":
    import uvicorn

    init()

    uvicorn.run(app, host="localhost", port=8000)
