from langchain_openai import OpenAI
from fastapi import FastAPI
from langserve import add_routes

llm = OpenAI()
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
# Adding chain route
add_routes(app, llm, path="/first_llm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
