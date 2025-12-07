import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma


load_dotenv()

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
    model="bge-large:latest",
    check_embedding_ctx_length=False
)
vector_store = Chroma(
    collection_name="example_rag_with_bge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

model = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.3, timeout=30, max_tokens=96e3, max_retries=5
)

agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt="你可以会使用检索工具来回答用户的问题。",
)


# 第一轮对话
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "讲一下3i/Atlas"}]}
)

for message in resp["messages"]:
    message.pretty_print()

