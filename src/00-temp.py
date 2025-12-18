# from langsmith import Client
# from langchain_openai import ChatOpenAI
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # 从langsmith获取提示词模板
# client = Client()
# prompt = client.pull_prompt("rlm/rag-prompt")
# print(prompt.messages)
# print(prompt.messages[0].prompt.template)
# # https://docs.langchain.com/langsmith/manage-prompts-programmatically



# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma

# embeddings = OpenAIEmbeddings(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama", 
#     model="bge-large:latest",
#     check_embedding_ctx_length=False
# )
# vector_store = Chroma(
#     collection_name="example_archive_with_bge",
#     embedding_function=embeddings,
#     persist_directory="./chroma_db",
# )
# retriever = vector_store.as_retriever()
# res = retriever.invoke("患者张三九的基本信息？")
# print(res)
# print(len(res), type(res[0]))


from langchain_core.prompts import PromptTemplate

prompt_tmpl = PromptTemplate.from_file(
    "./data/attraction_agent_prompt", encoding="utf-8"
)

print(prompt_tmpl)
print(prompt_tmpl.template)