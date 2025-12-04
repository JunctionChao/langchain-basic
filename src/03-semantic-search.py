from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


embeddings = OpenAIEmbeddings(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
    model="bge-large:latest",
    # model='quentinz/bge-large-zh-v1.5:latest',
    # model='nomic-embed-text:latest',
    check_embedding_ctx_length=False # 关闭上下文长度检查, 默认情况下会执行tokenization过程来检查文本长度，这与Ollama的OpenAI兼容API不完全兼容
)

# 向量库
vector_store = Chroma(
    collection_name="example_archive_with_bge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 相似向量查询
query = "张三九的基本信息是什么？"
docs = vector_store.similarity_search(query, k=3)
for doc in docs:
    print(doc.page_content[:30], '...')

# 带分数的相似度查询
docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
for doc, score in docs_with_scores: # score越小, 表示越相似
    print(f"Score: {score:.4f}")
    print(doc.page_content[:30], '...')
    print()

# 手动向量化，然后查询
query_vector = embeddings.embed_query(query)
docs = vector_store.similarity_search_by_vector(query_vector, k=3)
for doc in docs:
    print(doc.page_content[:30], '...')



# langchain概念： 将大模型，提示词，tools, output_parser 等组件组合起来，形成一个完整的链, 要求每个环节是Runnable类型
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain  # function to Runnable
def retriever(query: str, k: int = 3) -> list[Document]:
    return vector_store.similarity_search(query, k=k)

print('====封装为检索器retriever:')
result = retriever.invoke(query, k=2)
for doc in result:
    print(doc.page_content[:30], '...')