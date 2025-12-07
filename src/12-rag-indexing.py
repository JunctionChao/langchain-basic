from langchain_community.document_loaders import WebBaseLoader
import bs4
from pprint import pprint


# 1. 读取网页    List[Document]
page_url = "https://news.cctv.com/2025/10/30/ARTIeIcUkKPBZ1zRX3iy8Zwa251030.shtml"
bs4_strainer = bs4.SoupStrainer()
loader = WebBaseLoader(page_url, bs_kwargs={"parse_only": bs4_strainer})
docs = loader.load()
print(len(docs), type(docs[0]))  # 1  <class 'langchain_core.documents.base.Document'>
pprint(docs[0])


# 2. 分割文本，文本块    List[Document]
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每个文本块的最大字符数
    chunk_overlap=200,  # 块之间的重叠字符数
    add_start_index=True,  # 添加起始索引
)

splits = text_splitter.split_documents(docs)
print(len(splits), type(splits[0]))  # 5  <class 'langchain_core.documents.base.Document'>
pprint(splits[0])


# 3. 向量化文本块    List[Document]
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
    model="bge-large:latest",
    # model='quentinz/bge-large-zh-v1.5:latest',
    # model='nomic-embed-text:latest',
    check_embedding_ctx_length=False # 关闭上下文长度检查, 默认情况下会执行tokenization过程来检查文本长度，这与Ollama的OpenAI兼容API不完全兼容
)
# # 向量化两种策略，用于建档embed_documents，用于查询embed_query
# vectors = embeddings.embed_documents([split.page_content for split in splits])
# print(len(vectors[0]))  # 1024


# 4. 向量持久化数据库
from langchain_chroma import Chroma


vector_store = Chroma(
    collection_name="example_rag_with_bge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
if vector_store._collection_name in vector_store._client.list_collections():
    vector_store._client.delete_collection(vector_store._collection_name)
    vector_store._client.create_collection(vector_store._collection_name)

ids = vector_store.add_documents(splits)
print(len(ids)) # 5
print(ids)
