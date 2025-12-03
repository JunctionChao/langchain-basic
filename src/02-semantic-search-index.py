from langchain_community.document_loaders import PyPDFLoader  # pip install langchain-community pypdf
from pprint import pprint


# 1. 按页加载pdf文件    List[Document]
file_path = "./pdfs/健康档案.pdf"
loader = PyPDFLoader(file_path, mode='page') # mode in ['page', 'single']
docs = loader.load()
print(len(docs), type(docs[0]))  # 6  <class 'langchain_core.documents.base.Document'>
pprint(docs[0])
# Document(
#   page_content='健康档案...',
#   metadata={
#       'producer': 'macOS 版本14.6.1（版号23G93） Quartz PDFContext', 
#       'creator': 'WPS 文字', 
#       'creationdate': '2024-08-25T11:44:53+03:44', 
#       'author': 'NanGe', 
#       'moddate': '2024-08-25T11:44:53+03:44', 
#       'sourcemodified': "D:20240825114453+03'44'", 
#       'trapped': '/False', 
#       'source': './pdfs/健康档案.pdf', 
#       'total_pages': 6, 
#       'page': 0, 
#       'page_label': '1',
#       ...
#   })


# 2. 分割文本，文本块    List[Document]
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,    # 每个文本块的最大字符数
    chunk_overlap=200,  # 块之间的重叠字符数
    add_start_index=True,  # 添加起始索引
)

splits = text_splitter.split_documents(docs)
print(len(splits), type(splits[0]))  # 10  <class 'langchain_core.documents.base.Document'>
pprint(splits[0])
# Document(
#   page_content='健康档案...',
#   metadata={
#     'producer': 'macOS 版本14.6.1（版号23G93） Quartz PDFContext', 
#     'creator': 'WPS 文字', 
#     'creationdate': '2024-08-25T11:44:53+03:44', 
#     'moddate': '2024-08-25T11:44:53+03:44', 
#     'sourcemodified': "D:20240825114453+03'44'", 
#     'trapped': '/False', 
#     'source': './pdfs/健康档案.pdf', 
#     'total_pages': 6, 
#     'page': 0, 
#     'page_label': '1', 
#     'start_index': 0
#   })
# pprint(splits[-1])


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
    collection_name="example_archive_with_bge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
ids = vector_store.add_documents(splits)
print(len(ids)) # 10
print(ids)