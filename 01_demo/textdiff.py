from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# 注意需要安装 langchain-chroma：pip install langchain-chroma
from langchain_chroma import Chroma

# 加载数据
loader = TextLoader('./SteveJobsSpeech.txt')
documents = loader.load()

# 分块数据
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 数据嵌入向量化
vectorStore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 创建检索器
retriever = vectorStore.as_retriever()
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
keep the answer concise.
Question: 
-------
{question}
------- 

Context: 
-------
{context} 
-------
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser()
)
