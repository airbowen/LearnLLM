from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


texts = [
    "我家的狗全身大部分都是黑色的，但尾巴是白色的，特别喜欢出去玩",
    "我家狗狗身上黑乎乎的，就尾巴那一块儿白，它老爱往外跑",
    "我家的猫全身黑色，也很喜欢出门",
    "夏天特别适合游泳"
]

# input the openai api key 

vector_store = Chroma.from_texts(texts, embedding=OpenAIEmbeddings())

question = "简单描述下我家的狗"

retriever = vector_store.as_retriever(search_type ="similarity", search_kwargs={"k": 3})
print(retriever.get_relevant_documents(question))
