from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

doc_list_1 = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
]

bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

doc_list_2 = [
    "I like apples",
    "You like apples",
    "You like oranges",
]

vectorstore = Chroma.from_texts(
    doc_list_2, OpenAIEmbeddings(), metadatas=[{"source": 2}] * len(doc_list_2)
)

chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

docs = ensemble_retriever.get_relevant_documents("apples")
print(docs)
