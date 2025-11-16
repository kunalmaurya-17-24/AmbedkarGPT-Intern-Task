from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate



# Load document

loader = TextLoader("speech.txt")
docs = loader.load()

# print("Loaded Document :", len(docs))
# print("Sample :", docs[0].page_content[:200])


# Split text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)


#hugging face embedders
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={}
)


#chroma store db
vectordb = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="chroma_db"
)

print("Vector DB created successfully")

retriever = vectordb.as_retriever(search_kwargs={"k": 2})


# LAMA LAMA

llm = Ollama(model="mistral")


#pipeline here
prompt = ChatPromptTemplate.from_template("""
You are AmbedkarGPT. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {input}

Answer:
""")

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)


#cli now
print("\n--- AmbedkarGPT CLI ---")

while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() in ["exit", "quit"]:
        break

    response = rag_chain.invoke({"input": question})
    print("\nAnswer:", response["answer"])
