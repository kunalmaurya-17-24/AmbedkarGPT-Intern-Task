import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load documents rn
def load_speech(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: '{path}' not found.")
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()

# chroma store storeage 
def init_db(docs, embeddings, persist_dir="chroma_db"):
    if os.path.exists(persist_dir):
        print("Loading existing Chroma database...")
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print("Creating new Chroma database...")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_dir
        )
        db.persist()
    return db

# pipeline is here
def rag_pipeline(vectorstore):
    llm = ChatOllama(model="mistral",stream=True)
    prompt_template = """
You are AmbedkarGPT. Answer the question based ONLY on the provided context.
If the answer is not in the context, reply:
"I cannot answer this from the given text."

Context:
{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    pipeline = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# CLI wit steaming like gpt

def cli(rag_chain):
    print("\n--- AmbedkarGPT CLI (Streaming Enabled) ---")
    print("Type 'exit' to quit.")
    while True:
        question = input("\nAsk a question: ")
        if question.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("\nAnswer: ", end="", flush=True)

        # STREAM THE RESPONSE
        for chunk in rag_chain.stream(question):
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n" + "-" * 50)


def main():
    print("Loading speech.txt ...")
    docs = load_documents("speech.txt")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = build_vector_db(chunks, embeddings)
    pipeline = rag_pipeline(db)

    cli(rag_chain)

if __name__ == "__main__":
    main()
