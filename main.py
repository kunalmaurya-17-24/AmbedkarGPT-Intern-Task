import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# just load the file (basic check)
def load_speech(path):
    if not os.path.exists(path):
        print("file not found:", path)
        raise FileNotFoundError(path)
    return TextLoader(path, encoding="utf-8").load()


# chroma storage (load or create)
def init_db(docs, embeddings, persist_dir="chroma_db"):
    if os.path.exists(persist_dir):
        print("Using existing Chroma DB...")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print("Creating new Chroma DB...")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_dir
        )
        db.persist()
        return db


# build the RAG chain (llm + retriever)
def build_pipeline(vstore):
    llm = ChatOllama(model="mistral", stream=True)

    # keeping the prompt simple, not too LLM-formal
    template = """
You're AmbedkarGPT. Use ONLY the context.  
If it's not in the context, just say you don't know.

Context:
{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vstore.as_retriever(search_kwargs={"k": 2})

    def join_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | join_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain


# simple CLI
def cli(chain):
    print("\n--- AmbedkarGPT CLI ---")
    print("(type 'exit' to quit)\n")

    while True:
        q = input("Ask something: ").strip()
        if q.lower() in ("exit", "quit"):
            print("bye!")
            break

        print("Answer: ", end="", flush=True)

        # streaming
        for chunk in chain.stream(q):
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n" + "-" * 40)


def main():
    print("Loading speech.txt...")
    docs = load_speech("speech.txt")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    parts = splitter.split_documents(docs)
    print("chunks:", len(parts))  # small debug print, very human-like

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = init_db(parts, emb)

    chain = build_pipeline(db)

    cli(chain)


if __name__ == "__main__":
    main()
