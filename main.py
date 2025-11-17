from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



PERSIST_DIR = "chroma_db"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    hf_embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
    
    if os.path.exists(PERSIST_DIR) and os.path.isdir(PERSIST_DIR):
        print("Loading existing vectorDB")
        vectorstore = Chroma(
            collection_name="collection_db",
            embedding_function=hf_embeddings,
            persist_directory=PERSIST_DIR
        )
    else:
        print("Creating vector store")

        loader = TextLoader('speech.txt')
        documents = loader.load()
        

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=200,
            chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(documents)
        

        vectorstore = Chroma(
            collection_name="collection_db",
            embedding_function=hf_embeddings,
            persist_directory=PERSIST_DIR
        )
        

        texts = [doc.page_content for doc in split_docs]
        vectorstore.add_texts(texts=texts)
    

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    

    prompt = ChatPromptTemplate.from_messages([
        """
You are AmbedkarGPT. Answer the question based ONLY on the provided context.
If the answer is not in the context, reply:
"I cannot answer this from the given text."

Context:
{context}

Question: {question}

Answer:  
"""
    ])
    

    llm = ChatOllama(model="mistral", stream=True)
    

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    
    print("\n" + "="*50)
    print("AmbedkarGPT - Q and Ans tool")
    print("="*50)
    print("Type 'exit' to quit.\n")
    

    while True:
        user_input = input("Question: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
        
        print("\nAnswer: ", end="")
        for chunk in rag_chain.stream(user_input):
            if hasattr(chunk, 'content'):
                print(chunk.content, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
