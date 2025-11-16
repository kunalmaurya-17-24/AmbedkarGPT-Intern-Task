---

# **AmbedkarGPT – Intern Task**

A simple **RAG-based command-line Q&A system** built as part of the **Kalpit AI Intern Assignment**.
The system loads a short speech by Dr. B. R. Ambedkar, creates embeddings, stores them in a local ChromaDB, retrieves relevant chunks, and answers questions using **Ollama – Mistral 7B**.

---

## 📌 **Features**

* Load local text file (`speech.txt`)
* Split text into chunks using **RecursiveCharacterTextSplitter**
* Generate embeddings using
  `sentence-transformers/all-MiniLM-L6-v2`
* Store vectors locally using **ChromaDB**
* Retrieve relevant chunks using similarity search
* Generate answers using **Ollama (Mistral 7B)**
* Interactive **command-line interface (CLI)**

---

## 📂 **Project Structure**

```
AmbedkarGPT-Intern-Task/
│── main.py
│── speech.txt
│── requirements.txt
│── README.md
└── chroma_db/   (auto-created at runtime)
```

---

## 🛠 **Setup Instructions**

### **1. Install Python (3.10 or 3.11)**

LangChain compatibility requires these versions.

---

### **2. Create & Activate a Virtual Environment**

```
python -m venv venv
venv\Scripts\activate
```

---

### **3. Install Dependencies**

```
pip install -r requirements.txt
```

---

### **4. Install Ollama**

Download from:
[https://ollama.com/download](https://ollama.com/download)

Then pull the Mistral model:

```
ollama pull mistral
```

Test it:

```
ollama run mistral
```

---

## ▶️ **Run the Program**

```
python main.py
```

You will see:

```
--- AmbedkarGPT CLI ---
Ask a question (or type 'exit'):
```

Example:

**Q:** *What is the real enemy?*
**A:** The belief in the shastras.

---

## 📘 **How It Works (RAG Pipeline)**

### **1. Load Speech**

`speech.txt` is loaded using `TextLoader`.

### **2. Split Text**

Text is chunked using `RecursiveCharacterTextSplitter`.

### **3. Embeddings**

Chunks are embedded with
`sentence-transformers/all-MiniLM-L6-v2`.

### **4. Store in ChromaDB**

Embeddings + metadata are saved in a local `chroma_db/` folder.

### **5. Retrieve**

Chroma retriever returns relevant chunks based on similarity search.

### **6. LLM Answer Generation**

**Mistral 7B** via Ollama generates the final response using retrieved context.

---

## 📄 **Requirements**

See `requirements.txt` for the full list of dependencies.

---

## ✔️ **Status**

Fully working prototype delivered as required under **Phase 1 – Core Skills Evaluation**.

---
