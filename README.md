# YouTube ChatBot 🤖

## 🔎 About the Project

YouTube is one of the biggest sources of information, tutorials, and entertainment, but many videos are **long, hard to navigate, or packed with information** that viewers may not have time to watch fully. While YouTube provides transcripts, they can be overwhelming to read and don’t allow for natural, conversational exploration.

The **YouTube ChatBot** solves this by turning any video with captions into an **interactive, AI-powered assistant**. Instead of watching hours of video or scrolling through transcripts, you can simply ask questions like:

* *“Summarize this video in a few sentences.”*
* *“What are the key points explained?”*
* *“At what point does the video explain X?”*

It works by combining **YouTube transcripts** with **Retrieval-Augmented Generation (RAG)**, where a Large Language Model (LLM) answers questions based on the transcript content, ensuring responses are grounded in the actual video. This makes video content more **accessible, searchable, and efficient** to consume.

### 🎯 Use Cases

* **Students** – Quickly summarize lectures and tutorials.
* **Researchers** – Extract key insights from talks, interviews, or conferences.
* **Content creators** – Repurpose or analyze their own video content.
* **General users** – Save time by asking questions instead of watching full videos.

---

## 📚 How It Works

1. **Transcript Fetching**

   * Uses `youtube-transcript-api` to download the transcript of a given YouTube video (if available).

2. **Text Processing**

   * Splits transcripts into overlapping chunks using LangChain’s `RecursiveCharacterTextSplitter`. This ensures chunks are large enough for context but small enough for embedding.

3. **Embedding Generation**

   * Each chunk is converted into high-dimensional vector embeddings using **Google Generative AI embeddings**.

4. **Vector Database (ChromaDB)**

   * Embeddings are stored in **Chroma**, which allows fast similarity searches to retrieve relevant transcript parts when a user asks a question.

5. **Retriever + LLM**

   * A retriever pulls the most relevant transcript sections.
   * An LLM (e.g., Google Gemini via LangChain) uses those sections to answer user questions accurately, without hallucination.

---

## 🚀 Features

* ✅ Query any YouTube video with available captions
* ✅ Get **summaries, explanations, or answers** directly from the transcript
* ✅ Uses **RAG** to combine retrieval with LLM reasoning
* ✅ Scalable to long videos with transcript chunking
* ✅ Modular and easy to extend (swap out embeddings, LLMs, or databases)

---

## 🛠️ Tech Stack

* **Python** (>=3.9)
* [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
* [LangChain](https://python.langchain.com/)
* [ChromaDB](https://docs.trychroma.com/)
* [Google Generative AI](https://ai.google.dev/)
* Jupyter Notebook / Google Colab

---

## 📂 Project Structure

```
Youtube_ChatBot.ipynb   # Main notebook with full pipeline implementation
```

Inside the notebook:

* **Step 1a**: Transcript fetching
* **Step 1b**: Text splitting
* **Step 2**: Embedding generation
* **Step 3**: Store & index in ChromaDB
* **Step 4**: Create retriever
* **Step 5**: Question-answering with RetrievalQA chain

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Youtube-ChatBot.git
cd Youtube-ChatBot
```

### 2. Install dependencies

```bash
pip install youtube-transcript-api langchain langchain-google-genai langchain-chroma
```

### 3. Set up Google API key

```python
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

### 4. Run the notebook

Open `Youtube_ChatBot.ipynb` in Jupyter Notebook or Google Colab and execute step by step.

---

## 📖 Example Workflow

1. **Fetch transcript**

```python
from youtube_transcript_api import YouTubeTranscriptApi
transcript = YouTubeTranscriptApi.get_transcript("VIDEO_ID")
```

2. **Split text**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([result])
```

3. **Generate embeddings & store**

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(texts, embeddings)
```

4. **Query with retriever**

```python
retriever = vectorstore.as_retriever()
retriever.get_relevant_documents("What is the video about?")
```

5. **Conversational QA**

```python
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=GoogleGenerativeAI(model="gemini-pro"),
    retriever=retriever,
)
response = qa.run("Summarize the key points in 3 sentences.")
print(response)
```

---

## ⚠️ Limitations

* Works only if transcripts are available for the video
* Transcript quality depends on YouTube captions
* Defaults to English (can be extended to other languages)
* Requires Google API key for embeddings & LLM

---

## 🔮 Future Enhancements

* Add **multi-language support**
* Develop a **web UI** with Streamlit/FastAPI
* Integrate **chat memory** for multi-turn conversations
* Support for **multiple video sources** (batch mode)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Open a PR or start a discussion in the Issues tab.

