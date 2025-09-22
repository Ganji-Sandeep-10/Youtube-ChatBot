# YouTube ChatBot 🎥💬

A Retrieval-Augmented Generation (RAG) chatbot that allows you to **query YouTube video transcripts** conversationally. The project uses the YouTube Transcript API to fetch transcripts, processes them into vector embeddings with LangChain + Google Generative AI, stores them in ChromaDB, and enables interactive Q\&A over video content.

---

## 🚀 Features

* Fetch transcripts directly from YouTube videos
* Split long transcripts into manageable text chunks
* Create vector embeddings using **Google Generative AI embeddings**
* Store & index embeddings in **ChromaDB**
* Query with natural language questions
* Retrieve context-aware answers from transcripts

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
Youtube_ChatBot.ipynb   # Main notebook with full pipeline
```

The notebook is divided into steps:

1. **Transcript Fetching** – Download captions from a YouTube video.
2. **Text Splitting** – Break transcript into chunks for embeddings.
3. **Embedding Generation** – Convert text into vector embeddings.
4. **Vector Store** – Save embeddings to Chroma for retrieval.
5. **Retriever & QA Chain** – Query and get answers using RAG.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Youtube-ChatBot.git
cd Youtube-ChatBot
```

### 2. Install dependencies

Run inside your Jupyter Notebook / Colab:

```bash
pip install youtube-transcript-api langchain langchain-google-genai langchain-chroma
```

### 3. Set up Google API key

This project uses **Google Generative AI** embeddings. You need an API key:

```python
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

### 4. Run the notebook

Open `Youtube_ChatBot.ipynb` in Jupyter Notebook or Google Colab and execute the cells step by step.

---

## 📖 Usage Example

1. **Fetch transcript:**

```python
from youtube_transcript_api import YouTubeTranscriptApi
transcript = YouTubeTranscriptApi.get_transcript("VIDEO_ID")
```

2. **Split text into chunks:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([result])
```

3. **Create embeddings & store in Chroma:**

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(texts, embeddings)
```

4. **Query with retriever:**

```python
retriever = vectorstore.as_retriever()
query = "What is the main topic of the video?"
retriever.get_relevant_documents(query)
```

5. **Ask chatbot-style questions**

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

* Only works for videos with transcripts available
* Transcript fetching depends on YouTube’s auto-captions (quality may vary)
* Default setup retrieves **English** transcripts
* Requires Google API key for embeddings & LLM

---

## 🔮 Future Improvements

* Add support for **multi-language transcripts**
* Build a **web interface** (Streamlit / FastAPI)
* Integrate **memory** for longer conversations
* Enhance summarization & topic extraction

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

