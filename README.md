# Autonomous Research Copilot 🤖🔍

An advanced multi-agent research platform built using **LangGraph**, **LangChain**, and **OpenAI GPT-4**, featuring intelligent orchestration for iterative document retrieval and automated synthesis using a **FAISS Vector Database**.

## 🚀 Features
- **Multi-Agent Architecture**: Utilizes LangGraph to manage states across distinct specialized agents (Planner, Researcher, Writer).
- **Intelligent Retrieval**: Uses semantic chunking and dense vector search via FAISS to retrieve contextually relevant knowledge.
- **Automated Synthesis**: Automatically handles long context compilation to output comprehensive, structured markdown reports.

## 🛠️ Tech Stack
- **Frameworks**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4o-mini
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Language**: Python

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd research-copilot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables:**
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```
