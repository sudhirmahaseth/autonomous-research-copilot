import os
from dotenv import load_dotenv
from vector_store import VectorStoreManager
from agents import create_research_graph

load_dotenv()

def main():
    # 1. Mock Data Source (Real-world mein aap PDF/Web data load karenge)
    mock_documents = [
        "Generative AI and Large Language Models are transforming DevOps by automating infrastructure code generation.",
        "LangGraph enables developers to build complex multi-agent workflows with state management and cyclic graphs.",
        "FAISS is an efficient library for dense vector clustering and similarity search, heavily used in modern RAG pipelines.",
        "LLMOps focuses on operationalizing LLMs, tracking latency, token usage, and mitigating hallucinations in production."
    ]
    
    print("⏳ Initializing FAISS Vector Store...")
    vector_manager = VectorStoreManager()
    vector_manager.initialize_with_texts(mock_documents)
    
    # 2. Compile LangGraph
    print("⚙️ Compiling LangGraph Multi-Agent Workflow...")
    graph = create_research_graph(vector_manager)
    
    # 3. Execute Graph
    topic = "How Multi-Agent workflows and FAISS help in LLM applications"
    print(f"🚀 Running Research Copilot for Topic: '{topic}'...\n")
    
    initial_state = {
        "research_topic": topic,
        "plan": "",
        "retrieved_context": "",
        "final_report": "",
        "revision_count": 0
    }
    
    result = graph.invoke(initial_state)
    
    print("================ FINAL GENERATED REPORT ================")
    print(result["final_report"])
    print("========================================================")

if __name__ == "__main__":
    main()
