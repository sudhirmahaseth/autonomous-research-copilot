from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

# 1. State Definition
class AgentState(TypedDict):
    research_topic: str
    plan: str
    retrieved_context: str
    final_report: str
    revision_count: int

# LLM Initialize
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 2. Agent Nodes
def planner_node(state: AgentState):
    """Topic ke liye ek research plan banata hai"""
    topic = state["research_topic"]
    prompt = f"Create a concise research plan or sub-queries to gather info about: {topic}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"plan": response.content}

def researcher_node(state: AgentState, vector_manager: VectorStoreManager):
    """Plan ke basis par vector store se information nikalta hai"""
    plan = state["plan"]
    # FAISS vector store se query karke context nikalna
    context = vector_manager.retrieve_docs(query=plan, k=4)
    return {"retrieved_context": context}

def writer_node(state: AgentState):
    """Context ko analyze karke final structured report likhta hai"""
    topic = state["research_topic"]
    context = state["retrieved_context"]
    prompt = f"Based on the following context, write a professional research report on '{topic}':\n\nContext:\n{context}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}

# 3. Graph Assembly
def create_research_graph(vector_manager):
    workflow = StateGraph(AgentState)
    
    # Nodes add karein (Lambda use karke vector_manager pass kiya)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", lambda state: researcher_node(state, vector_manager))
    workflow.add_node("writer", writer_node)
    
    # Edges define karein (Sequential Flow)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()
