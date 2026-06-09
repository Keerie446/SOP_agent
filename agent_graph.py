from typing import Annotated, Sequence, TypedDict, Literal
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent_tools import TOOLS

# Define the state for the Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_sop: str  # The active SOP being executed
    plan: list[str]   # List of steps to execute
    current_step: int # Index of the current step
    status: str

# Instantiate the LLM
# Assuming the user's environment has GEMINI 3.1 Pro (gemini-1.5-pro or similar valid name)
# Fallback to gemini-1.5-pro as it's the standard high-end Google model accessible via langchain
llm = ChatOllama(model="llama3.1", temperature=0)

# Bind tools to LLM for the Executor node
llm_with_tools = llm.bind_tools(TOOLS)

def planner_node(state: AgentState):
    """Plans the steps needed to accomplish the user's goal based on SOPs."""
    messages = state["messages"]
    
    # Simple prompt for the planner
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SOP Planning Assistant. Based on the conversation history, determine if a new plan is needed. If the user asks to execute a procedure, write down a high-level plan as a numbered list. Reply only with the plan. If no new plan is needed, return 'NO_NEW_PLAN'."),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    # We could parse this better, but for simplicity, we just inject it into state
    content = response.content.strip()
    
    new_plan = state.get("plan", [])
    if content != "NO_NEW_PLAN" and "1." in content:
        # crude parser for numbered lists
        lines = [line.strip() for line in content.split('\\n') if line.strip().startswith(tuple(str(i) for i in range(1, 10)))]
        new_plan = lines
        return {"plan": new_plan, "current_step": 0, "messages": [AIMessage(content=f"Generated Plan:\\n{content}")]}
    
    return {}

def executor_node(state: AgentState):
    """Executes the current step using available tools."""
    messages = state["messages"]
    plan = state.get("plan", [])
    current_step_idx = state.get("current_step", 0)
    
    context = ""
    if plan and current_step_idx < len(plan):
        context = f"\\nCurrent active plan step ({current_step_idx+1}/{len(plan)}): {plan[current_step_idx]}"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SOP Execution Assistant. You have tools available to search SOPs, make HTTP requests, interact with a mock browser, and verify data. Address the user's request. Always retrieve the specific SOP rules if asked to execute a procedure.{context}"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # Format the prompt with the context
    formatted_prompt = prompt.invoke({"messages": messages, "context": context})
    response = llm_with_tools.invoke(formatted_prompt)
    
    return {"messages": [response]}

def verifier_node(state: AgentState):
    """Checks if the current step is completed."""
    # This is a simplified verifier. In a real system, it would analyze tool outputs to determine progress.
    messages = state["messages"]
    if not messages:
        return {}

    # If the executor answered the user or successfully finished tool calls without needing more tools:
    if len(state.get("plan", [])) > 0 and state.get("current_step", 0) < len(state.get("plan", [])):
        # Naive increment for demo purposes when a tool finishes successfully
        # Real logic would check if step goal is met
        pass

    return {}

# Define logic to route between nodes
def should_continue(state: AgentState) -> Literal["tools", "verifier"]:
    messages = state["messages"]
    if not messages:
        return "verifier"
        
    last_message = messages[-1]
    # If there is a tool call, route to tools
    if getattr(last_message, "tool_calls", None):
        return "tools"
    # Otherwise, go to verifier
    return "verifier"

def verifier_router(state: AgentState) -> Literal["executor", "__end__"]:
    # If plan is active and not fully executed, loop back to executor or end
    # For this simplified version, we'll just end after the verifier
    return "__end__"

# Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
tool_node = ToolNode(TOOLS)
workflow.add_node("tools", tool_node)
workflow.add_node("verifier", verifier_node)

# Add edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue, {"tools": "tools", "verifier": "verifier"})
workflow.add_edge("tools", "executor")  # loop back to evaluate tool output
workflow.add_conditional_edges("verifier", verifier_router, {"executor": "executor", "__end__": END})

# Compile
app = workflow.compile()
