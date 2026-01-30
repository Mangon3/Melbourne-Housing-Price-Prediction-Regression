from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings
from src.graph.state import AgentState
from src.graph.nodes import call_model
from src.tools.registry import tools
def create_workflow(llm):
    """
    Creates and compiles the StateGraph with the provided LLM instance.
    """

    model_with_tools = llm.bind_tools(tools)

    def agent_node(state):
        return call_model(state, model_with_tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")

    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
