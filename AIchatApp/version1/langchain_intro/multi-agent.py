from typing import TypedDict
class AgentState(TypedDict):
	current_result: str
 
def AddHello(state):
    return {"current_result": "hello, " + state["current_result"]}

def AddGoodBye(state):
    return {"current_result": state["current_result"] + ",  now good bye!"}

from langgraph.graph import StateGraph, END
workflow = StateGraph(AgentState)

workflow.add_node("hello_node", AddHello)
workflow.add_node("goodbye_node", AddGoodBye)

def router(state):
    if "no run goodbye_node" in state["current_result"]:
        return END
    else:
        return "goodbye_node"

workflow.add_conditional_edges("hello_node", router)
workflow.add_edge("goodbye_node", END)
workflow.set_entry_point("hello_node")
graph = workflow.compile()

print(graph.invoke({"current_result": "test"}))
# output
# {'current_result': 'hello, test,  now good bye!'}
