
# Import Nodes
from nodes.gatekeeper_node import gatekeeper_node
from nodes.reasearch_question_node import research_question_node
from nodes.data_understanding_node import data_understanding_node
from nodes.data_preprocessing_node import data_preprocessing_node
from nodes.modelling_node import modelling_node
from nodes.evaluation_nodes.evaluation_metrics_foundational import evaluation_metrics_foundational_node
from nodes.evaluation_nodes.evaluation_metrics_specialised import evaluation_metrics_specialised_node
from nodes.evaluation_nodes.evaluation_theoretical_orientation import evaluation_theoretical_orientation_node
from nodes.evaluation_nodes.evaluation_interpretability import evaluation_interpretability_node
from nodes.evaluation_nodes.evaluation_ethical_social import evaluation_ethical_social_node

# Import Graph related utilities
from utils.dsrp_state import DSRPState
from langgraph.graph import StateGraph, END

# Define function
def route_after_gatekeeper(state: DSRPState):
    if state["gatekeeper"]["final_classification"] == "Exclude":
        return END
    return "research_question"



# Define the sequential graph

builder = StateGraph(DSRPState)

builder.add_node("gatekeeper", gatekeeper_node)
builder.add_node("research_question", research_question_node)
builder.add_node("data_understanding", data_understanding_node)
builder.add_node("data_preprocessing", data_preprocessing_node)
builder.add_node("modelling", modelling_node)
builder.add_node("evaluation_metrics_foundational_node", evaluation_metrics_foundational_node)
builder.add_node("evaluation_metrics_specialised_node", evaluation_metrics_specialised_node)
builder.add_node("evaluation_theoretical_orientation_node", evaluation_theoretical_orientation_node)
builder.add_node("evaluation_interpretability_node", evaluation_interpretability_node)
builder.add_node("evaluation_ethical_social_node", evaluation_ethical_social_node)


builder.set_entry_point("gatekeeper")

# Conditional routing after gatekeeper
builder.add_conditional_edges(
    "gatekeeper",
    route_after_gatekeeper
)

# Sequential edges
builder.add_edge("research_question", "data_understanding")
builder.add_edge("data_understanding", "data_preprocessing")
builder.add_edge("data_preprocessing", "modelling")
builder.add_edge("modelling", "evaluation_metrics_foundational_node")
builder.add_edge("evaluation_metrics_foundational_node", "evaluation_metrics_specialised_node")
builder.add_edge("evaluation_metrics_specialised_node", "evaluation_theoretical_orientation_node")
builder.add_edge("evaluation_theoretical_orientation_node", "evaluation_interpretability_node")
builder.add_edge("evaluation_interpretability_node", "evaluation_ethical_social_node")
builder.add_edge("evaluation_ethical_social_node", END)



graph = builder.compile()


