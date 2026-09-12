
# Set WD at root
import sys
import os

root_path = os.path.abspath(os.path.join('..'))
# Set WD at ROOT
os.chdir(root_path)
# Print to verify
print(f"Current working directory: {os.getcwd()}")

# Utilities
from utils.load_yaml_prompt import load_yaml_prompt
from utils.load_vector_query import load_vector_query
from utils.parse_llm_json import parse_llm_json

# Import Nodes
from nodes.gatekeeper_node import gatekeeper_node
from nodes.reasearch_question_node import research_question_node
from nodes.data_understanding_node import data_understanding_node
from nodes.data_preprocessing_node import data_preprocessing_node
from nodes.modelling_node import modelling_node
from nodes.evaluation_nodes.evaluation_metrics_foundational import evaluation_metrics_foundational_node
from nodes.evaluation_nodes.evaluation_theoretical_orientation import evaluation_theoretical_orientation_node
from nodes.evaluation_nodes.evaluation_interpretability import evaluation_interpretability_node
from nodes.evaluation_nodes.evaluation_ethical_social import evaluation_ethical_social_node

#State
from utils.dsrp_state import DSRPState

# Define utility function
def route_after_gatekeeper(state: DSRPState):
    decision = str(state.get("gatekeeper", {}).get("final_classification", "")).strip().lower()
    if decision == "include":
        return "workflow_start"
    # Stop workflow for Exclude and Borderline (or any non-Include decision).
    return END

def route_after_specialised(state: DSRPState):
    if not state["modelling"]["specialised_paradigms"]:
        return END
    return "modelling_sub_specialised"

#----------------------------------------------------------------------------------------------------
#                        The Agentic Workflow/Graph
#----------------------------------------------------------------------------------------------------

# parallel graph with gatekeeper-controlled entry
from langgraph.graph import StateGraph, END

builder = StateGraph(DSRPState)

builder.add_node("gatekeeper", gatekeeper_node)
builder.add_node("workflow_start", lambda state: {})
builder.add_node("workflow_complete", lambda state: {})
builder.add_node("research_question", research_question_node)
builder.add_node("data_understanding", data_understanding_node)
builder.add_node("data_preprocessing", data_preprocessing_node)
builder.add_node("modelling", modelling_node)
builder.add_node("evaluation_metrics_foundational_node", evaluation_metrics_foundational_node)
builder.add_node("evaluation_theoretical_orientation_node", evaluation_theoretical_orientation_node)
builder.add_node("evaluation_interpretability_node", evaluation_interpretability_node)
builder.add_node("evaluation_ethical_social_node", evaluation_ethical_social_node)

builder.set_entry_point("gatekeeper")

# Gatekeeper decides whether workflow continues.
builder.add_conditional_edges(
    "gatekeeper",
    route_after_gatekeeper
)

# Parallel fan-out starts immediately after Include.
builder.add_edge("workflow_start", "research_question")
builder.add_edge("workflow_start", "data_understanding")
builder.add_edge("workflow_start", "data_preprocessing")
builder.add_edge("workflow_start", "modelling")
builder.add_edge("workflow_start", "evaluation_theoretical_orientation_node")
builder.add_edge("workflow_start", "evaluation_interpretability_node")
builder.add_edge("workflow_start", "evaluation_ethical_social_node")

# Foundational metrics depends on modelling output.
builder.add_edge("modelling", "evaluation_metrics_foundational_node")

# Join all active branches before END.
builder.add_edge("research_question", "workflow_complete")
builder.add_edge("data_understanding", "workflow_complete")
builder.add_edge("data_preprocessing", "workflow_complete")
builder.add_edge("evaluation_metrics_foundational_node", "workflow_complete")
builder.add_edge("evaluation_theoretical_orientation_node", "workflow_complete")
builder.add_edge("evaluation_interpretability_node", "workflow_complete")
builder.add_edge("evaluation_ethical_social_node", "workflow_complete")

builder.add_edge("workflow_complete", END)

graph = builder.compile()