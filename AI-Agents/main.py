# Master LangGraph Orchestration: GiveRouter → PhotoValidator → VaultDecider → RewardAgent

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.give_router import RouterAgent
from agents.photo_validator import test
from agents.vault_decider import VaultDeciderAgent
from agents.reward_agent import RewardAgent
import json

# --- Shared State Across Agents ---
class GlobalState(TypedDict):
    tokens: int # used in give router, vault
    vendor_id: str # used in give router, vault
    vendor_apy: float # used in vault
    status: Optional[str] # used in give router
    score: float # used in photo validator
    validation_result: bool # used in photo validator
    action: Optional[str] # used in vault
    selected_vault: Optional[str] # used in vault
    viewer_id: str # used in reward agent
    verified_gives: int # used in reward agent
    reward_type: Optional[str] # used in reward agent
    reward_status: Optional[str] # used in reward agent
    photo_path: Optional[str] # used in photo validator

# --- Wrapper for GiveRouterAgent ---
def give_router_node(state: GlobalState) -> GlobalState:
    agent = RouterAgent()
    result = agent.run({
        "tokens": state["tokens"],
        "vendor_id": state["vendor_id"]
    })

    state["status"] = result.get("status", "not_transferred")  # Save full context
    state["action"] = state["status"]  # For vault logic consistency
    
    return state

# --- Wrapper for PhotoValidatorAgent ---
def photo_validator_node(state: GlobalState) -> GlobalState:
    result = test(state.get("photo_path"))
    state.update(result)  # expects proof_score, proof_valid
    return state

# --- Wrapper for VaultDeciderAgent ---
def vault_decider_node(state: GlobalState) -> GlobalState:
    agent = VaultDeciderAgent()
    result = agent.run({
        "tokens": state["tokens"],
        "vendor_id":  state["vendor_id"]
    })
    state.update(result)  # expects action, selected_vault
    return state

# --- Wrapper for RewardAgent ---
def reward_agent_node(state: GlobalState) -> GlobalState:
    agent = RewardAgent()
    result = agent.run({
        "viewer_id": state["viewer_id"],
        "verified_gives": state["verified_gives"]
    })
    state.update(result)  # reward_type, reward_status
    return state

# --- Master Graph ---
workflow = StateGraph(GlobalState)
workflow.add_node("give_router", give_router_node)
workflow.add_node("photo_validator", photo_validator_node)
workflow.add_node("vault_decider", vault_decider_node)
workflow.add_node("reward_agent", reward_agent_node)

workflow.set_entry_point("give_router")
workflow.add_edge("give_router", "photo_validator")
workflow.add_edge("photo_validator", "vault_decider")
workflow.add_edge("vault_decider", "reward_agent")
workflow.add_edge("reward_agent", END)

compiled = workflow.compile()

# --- End-to-End Test ---
if __name__ == "__main__":
    initial_state = {
        "tokens": 20,
        "vendor_id": "vendor_456",
        "viewer_id": "user_123",
        "verified_gives": 22,
        "photo_path": "./images/sharing.jpg" 
    }

    print("\n[Master Flow] Running end-to-end simulation...")

    result = compiled.invoke(initial_state)
    
    print("\n+++++++++++++++++ Master Flow Simulation Ended +++++++++++++++++\n")

    print("Saving result to result.json...")
    try:
        with open("result.json", "w") as f:
            json.dump(result, f, indent=4)
        print("\nResult saved to result.json")   
    except:
        print("\nError saving result to result.json")  