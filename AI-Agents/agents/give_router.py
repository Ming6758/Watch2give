from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
import logging

logger = logging.getLogger("GiveRouterAgent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Define State Schema
class RouterAgentState(TypedDict):
    tokens: int
    vendor_id: str
    status: Optional[str]


# RouterAgent Class
class RouterAgent:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RouterAgentState)

        builder.add_node("pass_state", self.pass_state)
        builder.add_node("trigger_transfer", self.trigger_token_transfer)

        builder.add_conditional_edges(
            "pass_state",
            self.route_decision,
            {
                "trigger_transfer": "trigger_transfer",
                "__end__": END
            }
        )

        builder.add_edge("trigger_transfer", END)
        builder.set_entry_point("pass_state")

        return builder.compile()

    def pass_state(self, state: RouterAgentState) -> RouterAgentState:
        logger.info(f"[RouterAgent] - Passing state forward: {state}")
        return state

    def route_decision(self, state: RouterAgentState) -> Literal["trigger_transfer", "__end__"]:
        logger.info(f"[RouterAgent] - Deciding next action based on tokens: {state['tokens']}")
        if state["tokens"] >= self.threshold:
            logger.info("[RouterAgent] - Threshold met. Routing to trigger_transfer.")
            return "trigger_transfer"
        logger.info("[RouterAgent] - Threshold not met. Ending flow.")
        return "__end__"

    def trigger_token_transfer(self, state: RouterAgentState) -> RouterAgentState:
        vendor_id = state.get("vendor_id")
        tokens = state.get("tokens", 0)

        try:
            logger.info(f"[RouterAgent] Initiating token transfer: {tokens} tokens → vendor {vendor_id}")

            # Placeholder for real logic (e.g., smart contract call, API)
            # result = transfer_to_vendor_onchain(vendor_id, tokens)

            logger.info(f"[RouterAgent] Transfer successful.")
            return {**state, "status": "transferred"}

        except Exception as e:
            logger.error(f"[RouterAgent] Transfer failed for vendor {vendor_id} with error: {e}")
            return {**state, "status": "transfer_failed", "error_message": str(e)}


    def run(self, input_state: RouterAgentState) -> RouterAgentState:
        assert "tokens" in input_state, "Missing 'tokens'"
        assert "vendor_id" in input_state, "Missing 'vendor_id'"
        logger.info("[1/4] Running GiveRouterAgent...")
        
        result = self.graph.invoke(input_state)

        if "status" not in result:
            logger.info("[RouterAgent] - No transfer triggered, marking as 'not_transferred'")
            result["status"] = "not_transferred"

        return self.graph.invoke(input_state)

# Test Run
if __name__ == "__main__":
    agent = RouterAgent()
    test_input = {
        "tokens": 30,
        "vendor_id": "vendor_456"
    }
    result = agent.run(test_input)
    print(f"[RouterAgent] - Final Result: {result}")
