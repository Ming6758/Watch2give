from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import logging
logger = logging.getLogger("VaultDecider")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Simulated vault data (could come from on-chain query in real deployment)
VENDOR_VAULTS = {
    "vendor_123": {"apy": 9.0},
    "vendor_456": {"apy": 12.5},
    "vendor_789": {"apy": 6.7},
}

class VaultDeciderState(TypedDict):
    tokens: int
    vendor_id: str
    vendor_apy: float
    action: Optional[str]
    selected_vault: Optional[str]

class VaultDeciderAgent:
    def __init__(self, stake_threshold=10, redeem_threshold=3, apy_threshold=10):
        self.stake_threshold = stake_threshold
        self.redeem_threshold = redeem_threshold
        self.apy_threshold = apy_threshold
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(VaultDeciderState)

        graph.add_node("check_balance", self.check_balance)
        graph.add_node("fetch_vendor_apy", self.fetch_vendor_apy)
        graph.add_node("stake_tokens", self.stake)
        graph.add_node("redeem_tokens", self.redeem)

        graph.add_edge("check_balance", "fetch_vendor_apy")

        graph.add_conditional_edges(
            "fetch_vendor_apy",
            self.route_decision,  # <- now this only returns a string
            {
                "stake": "stake_tokens",
                "redeem": "redeem_tokens",
                "__end__": END
            }
        )

        graph.add_edge("stake_tokens", END)
        graph.add_edge("redeem_tokens", END)
        graph.set_entry_point("check_balance")

        return graph.compile()

    def check_balance(self, state: VaultDeciderState) -> VaultDeciderState:
        logger.info(f"[VaultDecider] Checking token balance: {state['tokens']}")
        return state

    def fetch_vendor_apy(self, state: VaultDeciderState) -> VaultDeciderState:
        apy = VENDOR_VAULTS.get(state["vendor_id"], {}).get("apy", 0)
        logger.info(f"[VaultDecider] Vendor APY: {apy}%")
        return {**state, "vendor_apy": apy}

    def route_decision(self, state: VaultDeciderState) -> str:
        apy = state.get("vendor_apy", 0)
        if not apy:
            logger.warning(f"Vendor {state['vendor_id']} not found in vault registry.")
            return "__end__"

        if state["tokens"] >= self.stake_threshold and apy >= self.apy_threshold:
            return "stake"
        elif state["tokens"] >= self.redeem_threshold:
            return "redeem"
        return "__end__"

    def stake(self, state: VaultDeciderState) -> VaultDeciderState:
        logger.info(f"[VaultDecider] Staking {state['tokens']} tokens with vendor {state['vendor_id']}.")
        state["action"] = "staked"
        state["selected_vault"] = state["vendor_id"]
        return state

    def redeem(self, state: VaultDeciderState) -> VaultDeciderState:
        logger.info(f"[VaultDecider] Redeeming {state['tokens']} tokens with vendor - '{state['vendor_id']}'.")
        state["action"] = "redeemed"
        return state

    def run(self, input_state: VaultDeciderState) -> VaultDeciderState:
        logger.info("[3/4] Running VaultDeciderAgent...")
        return self.graph.invoke(input_state)

# Example usage
def test():
    agent = VaultDeciderAgent()
    result = agent.run(
        {
        "tokens": 20,
        "vendor_id": "vendor_456"
        }
    )
    print("[VaultDecider] Final result:", result)


if __name__ == "__main__":
    test()