from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import logging

logger = logging.getLogger("RewardAgent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RewardAgentState(TypedDict):
    viewer_id: str
    verified_gives: int
    reward_type: Optional[str]
    reward_gives: int
    reward_status: Optional[str]


class RewardAgent:
    def __init__(self, reward_thresholds=None):
        self.reward_thresholds = reward_thresholds or {
            10: "v-bucks",
            20: "robux",
            50: "mystery-nft"
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RewardAgentState)

        builder.add_node("check_reward", self.check_eligibility)
        builder.add_node("assign_reward", self.assign_reward)
        builder.add_node("dispatch_reward", self.dispatch_reward)

        builder.add_edge("check_reward", "assign_reward")

        builder.add_conditional_edges(
            "assign_reward",
            self.should_dispatch,
            {
                "dispatch_reward": "dispatch_reward",
                "__end__": END
            }
        )

        builder.add_edge("dispatch_reward", END)
        builder.set_entry_point("check_reward")
        return builder.compile()

    def check_eligibility(self, state: RewardAgentState) -> RewardAgentState:
        logger.info(f"[RewardAgent] - Checking eligibility for viewer '{state['viewer_id']}' with {state['verified_gives']} gives")
        return state

    def reward_router(self, state: RewardAgentState) -> str:
        gives = state["verified_gives"]
        for threshold in sorted(self.reward_thresholds.keys(), reverse=True):
            if gives >= threshold:
                state["reward_type"] = self.reward_thresholds[threshold]
                return "dispatch_reward"
        return "__end__"

    def should_dispatch(self, state: RewardAgentState) -> str:
        return "dispatch_reward" if state.get("reward_type") else "__end__"

    def dispatch_reward(self, state: RewardAgentState) -> RewardAgentState:
        logger.info(f"[RewardAgent] - Dispatching '{state['reward_type']}' to viewer '{state['viewer_id']}'")
        # Simulate API call or smart contract interaction here
        state["reward_status"] = "delivered"
        return state

    def assign_reward(self, state: RewardAgentState) -> RewardAgentState:
        gives = state["verified_gives"]
        for threshold in sorted(self.reward_thresholds.keys(), reverse=True):
            if gives >= threshold:
                reward = self.reward_thresholds[threshold]
                logger.info(f"[RewardAgent] - Assigning '{reward}' for {threshold} gives")
                return {**state, "reward_type": reward, "reward_gives": threshold}
        logger.warning("[RewardAgent] - No reward eligible")
        return {**state, "reward_type": None}

    def run(self, input_state: RewardAgentState) -> RewardAgentState:
        logger.info("[4/4] Running RewardAgent...")
        return self.graph.invoke(input_state)


# Example usage
if __name__ == "__main__":
    agent = RewardAgent()
    test_input = {
        "viewer_id": "user_123",
        "verified_gives": 60  # Should unlock mystery-nft
    }
    result = agent.run(test_input)
    print(f"[RewardAgent] - Final Result: {result}")
