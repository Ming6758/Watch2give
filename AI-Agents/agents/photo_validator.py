from langchain_core.tools import tool
from groq import Groq
import base64
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import re
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_groq import ChatGroq
import logging

logger = logging.getLogger("PhotoValidatorAgent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Hardcoded API key (for testing only - remove before committing to version control)
GROQ_API_KEY = "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"

# Correct model names
VISION_MODEL = "llama3-70b-8192"  # For image description
TEXT_MODEL = "llama3-8b-8192"     # For text processing

@tool
def validate_donation_photo(photo_path: str) -> str:
    """
    Uses the Groq vision model to extract a description of the donation/giving photo.
    This tool does NOT determine if it's valid or not — it just returns the description.
    """
    try:
        with open(photo_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded}"
    except FileNotFoundError:
        logger.error(f"Photo not found: {photo_path}")
        return "Image file error"
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return "Image processing error"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what's happening in this image in detail."},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            temperature=0.4,
            max_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return "Failed to analyze image"

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    score: float
    validation_result: bool

class PhotoValidatorAgent:
    def __init__(self, model, tools, system_prompt: str = "", threshold: float = 0.75):
        self.system = system_prompt or """You are a donation validator. Analyze the image description and:
        1. Check if it shows a clear donation/giving act
        2. Assign a score (0-1) based on clarity
        3. Return "Score: X" and a brief explanation"""
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.threshold = threshold

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile()

    # ... [rest of the PhotoValidatorAgent methods remain the same] ...

def test(photo_path: str) -> dict:
    logger.info("[2/4] Running PhotoValidatorAgent...")
    
    try:
        # Initialize ChatGroq with correct model name
        model = ChatGroq(
            model_name=TEXT_MODEL,
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )

        photo_agent = PhotoValidatorAgent(
            model,
            tools=[validate_donation_photo],
            system_prompt="""
            Analyze the image description and determine if it shows:
            1. A clear act of donation/giving
            2. Visible participants (donor and recipient)
            3. The item being donated
            Score (0-1) based on how clearly these elements are shown.
            """
        )

        result = photo_agent.graph.invoke({
            "messages": [
                HumanMessage(content=f"Please validate this donation photo: {photo_path}")
            ],
            "score": 0.0,
            "validation_result": False
        })
        
        logger.info(f"Validation Score: {result['score']:.2f}")
        logger.info(f"Validation Result: {'VALID' if result['validation_result'] else 'INVALID'}")
        return result
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return {
            "score": 0.0,
            "validation_result": False,
            "error": str(e)
        }

if __name__ == "__main__":
    try:
        test_image = "../images/sharing.jpg"  # Update with your actual image path
        result = test(test_image)
        print("\nFinal Validation:")
        print(f"Score: {result.get('score', 0):.2f}")
        print(f"Valid: {'YES' if result.get('validation_result', False) else 'NO'}")
        if "error" in result:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Critical error: {str(e)}")