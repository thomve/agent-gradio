from smolagents import load_tool, CodeAgent, InferenceClientModel, WebSearchTool
from dotenv import load_dotenv

load_dotenv()

image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = WebSearchTool()

agent = CodeAgent(
    tools=[search_tool, image_generation_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-0.5B-Instruct"),
    planning_interval=3
)

result = agent.run(
    "What is the capital of France? Generate an image of it.",
)