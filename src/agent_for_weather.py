import os

from dotenv import load_dotenv
from smolagents import CodeAgent, TransformersModel, WebSearchTool

load_dotenv()

model = TransformersModel(model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct")
agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
)

result = agent.run("What is the current weather in Lyon, France?")
print(result)