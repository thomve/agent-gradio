import os
import requests

from dotenv import load_dotenv
from smolagents import CodeAgent, TransformersModel, WebSearchTool, tool, InferenceClientModel

load_dotenv()

@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get the current weather at the given location using the WeatherStack API.

    Args:
        location: The location (city name).
        celsius: Whether to return the temperature in Celsius (default is False, which returns Fahrenheit).

    Returns:
        A string describing the current weather at the location.
    """
    units = "m" if celsius else "f"
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={location}&units={units}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()

        if data.get("error"):  # Check if there's an error in the response
            return f"Error: {data['error'].get('info', 'Unable to fetch weather data.')}"

        weather = data["current"]["weather_descriptions"][0]
        temp = data["current"]["temperature"]
        temp_unit = "°C" if celsius else "°F"

        return f"The current weather in {location} is {weather} with a temperature of {temp} {temp_unit}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

# model = TransformersModel(model_id="HuggingFaceTB/SmolLM3-3B")
model = InferenceClientModel()

agent = CodeAgent(
    tools=[get_weather],
    model=model,
    stream_outputs=True
)

result = agent.run("What is the current weather in Geneva, Switzerland? (Temperature given in Fahrenheit)")
print(result)