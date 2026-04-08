import os
import json
from ollama import Client
from dotenv import load_dotenv
import requests

load_dotenv()

print(os.environ.get("OLLAMA_API_KEY"))


client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY")},
)
messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?",
    },
]

# for part in client.chat('glm-5:cloud', messages=messages, stream=True):
#   print(part.message.content, end='', flush=True)

query1 = "What is the current price of Bitcoin? Today is 6th April 2026."
messages = [
    {"role": "user", "content": query1},
]

# for part in client.chat('glm-5:cloud', messages=messages, stream=True):
#   print(part.message.content, end='', flush=True)


# for part in client.chat('qwen3.5', messages=messages, stream=True):
#   print(part['message']['thinking'], end='')


# ============================================================
# Weather Tool
# ============================================================


def get_weather(lat: int, lon: int) -> str:
    """Get the current weather details in a given city"""
    url = "https://api.api-ninjas.com/v1/weather?lat={}&lon={}".format(lat, lon)
    api_key = os.environ.get("WEATHER_API_KEY")

    hds = {"X-Api-Key": api_key, "Content-Type": "application/json"}

    response = requests.get(url, headers=hds)
    data = response.json()
    # print(data)
    return data


# get_weather(51.5074, -0.1278)


# ============================================================
# Weather Tool Schema
# ============================================================

weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather details in a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "The Latitude of the place to get the weather for.",
                },
                "lon": {
                    "type": "number",
                    "description": "The Longitude of the place to get the weather for.",
                },
            },
            "required": ["lat", "lon"],
        },
    },
}


query2 = "What is the weather in Thane today. The date today is 6th April 2026. Its lat are19"
messages = [
    {"role": "user", "content": query2},
]

# for part in client.chat(
#     "glm-5:cloud", messages=messages, tools=[weather_tool_schema], stream=True
# ):
#     print(part.message.content, end="", flush=True)


# ============================================================
# Geo coordinates Tool
# ============================================================


def get_lat_lon(city: str) -> list[int]:
    """Get the exact latitude and longitude of a given place on earth"""
    api_key = os.environ.get("GEOCODING_API")
    url = "https://geocode.maps.co/search?q={}&api_key={}".format(city, api_key)

    response = requests.get(url)
    data = response.json()
    # print(data)
    return [data[0]["lat"], data[0]["lon"]]


# print(get_lat_lon("Thane"))

# ============================================================
# Geo coordinates Tool Schema
# ============================================================

geocoding_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_lat_lon",
        "description": "Get the exact latitude and longitude of a given place on earth",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The City whose latitude and longitude are to be found.",
                }
            },
            "required": ["city"],
        },
    },
}

# ============================================================

query3 = "What is the temperature in Thane today. The date today is 8th April 2026."
messages = [
    {"role": "user", "content": query3},
]

# for part in client.chat(
#     "glm-5:cloud",
#     messages=messages,
#     tools=[weather_tool_schema, geocoding_tool_schema],
#     stream=True,
# ):
#     print(part.message)
#     print("😭😭😭")


# ============================================================
# Agentic Framework
# ============================================================


class MyAgent:
    def __init__(
        self, client: Client, model: str, system: str = "", tools: list | None = None
    ) -> None:
        self.client = client
        self.model = model
        self.messages: list = []
        self.tools = tools if tools is not None else []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message: str = ""):
        if message:
            self.messages.append({"role": "user", "content": message})
        final_assistant_content = self.execute()
        if final_assistant_content:
            self.messages.append(
                {"role": "assistant", "content": final_assistant_content}
            )

        return final_assistant_content

    def execute(self):
        while True:
            tool_called = False

            for completion in self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                stream=True,
            ):
                response_msg = completion.message
                # print("🌸", response_msg.thinking)
                if getattr(response_msg, "thinking", None):
                    print("THINKING:", response_msg.thinking)

                if getattr(response_msg, "content", None):
                    print("CONTENT:", response_msg.content)

                if response_msg.tool_calls is not None:
                    print("✅ Tool Call needed", response_msg.tool_calls)
                    self.messages.append(response_msg)

                    tool_outputs = []
                    for tool_call in response_msg.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = tool_call.function.arguments

                        tool_output_content = f"Tool {fn_name} not found"
                        if fn_name in globals() and callable(globals()[fn_name]):
                            fn_to_call = globals()[fn_name]
                            executed_output = fn_to_call(**fn_args)
                            tool_output_content = str(executed_output)

                        print("*" * 10, tool_output_content)
                        tool_outputs.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": tool_output_content,
                            }
                        )

                    self.messages.extend(tool_outputs)
                    tool_called = True
                    break

            if tool_called:
                continue
        
            return response_msg.content
        


# ============================================================


query4 = "What is the temperature in Thane today. The date today is 8th April 2026."
# query5 = "What are the latitude and longitude of Thane?"
# messages = [
#     {"role": "user", "content": query5},
# ]

og = MyAgent(
    client=client,
    model="glm-5:cloud",
    system="You are a helpful agent. You always provide the final answer in a complete sentence with correct grammar",
    tools=[weather_tool_schema, geocoding_tool_schema],
)

print("⚠️⚠️")
final_result = og(query4)
