import os
from ollama import Client
from dotenv import load_dotenv
import requests
import httpx

load_dotenv()

# print(os.environ.get("OLLAMA_API_KEY"))

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


# ============================================================
# Weather Tool
# ============================================================


def get_weather(lat: int, lon: int) -> str:
    """Get the current weather details in a given city"""
    url = "https://api.api-ninjas.com/v1/weather?lat={}x&lon={}".format(lat, lon)
    api_key = os.environ.get("WEATHER_API_KEY")

    try:
        hds = {"X-Api-Key": api_key, "Content-Type": "application/json"}
        response = requests.get(url, headers=hds)
        return {"success": True, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# get_weather(51.5074, -0.1278)

# ============================================================
# Weather Tool 2
# ============================================================


def get_weather2(lat: int, lon: int) -> str:
    """Get the current weather details in a given city"""
    url = "https://api.ambeedata.com/weather/latest/by-lat-lng?lat={}&lng={}".format(lat, lon)
    api_key = os.environ.get("AMBEE_API_KEY")

    try:
        hds = {"X-Api-Key": api_key, "Content-Type": "application/json"}
        response = requests.get(url, headers=hds)
        return {"success": True, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    

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

# ============================================================
# Weather Tool Schema 2
# ============================================================

weather_tool_schema2 = {
    "type": "function",
    "function": {
        "name": "get_weather2",
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
            thinking_started = False
            answer_started = False

            for completion in self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                stream=True,
            ):
                response_msg = completion.message
                # print("🌸", response_msg.thinking)

                # THINKING PART
                if getattr(response_msg, "thinking", None):
                    # print("THINKING:", response_msg.thinking)
                    if not thinking_started:
                        print("\n🧠 Thining...\n", end="", flush=True)
                        thinking_started = True
                    print(response_msg.thinking, end="", flush=True)

                # CONTENT PART
                if getattr(response_msg, "content", None):
                    # print("CONTENT:", response_msg.content)
                    if not answer_started:
                        answer_started = True
                        print("\n💬 Asnwer...\n", end="", flush=True)
                    print(response_msg.content, end="", flush=True)

                # TOOL PART
                if response_msg.tool_calls is not None:
                    # print("✅ Tool Call needed", response_msg.tool_calls)
                    print("\n🔧 Tool Call:")
                    self.messages.append(response_msg)

                    tool_outputs = []
                    for tool_call in response_msg.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = tool_call.function.arguments
                        print(f"    -> {fn_name}({fn_args})")

                        tool_output_content = f"Tool {fn_name} not found"

                        if fn_name in globals() and callable(globals()[fn_name]):
                            fn_to_call = globals()[fn_name]
                            executed_output = fn_to_call(**fn_args)
                            tool_output_content = str(executed_output)

                        print(tool_output_content)
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
            print("\n")
            return response_msg.content


# ============================================================


# query4 = "I live in Mumbai. I want to go out and play at a park. What things do i need to carry with me? Today is 9th april 2026."

user_query = input("Enter your query: \n")


og = MyAgent(
    client=client,
    model="glm-5:cloud",
    system="""
    You are a helpful AI agent that can use tools to solve user queries.

When a user asks a question:
- Understand the intent carefully.
- Use available tools whenever required to get accurate information.
- If a tool fails or returns an error, do NOT stop.
- Try alternative tools if available.
- If multiple tools can solve the task, choose the most appropriate one.

Be concise, accurate, and logical in your reasoning
""",
    tools=[weather_tool_schema, weather_tool_schema2, geocoding_tool_schema],
)

print("⚠️⚠️")
og(user_query)
