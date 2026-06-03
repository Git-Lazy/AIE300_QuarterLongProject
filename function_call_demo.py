import ast
import json
import os
import re
from typing import Any, Dict, List, Tuple

import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment to run this demo.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city for which to get weather information.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_items",
        "description": "Search a small inventory of items by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search for in the inventory.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Compute a simple math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression using numbers and +, -, *, /, and parentheses.",
                }
            },
            "required": ["expression"],
        },
    },
]

INVENTORY = [
    {"name": "Red Apple", "description": "A crisp, juicy apple."},
    {"name": "Blue Shirt", "description": "A cotton shirt in size M."},
    {"name": "Coffee Mug", "description": "A ceramic mug for hot beverages."},
    {"name": "Weather Station", "description": "Indoor/outdoor temperature sensor."},
]


def safe_calculate(expression: str) -> str:
    expression = expression.strip()
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ValueError("Expression contains invalid characters.")

    node = ast.parse(expression, mode="eval")

    def validate(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.BinOp):
            validate(node.left)
            validate(node.right)
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            validate(node.operand)
        elif isinstance(node, ast.Num):
            return
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    validate(node)
    result = eval(compile(node, filename="<expr>", mode="eval"), {"__builtins__": {}}, {})
    return str(result)


def get_weather(city: str) -> str:
    city = city.strip()
    example_weather = {
        "Seattle": "cloudy, 58°F, light breeze",
        "New York": "sunny, 72°F, low humidity",
        "Tokyo": "partly cloudy, 66°F, gentle breeze",
    }
    return example_weather.get(city, f"clear, 70°F in {city}")


def search_items(query: str) -> str:
    matches = [item for item in INVENTORY if query.lower() in item["name"].lower() or query.lower() in item["description"].lower()]
    if not matches:
        return f"No items found for '{query}'."
    return json.dumps(matches, indent=2)


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    if tool_name == "get_weather":
        return get_weather(tool_args["city"])
    if tool_name == "search_items":
        return search_items(tool_args["query"])
    if tool_name == "calculate":
        return safe_calculate(tool_args["expression"])
    raise ValueError(f"Unknown tool: {tool_name}")


def parse_tool_call(message: Any) -> Tuple[str, Dict[str, Any]]:
    if getattr(message, "tool_calls", None):
        tool_call = message.tool_calls[0]
        arguments = tool_call.function.arguments
        name = tool_call.function.name
    elif getattr(message, "function_call", None):
        tool_call = message.function_call
        arguments = tool_call.arguments
        name = tool_call.name
    else:
        raise ValueError("No tool call found in the response message.")

    try:
        parsed_args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON arguments from model: {exc}\n{arguments}") from exc

    return name, parsed_args


def run_demo() -> None:
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that can use tools for weather, search, and calculations.",
    }
    user_message = {
        "role": "user",
        "content": "What is the weather in Seattle, and can you also calculate 12 / 5 for me?",
    }

    print("=== Step 1: Send user message with tool schemas ===")
    print(json.dumps({"system": system_message, "user": user_message}, indent=2))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_message, user_message],
        tools=TOOLS,
        temperature=0,
        max_tokens=512,
    )

    model_message = response.choices[0].message
    print("\n=== Step 2: Model response ===")
    if getattr(model_message, "content", None):
        print("Assistant content:", model_message.content)
    if getattr(model_message, "tool_calls", None):
        print("Model requested a tool call:")
        tool_name, tool_args = parse_tool_call(model_message)
        print("Tool name:", tool_name)
        print("Tool arguments:", json.dumps(tool_args, indent=2))

        tool_result = execute_tool(tool_name, tool_args)
        print("\n=== Step 3: Execute tool in code ===")
        print(f"{tool_name}() returned: {tool_result}")

        followup_message = {
            "role": "tool",
            "name": tool_name,
            "content": tool_result,
        }
        followup_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message, followup_message],
            tools=TOOLS,
            temperature=0,
            max_tokens=512,
        )

        final_message = followup_response.choices[0].message
        print("\n=== Step 4: Model final answer ===")
        print(final_message.content)
    else:
        print("Model did not request a tool call. Full assistant response:")
        print(model_message.content)


if __name__ == "__main__":
    run_demo()
