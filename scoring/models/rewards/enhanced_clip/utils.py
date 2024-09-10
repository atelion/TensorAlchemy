from typing import List, Dict, Union, TypedDict, Callable, Awaitable
import json
import httpx
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.shared_params import FunctionDefinition


# Type definitions
class ElementDict(TypedDict):
    description: str
    importance: float


class PromptBreakdown(TypedDict):
    elements: List[ElementDict]


BreakdownFunction = Callable[[str], Awaitable[PromptBreakdown]]

# Configuration functions
from neurons.config import get_openai_client, get_corcel_api_key


def get_prompt_breakdown_function() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="break_down_prompt",
            description="Break down an image prompt into key elements for CLIP",
            parameters={
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "Key element of the image, "
                                    + "Single Word (for CLIP analysis)",
                                },
                            },
                            "required": ["description"],
                        },
                        "description": "Key elements from the prompt",
                    }
                },
                "required": ["elements"],
            },
        ),
    )


def get_query_messages(
    prompt: str,
) -> List[
    Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]
]:
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="Break down image prompts into key elements."
            + " Each element should be concise and evaluatable."
            + " Assign importance based on significance to the image.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Break down this image prompt: {prompt}",
        ),
    ]


async def process_api_response(response_data: Dict) -> PromptBreakdown:
    if "choices" in response_data and response_data["choices"]:
        choice = response_data["choices"][0]

        if "message" in choice and "tool_calls" in choice["message"]:
            tool_call = choice["message"]["tool_calls"][0]

            if isinstance(tool_call, dict) and "function" in tool_call:
                return json.loads(tool_call["function"]["arguments"])

    raise ValueError("Unexpected response structure from API")


async def openai_breakdown(prompt: str) -> PromptBreakdown:
    client: AsyncOpenAI = get_openai_client()
    messages = get_query_messages(prompt)
    tool = get_prompt_breakdown_function()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        tool_choice={
            "type": "function",
            "function": {"name": tool["function"]["name"]},
        },
        tools=[tool],
        messages=messages,
    )

    return await process_api_response(response.model_dump())


async def corcel_breakdown(prompt: str) -> PromptBreakdown:
    api_key = get_corcel_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = get_query_messages(prompt)
    tool = get_prompt_breakdown_function()

    payload = {
        "model": "corcel/text-davinci-003",
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": 0,
        "tools": [tool.model_dump()],
        "tool_choice": {
            "type": "function",
            "function": {"name": tool.function.name},
        },
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.corcel.io/cortext/text",
            headers=headers,
            json=payload,
        )
        if response.status_code == 200:
            result = response.json()
            return await process_api_response(result)
        else:
            raise Exception(
                f"Corcel API request failed with status {response.status_code}"
            )


async def break_down_prompt(
    prompt: str, service: str = "openai"
) -> PromptBreakdown:
    services: Dict[str, BreakdownFunction] = {
        "openai": openai_breakdown,
        "corcel": corcel_breakdown,
    }

    if service not in services:
        raise ValueError(f"Invalid service specified: {service}")

    return await services[service](prompt)
