import random
import time
from typing import Optional
from loguru import logger

from neurons.config import get_corcel_api_key
from neurons.validator.services.openai.service import (
    get_openai_service,
    OpenAIRequestFailed,
)
from .corcel import call_corcel, corcel_parse_response


def get_random_creature():
    return random.choice(
        [
            "cat",
            "dog",
            "elephant",
            "lion",
            "butterfly",
            "eagle",
            "dolphin",
            "snake",
            "dragon",
            "unicorn",
            "phoenix",
            "mermaid",
            "centaur",
            "griffin",
            "werewolf",
            "fairy",
            "goblin",
            "minotaur",
            "pegasus",
            "kraken",
            "octopus",
            "panda",
            "giraffe",
            "kangaroo",
            "penguin",
            "parrot",
            "tiger",
            "bear",
            "rabbit",
            "turtle",
            "fox",
            "owl",
            "human",
            "man",
            "woman",
            "girl",
            "boy",
            "robot",
            "drone",
        ]
    )


def get_random_perspective():
    return random.choice(
        [
            "aerial",
            "close-up",
            "panoramic",
            "microscopic",
            "bird's-eye",
            "worm's-eye",
            "fisheye",
            "top-down",
            "side-view",
            "rear-view",
            "isometric",
            "first-person",
            "third-person",
            "macro",
            "wide-angle",
            "telephoto",
            "tilted",
            "skewed",
            "distorted",
            "flipped",
            "mirrored",
            "kaleidoscopic",
            "cross-section",
            "cutaway",
            "translucent",
            "x-ray",
            "thermal",
            "infrared",
            "ultraviolet",
            "low-angle",
            "high-angle",
        ]
    )


def get_random_adjective():
    return random.choice(
        [
            "happy",
            "sad",
            "excited",
            "tired",
            "hungry",
            "playful",
            "curious",
            "brave",
            "majestic",
            "enchanted",
            "mysterious",
            "radiant",
            "ethereal",
            "vibrant",
            "serene",
            "bustling",
            "whimsical",
            "luminous",
            "enigmatic",
            "shiny",
            "colorful",
            "rusty",
            "old",
            "new",
            "large",
            "small",
            "tall",
            "short",
            "wide",
            "narrow",
            "thick",
            "thin",
            "smooth",
        ]
    )


def get_random_object():
    return random.choice(
        [
            "book",
            "pen",
            "phone",
            "camera",
            "guitar",
            "bicycle",
            "car",
            "cup",
            "crystal",
            "tome",
            "amulet",
            "scepter",
            "chalice",
            "orb",
            "mirror",
            "locket",
            "tapestry",
            "sculpture",
            "lamp",
            "chair",
            "table",
            "umbrella",
            "hammer",
            "scissors",
            "knife",
            "spoon",
            "fork",
            "paintbrush",
            "vase",
            "clock",
            "globe",
            "telescope",
            "human",
            "human face",
        ]
    )


def get_random_background():
    return random.choice(
        [
            "spaceship",
            "near earth orbit",
            "futuristic city",
            "city of 2033",
            "mars",
            "beach",
            "mountains",
            "city",
            "countryside",
            "park",
            "library",
            "cafe",
            "bedroom",
            "forest",
            "castle",
            "cave",
            "island",
            "desert",
            "underwater",
            "sky",
            "garden",
            "ruins",
            "stadium",
            "mall",
            "factory",
            "farm",
            "school",
            "hospital",
            "airport",
            "train station",
            "bridge",
            "tunnel",
            "highway",
            "river",
            "lake",
            "ocean",
            "space",
        ]
    )


def generate_story_prompt() -> str:
    random.seed(int(time.time()))
    random_creature = get_random_creature()
    random_adjective = get_random_adjective()
    random_object = get_random_object()
    random_background = get_random_background()
    random_perspective = get_random_perspective()

    to_return: str = (
        "You are an image prompt generator. "
        "Your purpose is to generate a single, "
        "short story that can be used as a prompt for Dalle-3. "
        "Please ensure that the story is creative, "
        "visually descriptive, and coherent. "
        "The story should be less than 30 words. "
        "Avoid using any additional elements or deviating from "
        "the specified creature, adjective, object, and background."
        "The story **must** incorporate the following elements:\n\n"
        f"- Background: {random_background}\n\n"
        f"- Creature: {random_creature}\n"
        f"- Adjective: {random_adjective}\n"
    )

    if random.random() > 0.85:
        to_return += f"- Object: {random_object}\n"

    if random.random() > 0.85:
        to_return += f"- Perspective: {random_perspective}\n\n"

    return to_return


async def generate_random_prompt_gpt(
    model: str = "gpt-4",
    prompt: Optional[str] = None,
):
    response = None
    if not prompt:
        prompt = generate_story_prompt()

    if get_corcel_api_key():
        try:
            response = call_corcel(prompt)
            if response:
                response = corcel_parse_response(response)
                if response.startswith("{"):
                    response = None
        except Exception as e:
            logger.error(f"An unexpected error occurred calling corcel: {e}")
            logger.error("Falling back to OpenAI if available...")

    if not response:
        try:
            response = await get_openai_service().create_completion_request(
                model,
                prompt,
            )
        except OpenAIRequestFailed as e:
            logger.error(f"error during creation of completion prompt: {e}")

    if response:
        response = response.replace('"', "")
        response = response.strip()

    return response
