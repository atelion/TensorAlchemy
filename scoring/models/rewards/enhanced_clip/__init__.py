import traceback

import torch
import bittensor as bt

from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from typing import List

from neurons.utils.image import synapse_to_image
from neurons.config import get_device
from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType
from scoring.models.rewards.enhanced_clip.utils import (
    break_down_prompt,
    PromptBreakdown,
)


class EnhancedClipRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.ENHANCED_CLIP

    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # What is the lower cutoff for scoring
        self.threshold_min = 0.0

    def compute_clip_score(
        self,
        prompt_elements: PromptBreakdown,
        response: bt.Synapse,
    ) -> float:
        if not response.images:
            return 0.0

        if any(image is None for image in response.images):
            return 0.0

        try:
            # Convert synapse image to PIL Image
            image = synapse_to_image(response)

            descriptions = [
                element["description"]
                for element in prompt_elements["elements"]
            ]

            # Process inputs
            inputs = self.processor(
                text=descriptions,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: v.to(self.device) for k, v in inputs.items()
            }  # Move inputs to the same device as the model

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use raw logits
            logits_per_image = outputs.logits_per_image.squeeze()

            # Normalize logits to [0, 1] range
            similarities = (logits_per_image - logits_per_image.min()) / (
                logits_per_image.max() - logits_per_image.min()
            )

            # Apply a stricter threshold
            adjusted_similarities = torch.where(
                similarities > self.threshold_min,
                similarities,
                torch.zeros_like(similarities),
            )

            # Add 1 to each adjusted similarity before taking the product
            final_result: float = (adjusted_similarities + 1).prod().item() - 1

            for i, desc in enumerate(descriptions):
                logger.info(
                    f"Element: {desc}, "
                    f"Similarity: {similarities[i].item():.4f}, "
                    f"Adjusted: {adjusted_similarities[i].item():.4f}"
                )

            logger.info(f"Enhanced CLIP similarity score: {final_result:.4f}")

            # Now normalize by the length of the descriptions
            # This should give us a nicer ImageReward score compatable range
            # although not strictly bound to 0..1
            final_result /= len(descriptions)

            return final_result

        except Exception:
            logger.error(traceback.format_exc())
            return 0.0

    async def get_rewards(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        prompt_elements: PromptBreakdown = await break_down_prompt(
            synapse.prompt
        )

        def get_reward(response: bt.Synapse) -> float:
            return self.compute_clip_score(prompt_elements, response)

        rewards = await super().build_rewards_tensor(
            get_reward,
            synapse,
            responses,
        )

        return rewards
