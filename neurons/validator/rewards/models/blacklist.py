import numpy as np
import bittensor as bt
from loguru import logger

from neurons.utils.image import tensor_to_image

from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.models.types import RewardModelType


class BlacklistFilter(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.BLACKLIST

    def __init__(self):
        super().__init__()
        self.answer_blacklist = []
        self.question_blacklist = []

    def get_reward(self, response: bt.Synapse) -> float:
        # Check the number of returned images in the response
        if len(response.images) != response.num_images_per_prompt:
            return 1.0

        # If any images in the response fail the reward for that response is
        # 0.0
        for img_tensor in response.images:
            # Check if the image can be serialized
            try:
                image = tensor_to_image(img_tensor)

                # Check if the image is black image
                if np.array(image).sum() < 1:
                    return 1.0

            except Exception:
                logger.warning("Could not deserialise image")
                return 1.0

            # Check if the image has the type bt.tensor
            if not isinstance(img_tensor, bt.Tensor):
                return 1.0

            if image.shape[1] != response.width:
                return 1.0

            # check image size
            if image.shape[2] != response.height:
                return 1.0

        return 0.0
