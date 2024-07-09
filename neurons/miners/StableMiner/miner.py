from typing import Dict, Optional

import torch
from base import BaseMiner
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from loguru import logger
from neurons.miners.StableMiner.schema import ModelConfig, TaskType
from neurons.protocol import ModelType
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import warm_up


class StableMiner(BaseMiner):
    def __init__(self) -> None:
        self.t2i_model_custom: Optional[AutoPipelineForText2Image] = None
        self.t2i_model_alchemy: Optional[AutoPipelineForText2Image] = None
        self.i2i_model_custom: Optional[AutoPipelineForImage2Image] = None
        self.i2i_model_alchemy: Optional[AutoPipelineForImage2Image] = None
        self.safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self.processor: Optional[CLIPImageProcessor] = None
        self.model_configs: Dict[str, Dict[str, ModelConfig]] = {}

        super().__init__()

        try:
            # Load the models
            self.load_models()

            # Optimize model
            self.optimize_models()

            # Serve the axon
            self.start_axon()

            # Start the miner loop
            self.loop()
        except Exception as e:
            logger.error(f"Error in StableMiner initialization: {e}")
            raise

    def load_models(self) -> None:
        try:
            # Text-to-image
            # Note: Xrunner: Disable model loading as its not necessary
            # self.t2i_model_custom = self.load_t2i_model(self.config.miner.custom_model)

            # Image-to-image
            # self.i2i_model_custom = self.load_i2i_model(self.t2i_model_custom)

            # TODO: Alchemy model
            # self.t2i_model_alchemy = None
            # self.i2i_model_alchemy = None

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(self.config.miner.device)
            self.processor = CLIPImageProcessor()

            # Note: Xrunner: Disable model config part as its not necessary
            # self.setup_model_configs()
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def load_t2i_model(self, model_name: str) -> AutoPipelineForText2Image:
        try:
            model = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.config.miner.device)

            model.set_progress_bar_config(disable=True)
            model.scheduler = DPMSolverMultistepScheduler.from_config(
                model.scheduler.config
            )

            return model
        except Exception as e:
            logger.error(f"Error loading text-to-image model: {e}")
            raise

    def load_i2i_model(
        self, t2i_model: AutoPipelineForText2Image
    ) -> AutoPipelineForImage2Image:
        try:
            model = AutoPipelineForImage2Image.from_pipe(t2i_model).to(
                self.config.miner.device
            )

            model.set_progress_bar_config(disable=True)
            model.scheduler = DPMSolverMultistepScheduler.from_config(
                model.scheduler.config
            )

            return model
        except Exception as e:
            logger.error(f"Error loading image-to-image model: {e}")
            raise

    def get_model_config(
        self,
        model_type: ModelType,
        task_type: TaskType,
    ) -> ModelConfig:
        if model_type not in self.model_configs:
            raise ValueError(f"{model_type} was not found in model_configs!")

        if task_type not in self.model_configs[model_type]:
            raise ValueError(
                f"{task_type} was not found in model_configs {model_type}!"
            )

        return self.model_configs[model_type][task_type]

    def setup_model_configs(self) -> None:
        self.model_configs = {
            # TODO: Implement Alchemy Model
            # ModelType.ALCHEMY: {
            #     # Text-to-image
            #     TaskType.TEXT_TO_IMAGE: ModelConfig(
            #         args=self.t2i_args,
            #         model=self.t2i_model_alchemy,
            #     ),
            #     TaskType.IMAGE_TO_IMAGE: ModelConfig(
            #         args=self.i2i_args,
            #         model=self.i2i_model_alchemy,
            #     ),
            # },
            ModelType.CUSTOM: {
                TaskType.TEXT_TO_IMAGE: ModelConfig(
                    args=self.t2i_args,
                    model=self.t2i_model_custom,
                ),
                # TODO: implement i2i
                # TaskType.IMAGE_TO_IMAGE: ModelConfig(
                #     args=self.i2i_args,
                #     model=self.i2i_model_custom,
                # ),
            },
        }

    def optimize_models(self) -> None:
        # TODO: Alchemy model
        return

        if self.config.miner.optimize:
            try:
                self.t2i_model_alchemy.unet = torch.compile(
                    self.t2i_model_alchemy.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )

                # Warm up model
                logger.info(
                    ">>> Warming up model with compile... "
                    "this takes roughly two minutes...",
                    color="yellow",
                )
                warm_up(self.t2i_model_alchemy, self.t2i_args)
            except Exception as e:
                logger.error(f"Error optimizing models: {e}")
                raise
