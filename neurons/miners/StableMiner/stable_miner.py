import torch
import copy
import asyncio

from typing import Dict, List, Optional, Any
from loguru import logger
from neurons.protocol import ModelType, ImageGeneration
from neurons.config import get_config
from neurons.miners.StableMiner.model_loader import ModelLoader
from neurons.miners.StableMiner.schema import (
    TaskType,
    TaskConfig,
    MinerConfig,
    TaskModelConfig,
)
from neurons.utils.nsfw import clean_nsfw_from_prompt
from neurons.miners.StableMiner.base import BaseMiner
from neurons.miners.StableMiner.utils import warm_up
from neurons.utils.image import image_to_base64
import torchvision.transforms as transforms
from diffusers.callbacks import SDXLCFGCutoffCallback


class StableMiner(BaseMiner):
    def __init__(self, task_configs: List[TaskConfig]) -> None:
        logger.info("Starting StableMiner initialization")
        super().__init__()

        self.task_configs = task_configs
        self.miner_config = MinerConfig()
        self.safety_checker: Optional[torch.nn.Module] = None
        self.processor: Optional[torch.nn.Module] = None

        logger.info("Initializing StableMiner...")
        self.initialize_all_models()
        self.optimize_models()
        self.start_axon()
        self.loop()

    def initialize_all_models(self) -> None:
        for task_config in self.task_configs:
            logger.info(
                f"Initializing models for task: {task_config.task_type}..."
            )
            self.initialize_model_for_task(task_config)
        self.setup_model_configs()
        self.log_gpu_memory_usage("after initializing models")

    def initialize_model_for_task(self, task_config: TaskConfig) -> None:
        self.log_gpu_memory_usage("before freeing cache")
        torch.cuda.empty_cache()
        self.log_gpu_memory_usage("after freeing cache")
        logger.info(f"Loading model for task: {task_config.task_type}")
        model = self.load_model(
            get_config().miner.custom_model, task_config.task_type
        )

        if task_config.model_type not in self.miner_config.model_configs:
            self.miner_config.model_configs[task_config.model_type] = {}

        self.miner_config.model_configs[task_config.model_type][
            task_config.task_type
        ] = TaskModelConfig(model=model)

        if (
            task_config.safety_checker
            and task_config.safety_checker_model_name
            and not self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].safety_checker
        ):
            self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].safety_checker = ModelLoader(
                get_config().miner
            ).load_safety_checker(
                task_config.safety_checker,
                task_config.safety_checker_model_name,
            )
            self.safety_checker = self.miner_config.model_configs[
                task_config.model_type
            ][task_config.task_type].safety_checker

        if (
            task_config.processor
            and not self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].processor
        ):
            self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].processor = ModelLoader(get_config().miner).load_processor(
                task_config.processor
            )
            self.processor = self.miner_config.model_configs[
                task_config.model_type
            ][task_config.task_type].processor

        if (
            get_config().refiner.enable
            and task_config.refiner_class
            and task_config.refiner_model_name
        ):
            logger.info(f"Loading refiner for task: {task_config.task_type}")
            self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].refiner = ModelLoader(get_config()).load_refiner(
                model, task_config
            )
            logger.info(f"Refiner loaded for task: {task_config.task_type}")
            self.log_gpu_memory_usage(
                f"after loading model for task {task_config.task_type}"
            )

    def get_model_config(
        self, model_type: ModelType, task_type: TaskType
    ) -> TaskModelConfig:
        if model_type not in self.miner_config.model_configs:
            raise ValueError(f"{model_type} was not found in model_configs!")
        if task_type not in self.miner_config.model_configs[model_type]:
            raise ValueError(
                f"{task_type} was not found in model_configs {model_type}!"
            )

        return self.miner_config.model_configs[model_type][task_type]

    def get_config_for_task_type(self, task_type: TaskType):
        for config in self.task_configs:
            if config.task_type == task_type:
                return config
        logger.info(f" No config found for task type {task_type}..")
        return None

    def load_model(
        self, model_name: str, task_type: TaskType
    ) -> torch.nn.Module:
        try:
            logger.info(f"Loading model {model_name} for task {task_type}...")
            task_config = self.get_config_for_task_type(task_type)
            model_loader = ModelLoader(get_config())
            model = model_loader.load(model_name, task_config)
            logger.info(f"Model {model_name} loaded successfully.")
            return model
        except Exception as e:
            logger.error(
                f"Error loading {task_type.value} model: {e}, skipping..."
            )

    def setup_model_configs(self) -> None:
        logger.info("Setting up model configurations...")

        for task_config in self.task_configs:
            model_type = task_config.model_type
            task_type = task_config.task_type

            if model_type not in self.miner_config.model_configs:
                self.miner_config.model_configs[model_type] = {}

            self.miner_config.model_configs[model_type][
                task_type
            ].args = self.get_args_for_task(task_type)

        logger.info("Model configurations set up successfully.")

    def get_args_for_task(self, task_type: TaskType) -> dict:
        if task_type == TaskType.TEXT_TO_IMAGE:
            return {
                "guidance_scale": 7.5,
                "num_inference_steps": 20,
            }
        elif task_type == TaskType.IMAGE_TO_IMAGE:
            return {
                "guidance_scale": 5,
                "strength": 0.6,
            }
        else:
            return {}

    def log_gpu_memory_usage(self, stage: str) -> None:
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            free = total - allocated

            logger.info(f"GPU memory allocated {stage}: {allocated:.2f} MB")
            logger.info(
                f"Max GPU memory allocated {stage}: {max_allocated:.2f} MB"
            )
            logger.info(f"Total GPU memory: {total:.2f} MB")
            logger.info(f"Free GPU memory: {free:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to log GPU memory usage {stage}: {str(e)}")

    def optimize_models(self) -> None:
        logger.info("Optimizing models...")
        if not get_config().miner.optimize:
            logger.info("Model optimization is disabled.")
            return

        try:
            for model_type, tasks in self.miner_config.model_configs.items():
                for task_type, config in tasks.items():
                    if config.model:
                        logger.info(f"Compiling model for task: {task_type}")
                        config.model.unet = torch.compile(
                            config.model.unet,
                            mode="reduce-overhead",
                            fullgraph=True,
                        )
                        logger.info(f"Warming up model for task: {task_type}")
                        warm_up(config.model, config.args)
            logger.info("Models optimized successfully.")
        except Exception as e:
            logger.error(f"Error optimizing models: {e}")
            raise

    async def _attempt_generate_images(
        self, synapse: ImageGeneration, model_config: TaskModelConfig
    ) -> List[str]:
        images = []
        for attempt in range(3):
            try:
                model_args = self._setup_model_args(synapse, model_config)
                seed: int = synapse.seed
                model_args["generator"] = [
                    torch.Generator(
                        device=get_config().miner.device
                    ).manual_seed(seed)
                ]

                # Set CFG Cutoff
                model_args["callback_on_step_end"] = SDXLCFGCutoffCallback(
                    cutoff_step_ratio=0.4
                )

                images = self.generate_with_refiner(model_args, model_config)

                logger.info(
                    f"Successful image generation after {attempt + 1} attempt(s).",
                )
                break
            except Exception as e:
                logger.error(
                    f"Error in attempt number {attempt + 1} to generate an image:"
                    + f" {e}... sleeping for 5 seconds..."
                )
                await asyncio.sleep(5)

        if len(images) == 0:
            logger.info(f"Failed to generate any images after {3} attempts.")

        images = self._filter_nsfw_images(images)
        return [image_to_base64(image) for image in images]

    def _setup_model_args(
        self, synapse: ImageGeneration, model_config: TaskModelConfig
    ) -> Dict[str, Any]:
        model_args: Dict[str, Any] = copy.deepcopy(model_config.args)
        try:
            model_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
            model_args["width"] = synapse.width
            model_args["denoising_end"] = 0.8
            model_args["output_type"] = "latent"
            model_args["height"] = synapse.height
            model_args["num_images_per_prompt"] = synapse.num_images_per_prompt
            model_args["guidance_scale"] = synapse.guidance_scale
            if synapse.negative_prompt:
                model_args["negative_prompt"] = [synapse.negative_prompt]

            model_args["num_inference_steps"] = getattr(
                synapse, "steps", model_args.get("num_inference_steps", 50)
            )

            if synapse.generation_type.upper() == TaskType.IMAGE_TO_IMAGE:
                model_args["image"] = transforms.transforms.ToPILImage()(
                    bt.Tensor.deserialize(synapse.prompt_image)
                )
        except AttributeError as e:
            logger.error(f"Error setting up model args: {e}")

        return model_args

    def generate_with_refiner(
        self, model_args: Dict[str, Any], model_config: TaskModelConfig
    ) -> List:
        model = model_config.model.to(get_config().miner.device)
        refiner = (
            model_config.refiner.to(get_config().miner.device)
            if model_config.refiner
            else None
        )

        if refiner and get_config().refiner.enable:
            # Init refiner args
            refiner_args = self.setup_refiner_args(model_args)
            images = model(**model_args).images

            refiner_args["image"] = images
            images = refiner(**refiner_args).images

        else:
            images = model(
                **self.without_keys(
                    model_args, ["denoising_end", "output_type"]
                )
            ).images
        return images

    def setup_refiner_args(self, model_args: Dict[str, Any]) -> Dict[str, Any]:
        refiner_args = {
            "denoising_start": model_args["denoising_end"],
            "prompt": model_args["prompt"],
            "num_inference_steps": int(model_args["num_inference_steps"] * 0.2),
        }
        model_args["num_inference_steps"] = int(
            model_args["num_inference_steps"] * 0.8
        )
        return refiner_args

    def _filter_nsfw_images(
        self, images: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        if not images:
            return images
        try:
            if any(self.nsfw_image_filter(images)):
                logger.info("An image was flagged as NSFW: discarding image.")
                self.stats.nsfw_count += 1
                return [self.empty_image_tensor() for _ in images]
        except Exception as e:
            logger.error(f"Error in NSFW filtering: {e}")
        return images

    def nsfw_image_filter(self, images: List[torch.Tensor]) -> List[bool]:
        clip_input = self.processor(
            [image for image in images],
            return_tensors="pt",
        ).to(get_config().miner.device)

        return self.safety_checker.forward(
            clip_input.pixel_values.to(
                get_config().miner.device,
            ),
        )

    def empty_image_tensor(self) -> torch.Tensor:
        return torch.zeros(3, 512, 512, dtype=torch.uint8)

    def without_keys(self, d: Dict, keys: List[str]) -> Dict:
        return {k: v for k, v in d.items() if k not in keys}
