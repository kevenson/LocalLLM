import os
import torch
import json
from typing import Dict, Any, Optional, List, Union
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    AutoModelForVision2Seq, pipeline
)


class ModelManager:
    """
    Manages loading, caching, and switching between different models.
    """

    def __init__(self, config_path: str = "../config/models.json", use_gpu: bool = True):
        self.config_path = config_path
        self.models_config = self._load_config()
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.loaded_processors = {}
        self.loaded_pipelines = {}

        # Check CUDA availability first
        cuda_available = torch.cuda.is_available()
        if use_gpu and not cuda_available:
            print("WARNING: CUDA is not available. This could be because:")
            print("  - PyTorch was not installed with CUDA support")
            print("  - NVIDIA drivers are not properly installed")
            print("  - Your GPU is not CUDA-compatible")
            print("Falling back to CPU mode. Models will load but run much slower.")

        # GPU settings
        self.use_gpu = use_gpu and cuda_available
        self.device = "cuda" if self.use_gpu else "cpu"

        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
        else:
            print("Using CPU mode (slower performance)")
            self.gpu_mem_total = 0

    def _load_config(self) -> Dict:
        """Load model configurations from JSON file."""
        if not os.path.exists(self.config_path):
            # Create default config if it doesn't exist
            default_config = {
                "text_models": {
                    "phi-4-reasoning-plus": {
                        "model_id": "microsoft/Phi-4-reasoning-plus",
                        "type": "text",
                        "precision": "fp16",
                        "max_length": 2048
                    },
                    # Add more models as needed
                },
                "vision_models": {
                    # Example vision model
                    "llava-1.5": {
                        "model_id": "llava-hf/llava-1.5-7b-hf",
                        "type": "vision",
                        "precision": "fp16",
                        "max_length": 2048
                    }
                },
                "multimodal_models": {
                    # Example multimodal model
                    "idefics-9b": {
                        "model_id": "HuggingFaceM4/idefics-9b-instruct",
                        "type": "multimodal",
                        "precision": "int8",
                        "max_length": 1024
                    }
                }
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_available_models(self, model_type: Optional[str] = None) -> List[str]:
        """Get list of available models, optionally filtered by type."""
        available_models = []

        if model_type is None or model_type == "text":
            available_models.extend(self.models_config.get("text_models", {}).keys())

        if model_type is None or model_type == "vision":
            available_models.extend(self.models_config.get("vision_models", {}).keys())

        if model_type is None or model_type == "multimodal":
            available_models.extend(self.models_config.get("multimodal_models", {}).keys())

        return available_models

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        for model_type in ["text_models", "vision_models", "multimodal_models"]:
            if model_name in self.models_config.get(model_type, {}):
                return self.models_config[model_type][model_name]

        raise ValueError(f"Model '{model_name}' not found in configuration.")

    def load_model(self, model_name: str, use_pipeline: bool = True) -> None:
        """
        Load a model by name.

        Args:
            model_name: Name of the model as defined in the config
            use_pipeline: Whether to use the pipeline API or load the model directly
        """
        if model_name in self.loaded_models and not use_pipeline:
            print(f"Model '{model_name}' already loaded.")
            return

        if model_name in self.loaded_pipelines and use_pipeline:
            print(f"Pipeline for '{model_name}' already loaded.")
            return

        # Get model configuration
        model_config = self._get_model_config(model_name)
        model_id = model_config["model_id"]
        model_type = model_config["type"]
        precision = model_config.get("precision", "fp16")

        # Handle different precision options
        dtype = torch.float16 if precision == "fp16" else torch.float32
        quantization_config = None
        if precision == "int8":
            quantization_config = {"load_in_8bit": True}
        elif precision == "int4":
            quantization_config = {"load_in_4bit": True}

        # Load using pipeline if requested
        if use_pipeline:
            task = "text-generation"
            if model_type == "vision":
                task = "image-to-text"
            elif model_type == "multimodal":
                task = "image-to-text"  # For multimodal models like LLaVA

            self.loaded_pipelines[model_name] = pipeline(
                task,
                model=model_id,
                device=self.device,
                torch_dtype=dtype,
                **({} if quantization_config is None else quantization_config)
            )
            print(f"Loaded pipeline for model '{model_name}'")
            return

        # Otherwise load model components separately
        print(f"Loading model '{model_name}' ({model_id})...")

        # Load tokenizer first
        self.loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_id)

        # Load processor for vision/multimodal models
        if model_type in ["vision", "multimodal"]:
            self.loaded_processors[model_name] = AutoProcessor.from_pretrained(model_id)

        # Load the actual model based on type
        if model_type == "text":
            self.loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
                **(quantization_config or {})
            )
        elif model_type == "vision":
            self.loaded_models[model_name] = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
                **(quantization_config or {})
            )
        elif model_type == "multimodal":
            # Most multimodal models on HF use CausalLM architecture
            self.loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
                **(quantization_config or {})
            )

        print(f"Successfully loaded model '{model_name}'")

    def unload_model(self, model_name: str) -> None:
        """Unload a model to free up GPU memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()
            print(f"Unloaded model '{model_name}'")

        if model_name in self.loaded_tokenizers:
            del self.loaded_tokenizers[model_name]

        if model_name in self.loaded_processors:
            del self.loaded_processors[model_name]

        if model_name in self.loaded_pipelines:
            del self.loaded_pipelines[model_name]
            torch.cuda.empty_cache()
            print(f"Unloaded pipeline for '{model_name}'")

    def generate_text(self, model_name: str, prompt: str,
                      max_length: int = None, use_pipeline: bool = True,
                      **kwargs) -> str:
        """
        Generate text using a loaded text model.

        Args:
            model_name: Name of the model to use
            prompt: Input prompt for text generation
            max_length: Maximum length of generated text
            use_pipeline: Whether to use pipeline API or direct generation
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text as a string
        """
        model_config = self._get_model_config(model_name)

        if model_config["type"] != "text":
            raise ValueError(f"Model '{model_name}' is not a text model.")

        if max_length is None:
            max_length = model_config.get("max_length", 1024)

        # Ensure model is loaded
        if use_pipeline and model_name not in self.loaded_pipelines:
            self.load_model(model_name, use_pipeline=True)
        elif not use_pipeline and model_name not in self.loaded_models:
            self.load_model(model_name, use_pipeline=False)

        # Generate using pipeline
        if use_pipeline:
            pipe = self.loaded_pipelines[model_name]
            if isinstance(prompt, list):  # For chat format
                result = pipe(prompt, max_length=max_length, **kwargs)
            else:
                result = pipe(prompt, max_length=max_length, **kwargs)

            # Handle different pipeline output formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                return str(result[0])

            return str(result)

        # Generate using direct model API
        tokenizer = self.loaded_tokenizers[model_name]
        model = self.loaded_models[model_name]

        # Handle different prompt formats
        if isinstance(prompt, list):  # Chat format
            inputs = tokenizer(tokenizer.apply_chat_template(prompt, tokenize=False),
                               return_tensors="pt").to(self.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                **kwargs
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_image_text(self, model_name: str, image_path: str, prompt: str = None,
                           max_length: int = None, use_pipeline: bool = True,
                           **kwargs) -> str:
        """
        Process image and optional text prompt using a vision or multimodal model.

        Args:
            model_name: Name of the model to use
            image_path: Path to the image file
            prompt: Optional text prompt to accompany the image
            max_length: Maximum length of generated text
            use_pipeline: Whether to use pipeline API or direct generation
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text description or response as a string
        """
        from PIL import Image

        model_config = self._get_model_config(model_name)
        model_type = model_config["type"]

        if model_type not in ["vision", "multimodal"]:
            raise ValueError(f"Model '{model_name}' is not a vision or multimodal model.")

        if max_length is None:
            max_length = model_config.get("max_length", 1024)

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Ensure model is loaded
        if use_pipeline and model_name not in self.loaded_pipelines:
            self.load_model(model_name, use_pipeline=True)
        elif not use_pipeline and model_name not in self.loaded_models:
            self.load_model(model_name, use_pipeline=False)

        # Process using pipeline
        if use_pipeline:
            pipe = self.loaded_pipelines[model_name]
            if prompt:
                result = pipe(image, prompt=prompt, max_length=max_length, **kwargs)
            else:
                result = pipe(image, max_length=max_length, **kwargs)

            # Handle different pipeline output formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                return str(result[0])

            return str(result)

        # Process using direct model API
        processor = self.loaded_processors[model_name]
        model = self.loaded_models[model_name]

        # Handle different model input requirements
        if model_type == "vision":
            # For pure vision-to-text models
            inputs = processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    **kwargs
                )

            return processor.decode(outputs[0], skip_special_tokens=True)

        elif model_type == "multimodal":
            # For multimodal models like LLaVA
            prompt = prompt or "Describe this image in detail."
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    **kwargs
                )

            return processor.decode(outputs[0], skip_special_tokens=True)

    def get_gpu_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not self.use_gpu or not torch.cuda.is_available():
            return {"error": "GPU not available"}

        gpu_mem_alloc = torch.cuda.memory_allocated(0)
        gpu_mem_free = self.gpu_mem_total - gpu_mem_alloc

        return {
            "total_memory_GB": self.gpu_mem_total / (1024 ** 3),
            "used_memory_GB": gpu_mem_alloc / (1024 ** 3),
            "free_memory_GB": gpu_mem_free / (1024 ** 3),
            "utilization_percent": (gpu_mem_alloc / self.gpu_mem_total) * 100
        }