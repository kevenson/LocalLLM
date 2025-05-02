import os
import torch
import time
from typing import Dict, Any, List, Optional

# For GGUF models using llama.cpp
from llama_cpp import Llama


class LlamaModelManager:
    """Model manager for llama.cpp-based GGUF models."""

    def __init__(self, models_dir: str = "models/gguf"):
        """
        Initialize the LlamaModelManager.

        Args:
            models_dir: Directory where GGUF models are stored
        """
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_configs = {}

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Scan for available models
        self._scan_models()

    def _scan_models(self):
        """Scan for available GGUF models in the models directory."""
        self.available_models = []

        for filename in os.listdir(self.models_dir):
            if filename.endswith('.gguf'):
                model_name = os.path.splitext(filename)[0]
                model_path = os.path.join(self.models_dir, filename)
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)

                self.available_models.append(model_name)
                self.model_configs[model_name] = {
                    "path": model_path,
                    "size_gb": file_size_gb,
                    "type": "text"  # Default to text, can be updated later
                }

        print(f"Found {len(self.available_models)} GGUF models")

    def get_available_models(self) -> List[str]:
        """Get list of available GGUF models."""
        return self.available_models

    def load_model(self, model_name: str, n_gpu_layers: int = -1,
                   n_ctx: int = 4096, **kwargs) -> None:
        """
        Load a GGUF model using llama.cpp.

        Args:
            model_name: Name of the model to load
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            **kwargs: Additional parameters for llama.cpp
        """
        if model_name in self.loaded_models:
            print(f"Model '{model_name}' already loaded.")
            return

        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in models directory.")

        model_path = self.model_configs[model_name]["path"]
        print(f"Loading GGUF model from {model_path}...")

        start_time = time.time()

        # Configure GPU acceleration
        if torch.cuda.is_available():
            try:
                cuda_devices = kwargs.pop("cuda_devices", "0")  # Default to first GPU
                print(f"Using CUDA for model acceleration on device(s): {cuda_devices}")

                import os
                os.environ["LLAMA_CUBLAS"] = "1"  # Force CUDA usage before importing Llama

                # Load the model with GPU acceleration
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers= -1,#n_gpu_layers,  # -1 means all layers on GPU
                    n_ctx=n_ctx,
                    n_threads=kwargs.pop("n_threads", 8),
                    use_mlock=kwargs.pop("use_mlock", True),  # Pin memory
                    jinja=True,  # Enable Jinja template support
                    reasoning_format="deepseek",  # Use the appropriate reasoning format
                    **kwargs
                )

                # Store additional metadata
                self.model_configs[model_name]["n_gpu_layers"] = n_gpu_layers
                self.model_configs[model_name]["n_ctx"] = n_ctx

            except Exception as e:
                print(f"Error loading model with GPU: {str(e)}")
                print("Falling back to CPU mode (will be much slower)")

                # Fall back to CPU
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=0,  # No GPU layers
                    n_ctx=n_ctx,
                    **kwargs
                )
        else:
            print("CUDA not available, using CPU only (will be slow)")

            # CPU only
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=n_ctx,
                **kwargs
            )

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        # Store the loaded model
        self.loaded_models[model_name] = llm

    def unload_model(self, model_name: str) -> None:
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()  # Clear CUDA cache
            print(f"Model '{model_name}' unloaded")
        else:
            print(f"Model '{model_name}' not currently loaded")

    def generate_text(self, model_name: str, prompt: str,
                      max_tokens: int = 512, temperature: float = 0.7,
                      **kwargs) -> str:
        """
        Generate text using a loaded GGUF model.

        Args:
            model_name: Name of the model to use
            prompt: Text prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters for llama.cpp

        Returns:
            Generated text
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model '{model_name}' not loaded. Call load_model first.")

        llm = self.loaded_models[model_name]

        # Format prompt based on format (chat or regular)
        if isinstance(prompt, list):
            # Chat format - convert to format llama.cpp can understand
            formatted_prompt = self._format_chat_prompt(prompt)
        else:
            # Regular text prompt
            formatted_prompt = prompt

        # Generate text
        start_time = time.time()

        output = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=kwargs.pop("stop", None),
            echo=kwargs.pop("echo", False),
            **kwargs
        )

        generation_time = time.time() - start_time

        # Extract generated text
        if isinstance(output, dict):
            generated_text = output["choices"][0]["text"]
        else:
            generated_text = output

        print(f"Generated {len(generated_text)} characters in {generation_time:.2f} seconds")

        return generated_text

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt string for llama.cpp.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        formatted_prompt = ""

        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role == "system":
                formatted_prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n"
            else:
                # Default to user for unknown roles
                formatted_prompt += f"<|user|>\n{content}\n"

        # Add final assistant prompt
        formatted_prompt += "<|assistant|>\n"

        return formatted_prompt

    def get_gpu_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"error": "GPU not available"}

        gpu_mem_alloc = torch.cuda.memory_allocated(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
        gpu_mem_free = gpu_mem_total - gpu_mem_alloc

        return {
            "total_memory_GB": gpu_mem_total / (1024 ** 3),
            "used_memory_GB": gpu_mem_alloc / (1024 ** 3),
            "free_memory_GB": gpu_mem_free / (1024 ** 3),
            "utilization_percent": (gpu_mem_alloc / gpu_mem_total) * 100
        }