import gradio as gr
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_manager import ModelManager


class MinimalChatUI:
    """Ultra-simple Gradio-based chat UI for testing models - avoids compatibility issues."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.available_models = model_manager.get_available_models()
        self.current_model = None

    def load_model(self, model_name, use_pipeline):
        """Load the selected model."""
        if not model_name:
            return "Please select a model first."

        try:
            start_time = time.time()
            self.model_manager.load_model(model_name, use_pipeline=use_pipeline)
            load_time = time.time() - start_time
            self.current_model = model_name

            # Get GPU stats
            gpu_info = self.model_manager.get_gpu_info()

            return f"Model '{model_name}' loaded successfully in {load_time:.2f}s.\nGPU Memory: {gpu_info['used_memory_GB']:.2f}GB / {gpu_info['total_memory_GB']:.2f}GB ({gpu_info['utilization_percent']:.1f}%)"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def unload_model(self):
        """Unload the currently loaded model."""
        if not self.current_model:
            return "No model currently loaded."

        try:
            self.model_manager.unload_model(self.current_model)
            model_name = self.current_model
            self.current_model = None

            # Get GPU stats
            gpu_info = self.model_manager.get_gpu_info()

            return f"Model '{model_name}' unloaded successfully.\nGPU Memory: {gpu_info['used_memory_GB']:.2f}GB / {gpu_info['total_memory_GB']:.2f}GB ({gpu_info['utilization_percent']:.1f}%)"
        except Exception as e:
            return f"Error unloading model: {str(e)}"

    def generate_text(self, prompt, system_prompt, temperature, max_tokens, use_pipeline):
        """Generate text from the current model."""
        if not self.current_model:
            return "Please load a model first."

        try:
            # Prepare messages format
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add user prompt
            messages.append({"role": "user", "content": prompt})

            # Generate response
            start_time = time.time()

            response = self.model_manager.generate_text(
                self.current_model,
                messages if len(messages) > 1 else prompt,  # Use simple prompt if no system prompt
                max_length=max_tokens,
                temperature=temperature,
                use_pipeline=use_pipeline
            )

            generation_time = time.time() - start_time
            gpu_info = self.model_manager.get_gpu_info()

            # Add performance info
            response_with_info = f"{response}\n\n---\nGeneration time: {generation_time:.2f}s | GPU: {gpu_info['used_memory_GB']:.1f}GB / {gpu_info['total_memory_GB']:.1f}GB"

            return response_with_info

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def process_image(self, prompt, image, temperature, max_tokens, use_pipeline):
        """Process an image with a vision model."""
        if not self.current_model:
            return "Please load a model first."

        if image is None:
            return "Please upload an image."

        try:
            # Get model config to verify it's a vision model
            model_config = self.model_manager._get_model_config(self.current_model)
            if model_config["type"] not in ["vision", "multimodal"]:
                return f"Model '{self.current_model}' is not a vision or multimodal model."

            # Generate response
            start_time = time.time()

            response = self.model_manager.process_image_text(
                self.current_model,
                image,
                prompt=prompt,
                max_length=max_tokens,
                temperature=temperature,
                use_pipeline=use_pipeline
            )

            generation_time = time.time() - start_time
            gpu_info = self.model_manager.get_gpu_info()

            # Add performance info
            response_with_info = f"{response}\n\n---\nGeneration time: {generation_time:.2f}s | GPU: {gpu_info['used_memory_GB']:.1f}GB / {gpu_info['total_memory_GB']:.1f}GB"

            return response_with_info

        except Exception as e:
            return f"Error processing image: {str(e)}"

    def launch(self):
        """Create and launch a simple interface."""
        with gr.Blocks(title="LLM Testing") as interface:
            gr.Markdown("# 🧪 LLM Testing")
            gr.Markdown("Simple interface for testing local LLMs")

            with gr.Tab("Model Management"):
                model_dropdown = gr.Dropdown(
                    choices=self.available_models,
                    label="Select Model"
                )
                use_pipeline = gr.Checkbox(label="Use Pipeline API", value=True)

                with gr.Row():
                    load_btn = gr.Button("Load Model")
                    unload_btn = gr.Button("Unload Model")

                status_box = gr.Textbox(label="Status", lines=3)

                load_btn.click(
                    fn=self.load_model,
                    inputs=[model_dropdown, use_pipeline],
                    outputs=status_box
                )

                unload_btn.click(
                    fn=self.unload_model,
                    inputs=[],
                    outputs=status_box
                )

            with gr.Tab("Text Generation"):
                system_prompt = gr.Textbox(
                    label="System Prompt (Optional)",
                    placeholder="You are a helpful AI assistant...",
                    lines=2
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5
                )

                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=64, maximum=2048, value=512, step=64,
                        label="Max Tokens"
                    )

                generate_btn = gr.Button("Generate")
                response = gr.Textbox(label="Response", lines=10)

                generate_btn.click(
                    fn=self.generate_text,
                    inputs=[prompt, system_prompt, temperature, max_tokens, use_pipeline],
                    outputs=response
                )

                # Add example prompts
                gr.Examples(
                    examples=[
                        ["Tell me about yourself"],
                        ["Explain quantum computing in simple terms"],
                        ["Write a short poem about AI"]
                    ],
                    inputs=prompt
                )

            with gr.Tab("Vision Models"):
                image_input = gr.Image(type="filepath", label="Upload Image")

                image_prompt = gr.Textbox(
                    label="Prompt (Optional)",
                    placeholder="Describe this image...",
                    lines=2
                )

                with gr.Row():
                    vision_temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    vision_max_tokens = gr.Slider(
                        minimum=64, maximum=1024, value=256, step=64,
                        label="Max Tokens"
                    )

                vision_btn = gr.Button("Process Image")
                vision_response = gr.Textbox(label="Response", lines=10)

                vision_btn.click(
                    fn=self.process_image,
                    inputs=[image_prompt, image_input, vision_temperature, vision_max_tokens, use_pipeline],
                    outputs=vision_response
                )

                # Add example prompts for vision
                gr.Examples(
                    examples=[
                        ["Describe this image in detail"],
                        ["What objects do you see?"],
                        ["What's happening in this image?"]
                    ],
                    inputs=image_prompt
                )

        # Launch with share=True to ensure it works
        interface.launch(share=True)


if __name__ == "__main__":
    # Create model manager
    model_manager = ModelManager(config_path="../config/models.json")

    # Create and launch UI
    ui = MinimalChatUI(model_manager)
    ui.launch()