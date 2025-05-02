import gradio as gr
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_manager import ModelManager


class ChatUI:
    """Simple Gradio-based chat UI for testing models."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.available_models = model_manager.get_available_models()
        self.chat_history = []
        self.current_model = None
        self.interface = None

    def _update_available_models(self):
        """Refresh the list of available models."""
        self.available_models = self.model_manager.get_available_models()
        return gr.Dropdown.update(choices=self.available_models)

    def _load_selected_model(self, model_name, use_pipeline):
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

    def _unload_current_model(self):
        """Unload the currently loaded model."""
        if self.current_model:
            try:
                self.model_manager.unload_model(self.current_model)
                model_name = self.current_model
                self.current_model = None

                # Get GPU stats
                gpu_info = self.model_manager.get_gpu_info()

                return f"Model '{model_name}' unloaded successfully.\nGPU Memory: {gpu_info['used_memory_GB']:.2f}GB / {gpu_info['total_memory_GB']:.2f}GB ({gpu_info['utilization_percent']:.1f}%)"
            except Exception as e:
                return f"Error unloading model: {str(e)}"
        else:
            return "No model currently loaded."

    def _handle_text_message(self, message, history, system_prompt, temperature, max_tokens, use_pipeline):
        """Process a text message and update chat history."""
        if not self.current_model:
            return history + [[message, "Please load a model first."]]

        try:
            # Prepare chat history format
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add chat history
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:  # Skip empty responses
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current message
            messages.append({"role": "user", "content": message})

            # Generate response
            start_time = time.time()

            response = self.model_manager.generate_text(
                self.current_model,
                messages if len(messages) > 1 else message,  # Use simple prompt if no history
                max_length=max_tokens,
                temperature=temperature,
                use_pipeline=use_pipeline
            )

            generation_time = time.time() - start_time
            gpu_info = self.model_manager.get_gpu_info()

            # Add performance info at the end of response
            response_with_info = f"{response}\n\n---\nGeneration time: {generation_time:.2f}s | GPU: {gpu_info['used_memory_GB']:.1f}GB / {gpu_info['total_memory_GB']:.1f}GB"

            return history + [[message, response_with_info]]

        except Exception as e:
            return history + [[message, f"Error generating response: {str(e)}"]]

    def _handle_image_text_message(self, message, image, history, temperature, max_tokens, use_pipeline):
        """Process an image with optional text and update chat history."""
        if not self.current_model:
            return history + [[message, "Please load a model first."]]

        if image is None:
            return history + [[message, "Please upload an image."]]

        try:
            # Get model config to verify it's a vision model
            model_config = self.model_manager._get_model_config(self.current_model)
            if model_config["type"] not in ["vision", "multimodal"]:
                return history + [[message, f"Model '{self.current_model}' is not a vision or multimodal model."]]

            # Generate response
            start_time = time.time()

            image_path = image  # Gradio provides the path to the uploaded image

            response = self.model_manager.process_image_text(
                self.current_model,
                image_path,
                prompt=message,
                max_length=max_tokens,
                temperature=temperature,
                use_pipeline=use_pipeline
            )

            generation_time = time.time() - start_time
            gpu_info = self.model_manager.get_gpu_info()

            # Add performance info
            response_with_info = f"{response}\n\n---\nGeneration time: {generation_time:.2f}s | GPU: {gpu_info['used_memory_GB']:.1f}GB / {gpu_info['total_memory_GB']:.1f}GB"

            # For image messages, we'll show the image alongside the response in the history
            user_message = f"[Image] {message if message else 'Analyze this image'}"

            return history + [[user_message, response_with_info]]

        except Exception as e:
            return history + [[message, f"Error processing image: {str(e)}"]]

    def launch(self):
        """Create and launch the Gradio interface."""
        with gr.Blocks(title="LLM Testing Platform") as self.interface:
            gr.Markdown("# 🤖 LLM Testing Platform")
            gr.Markdown("Test different models from Hugging Face with a simple chat interface")

            with gr.Tab("Model Management"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            choices=self.available_models,
                            label="Select Model",
                            info="Choose a model to load"
                        )
                        refresh_button = gr.Button("🔄 Refresh Model List")
                        use_pipeline = gr.Checkbox(
                            label="Use Pipeline API",
                            value=True,
                            info="Use HuggingFace Pipeline API (recommended for most cases)"
                        )

                    with gr.Column(scale=2):
                        load_button = gr.Button("⬇️ Load Selected Model")
                        unload_button = gr.Button("⬆️ Unload Current Model")
                        model_status = gr.Textbox(label="Model Status", lines=4)

            with gr.Tab("Text Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        system_prompt = gr.Textbox(
                            label="System Prompt (Optional)",
                            placeholder="You are a helpful AI assistant...",
                            lines=2
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature"
                        )
                        max_tokens = gr.Slider(
                            minimum=64, maximum=4096, value=1024, step=64,
                            label="Max Tokens"
                        )

                    with gr.Column(scale=2):
                        # Use custom chatbot instead of ChatInterface to avoid button issues
                        chatbot = gr.Chatbot(label="Chat with Model")
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=2,
                            show_label=False
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Submit", variant="primary")
                            clear_btn = gr.Button("Clear")

                        # Example prompts
                        gr.Examples(
                            examples=["Tell me about yourself",
                                      "Explain quantum computing in simple terms",
                                      "Write a short poem about AI"],
                            inputs=msg
                        )

            with gr.Tab("Vision Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="filepath", label="Upload Image")
                        vision_temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature"
                        )
                        vision_max_tokens = gr.Slider(
                            minimum=64, maximum=2048, value=512, step=64,
                            label="Max Tokens"
                        )

                    with gr.Column(scale=2):
                        # Use custom chatbot for vision too
                        vision_chatbot = gr.Chatbot(label="Vision Model Chat")
                        vision_msg = gr.Textbox(
                            label="Message (optional)",
                            placeholder="Describe this image...",
                            lines=2,
                            show_label=False
                        )

                        with gr.Row():
                            vision_submit_btn = gr.Button("Submit", variant="primary")
                            vision_clear_btn = gr.Button("Clear")

                        # Example prompts for vision models
                        gr.Examples(
                            examples=["Describe this image in detail",
                                      "What objects do you see?",
                                      "What's happening in this image?"],
                            inputs=vision_msg
                        )

            # Set up event handlers

            # Model management
            refresh_button.click(self._update_available_models, outputs=model_dropdown)
            load_button.click(self._load_selected_model,
                              inputs=[model_dropdown, use_pipeline],
                              outputs=model_status)
            unload_button.click(self._unload_current_model, outputs=model_status)

            # Text chat
            text_chat_state = gr.State([])  # To store chat history

            def submit_text_message(message, history):
                if not message.strip():
                    return history, ""
                result = self._handle_text_message(
                    message, history, system_prompt.value,
                    temperature.value, max_tokens.value, use_pipeline.value
                )
                return result, ""

            submit_btn.click(
                submit_text_message,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )

            # Also submit on Enter key
            msg.submit(
                submit_text_message,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )

            # Clear button
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

            # Vision chat
            def submit_vision_message(message, history):
                if not image_input.value:
                    return history + [["", "Please upload an image first."]], ""
                result = self._handle_image_text_message(
                    message, image_input.value, history,
                    vision_temperature.value, vision_max_tokens.value, use_pipeline.value
                )
                return result, ""

            vision_submit_btn.click(
                submit_vision_message,
                inputs=[vision_msg, vision_chatbot],
                outputs=[vision_chatbot, vision_msg]
            )

            # Also submit on Enter key
            vision_msg.submit(
                submit_vision_message,
                inputs=[vision_msg, vision_chatbot],
                outputs=[vision_chatbot, vision_msg]
            )

            # Clear button for vision
            vision_clear_btn.click(lambda: ([], ""), outputs=[vision_chatbot, vision_msg])

        # Launch the interface
        self.interface.launch(share=True)


if __name__ == "__main__":
    # Create model manager
    model_manager = ModelManager(config_path="../config/models.json")

    # Create and launch UI
    ui = ChatUI(model_manager)
    ui.launch()