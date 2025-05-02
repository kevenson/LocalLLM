import os
import sys
import argparse

# Ensure we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_manager import ModelManager
from ui.chat_ui import ChatUI
from utils.performance import monitor_gpu


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["models", "ui", "utils", "config"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="LLM Testing Platform")
    parser.add_argument(
        "--config",
        type=str,
        default="config/models.json",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage and use CPU only"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Load a specific model at startup"
    )
    parser.add_argument(
        "--monitor-gpu",
        action="store_true",
        help="Enable continuous GPU monitoring"
    )

    args = parser.parse_args()

    # Create directories if they don't exist
    setup_directories()

    # Start GPU monitoring if requested
    if args.monitor_gpu:
        monitor_gpu()

    # Create model manager
    model_manager = ModelManager(
        config_path=args.config,
        use_gpu=not args.no_gpu
    )

    # Preload model if specified
    if args.model:
        try:
            model_manager.load_model(args.model)
            print(f"Successfully loaded model: {args.model}")
        except Exception as e:
            print(f"Error loading model {args.model}: {str(e)}")

    # Launch the UI
    #ui = ChatUI(model_manager)
    #ui.launch()

    # Launch the UI
    from ui.minimal_chat_ui import MinimalChatUI
    ui = MinimalChatUI(model_manager)
    ui.launch()


if __name__ == "__main__":
    main()