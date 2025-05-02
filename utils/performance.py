import torch
import time
import threading
import psutil
import os
import datetime


class GPUMonitor:
    """Monitors GPU performance metrics in a separate thread."""

    def __init__(self, interval=5.0, log_file="gpu_logs.csv"):
        """
        Initialize the GPU monitor.

        Args:
            interval: Monitoring interval in seconds
            log_file: Path to log file for recording metrics
        """
        self.interval = interval
        self.log_file = log_file
        self.running = False
        self.thread = None

        # Check if GPU is available
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.device_count = torch.cuda.device_count()
            self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
        else:
            self.device_count = 0
            self.device_names = []

    def start(self):
        """Start the monitoring thread."""
        if not self.has_gpu:
            print("No GPU available, monitoring disabled.")
            return

        if self.running:
            print("Monitoring already running.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True  # Make thread exit when main program exits
        self.thread.start()

        print(f"GPU monitoring started. Logging to {self.log_file}")

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            print("GPU monitoring stopped.")

    def _monitor_loop(self):
        """Main monitoring loop."""
        # Create log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                headers = ["timestamp", "device_id", "device_name", "memory_used_mb",
                           "memory_total_mb", "utilization_percent", "cpu_percent",
                           "system_memory_percent"]
                f.write(','.join(headers) + '\n')

        while self.running:
            try:
                # Get current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Get CPU and system memory usage
                cpu_percent = psutil.cpu_percent()
                system_memory = psutil.virtual_memory().percent

                # Log data for each GPU
                for device_id in range(self.device_count):
                    torch.cuda.set_device(device_id)

                    # Get memory usage in bytes and convert to MB
                    memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
                    memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)

                    # Calculate utilization
                    memory_used = memory_allocated  # memory_reserved can be used if needed
                    utilization = (memory_used / memory_total) * 100

                    # Log data
                    log_data = [
                        timestamp,
                        str(device_id),
                        self.device_names[device_id],
                        f"{memory_used:.2f}",
                        f"{memory_total:.2f}",
                        f"{utilization:.2f}",
                        f"{cpu_percent:.2f}",
                        f"{system_memory:.2f}"
                    ]

                    with open(self.log_file, 'a') as f:
                        f.write(','.join(log_data) + '\n')

                    # Print current status
                    print(f"GPU {device_id} ({self.device_names[device_id]}): "
                          f"{memory_used:.2f}MB / {memory_total:.2f}MB ({utilization:.2f}%) | "
                          f"CPU: {cpu_percent:.2f}% | System Memory: {system_memory:.2f}%")

            except Exception as e:
                print(f"Error in GPU monitoring: {str(e)}")

            # Wait for next interval
            time.sleep(self.interval)


def monitor_gpu(interval=5.0):
    """Start GPU monitoring in a separate thread."""
    if not torch.cuda.is_available():
        print("WARNING: Cannot monitor GPU - CUDA is not available")
        return None

    monitor = GPUMonitor(interval=interval)
    monitor.start()
    return monitor


def get_gpu_info():
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {
            "device_name": "CPU (CUDA not available)",
            "memory_allocated_gb": 0,
            "memory_reserved_gb": 0,
            "memory_total_gb": 0,
            "utilization_percent": 0,
            "warning": "GPU acceleration not available - using CPU only"
        }

    device_id = 0  # Default to first GPU
    torch.cuda.set_device(device_id)

    memory_allocated = torch.cuda.memory_allocated(device_id)
    memory_reserved = torch.cuda.memory_reserved(device_id)
    memory_total = torch.cuda.get_device_properties(device_id).total_memory

    return {
        "device_name": torch.cuda.get_device_name(device_id),
        "memory_allocated_gb": memory_allocated / (1024 ** 3),
        "memory_reserved_gb": memory_reserved / (1024 ** 3),
        "memory_total_gb": memory_total / (1024 ** 3),
        "utilization_percent": (memory_allocated / memory_total) * 100
    }


def optimize_for_inference(model):
    """Apply optimizations to a PyTorch model for faster inference."""
    # Set model to evaluation mode
    model.eval()

    # Disable gradient calculation
    for param in model.parameters():
        param.requires_grad = False

    return model


def estimate_memory_requirements(model_id):
    """
    Estimate memory requirements for a model based on its size.
    This is a rough estimate and actual usage may vary.
    """
    # TODO: Implement estimation based on model parameters and precision
    pass