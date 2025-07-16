import sys
import subprocess
import psutil
from pynvml import *


def get_system_memory():
   memory = psutil.virtual_memory()
   print(f"\nTotal system memory (RAM): {memory.total / (1024 ** 3):.2f} GB \n")


def get_cuda_toolkit_version():
    """
    Attempts to get the CUDA Toolkit version by running 'nvcc --version'
    or checking the CUDA_HOME environment variable.
    """
    try:
        # Try running nvcc --version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Cuda compilation tools" in line:
                return line.strip()
        return "Not found via nvcc --version"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to CUDA_HOME environment variable
        import os
        cuda_home = os.getenv('CUDA_HOME')
        if cuda_home:
            return f"CUDA_HOME environment variable set: {cuda_home}"
        return "CUDA Toolkit not found (nvcc not in PATH or CUDA_HOME not set)"


def bytes_to_gb(bytes_value):
    """Converts bytes to gigabytes."""
    return f"{bytes_value / (1024**3):.2f} GB"


def get_gpu_info():
    """
    Retrieves and prints detailed information about NVIDIA GPUs
    and checks for CUDA configuration.
    """

    print("NVIDIA GPU and CUDA Configuration Check:")

    try:
        nvmlInit()  # Initialize NVML
        print(f"  NVIDIA Driver Version: {nvmlSystemGetDriverVersion()}")
        
        # Get CUDA Driver Version (from NVML)
        # This indicates the CUDA version the installed driver supports.
        try:
            cuda_driver_version = nvmlSystemGetCudaDriverVersion()
            # The version is returned as (major, minor) integers, convert to string
            print(f"  CUDA Driver Version (from NVML): {cuda_driver_version}")   # {cuda_driver_version[1]}")
        except NVMLError as error:
            print(f"  CUDA Driver Version (from NVML): Not available ({error})")

        # Get CUDA Toolkit Version (from system)
        print(f"  CUDA Toolkit Version: {get_cuda_toolkit_version()}")

        device_count = nvmlDeviceGetCount()
        print(f"\nFound {device_count} NVIDIA GPU(s):\n")

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            print(f"--- GPU {i} ---")
            
            # GPU Name
            print(f"  Name: {nvmlDeviceGetName(handle)}")

            # Memory Info
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            print(f"  Memory Total: {bytes_to_gb(mem_info.total)}")
            print(f"  Memory Used: {bytes_to_gb(mem_info.used)}")
            print(f"  Memory Free: {bytes_to_gb(mem_info.free)}")

            # Temperature
            try:
                temp = nvmlDeviceGetTemperature(handle, 0) #, NVML_TEMPERATURE_GPU = 0)

                print(f"  Temperature: {temp}°C")
            except NVMLError as error:
                print(f"  Temperature: Not available ({error})")

            # Power Usage
            try:
                power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                power_limit = nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0 # Convert mW to W
                print(f"  Power Usage: {power_usage:.2f} W / {power_limit:.2f} W")
            except NVMLError as error:
                print(f"  Power Usage: Not available ({error})")

            # Fan Speed
            try:
                fan_speed = nvmlDeviceGetFanSpeed(handle)
                print(f"  Fan Speed: {fan_speed}%")
            except NVMLError as error:
                print(f"  Fan Speed: Not available ({error})")

            # Utilization
            try:
                utilization = nvmlDeviceGetUtilizationRates(handle)
                print(f"  GPU Utilization: {utilization.gpu}%")
                print(f"  Memory Utilization: {utilization.memory}%")
            except NVMLError as error:
                print(f"  Utilization: Not available ({error})")
            
            print()

    except NVMLError as error:
        print(f"Error initializing NVML: {error}")
        print("Please ensure NVIDIA drivers are installed and running.")
        print("On Linux, you might need to run 'sudo apt install nvidia-driver-xxx' (replace xxx with your desired version).")
    finally:
        try:
            nvmlShutdown() # Shutdown NVML
        except NVMLError as error:
            print(f"Error shutting down NVML: {error}")
    
