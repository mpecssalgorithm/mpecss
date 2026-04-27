from typing import Any, Dict, Optional


def log_peak_memory() -> float:
    # Returns peak RSS in MB.
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return 0.0


def log_gpu_memory() -> Optional[float]:
    # Returns current GPU memory usage in MB if NVIDIA GPU available.
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            total_mb = sum(float(line.strip()) for line in lines if line.strip())
            return total_mb
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def check_gpu_available() -> bool:
    # Check if NVIDIA GPU is available for computation.
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_system_info() -> Dict[str, Any]:
    # Get system information for benchmark logging.
    import platform
    import os

    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'total_ram_gb': None,
        'gpu_name': None,
        'gpu_memory_gb': None,
    }

    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['total_ram_gb'] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(',')
            if len(parts) >= 2:
                info['gpu_name'] = parts[0].strip()
                info['gpu_memory_gb'] = float(parts[1].strip()) / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return info
