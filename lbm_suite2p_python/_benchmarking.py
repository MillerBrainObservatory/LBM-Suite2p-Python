import psutil
import subprocess


def get_cpu_percent():
    return psutil.cpu_percent(interval=None)


def get_ram_used():
    return psutil.virtual_memory().used // (1024 * 1024)


def get_gpu_stats():
    """Return (gpu_util%, mem_used_MiB) for GPU 0 (or zeros if unavailable)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        util, mem = result.stdout.strip().split("\n")[0].split(", ")
        return int(util), int(mem)
    except Exception:
        return 0, 0
