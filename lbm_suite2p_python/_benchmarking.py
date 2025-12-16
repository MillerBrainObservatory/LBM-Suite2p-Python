import psutil


def get_cpu_percent():
    return psutil.cpu_percent(interval=None)


def get_ram_used():
    return psutil.virtual_memory().used // (1024 * 1024)
