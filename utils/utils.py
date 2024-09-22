import torch


def device_cuda_mps_cpu():
    # seting up the device cuda, mps or cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device
