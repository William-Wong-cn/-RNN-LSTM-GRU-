# encoding-utf-8
def print_device():
    import torch

    print("使用设备:", "CUDA" if torch.cuda.is_available() else "CPU")
