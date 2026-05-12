import torch


def get_device(device: torch.device | str | None = None) -> torch.device:
    """Get the device to use for computations.
    
    Args:
        device: Device specification. Can be:
            - None or "auto": Automatically select cuda if available, else cpu
            - "cuda", "cpu", "mps": Specific device type
            - torch.device instance: Return as-is
            
    Returns:
        The resolved torch.device.
    """
    def _validate_device(resolved: torch.device) -> torch.device:
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available on this machine.")
        if resolved.type == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise ValueError("MPS device requested but MPS is not available on this machine.")
        return resolved

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        return _validate_device(device)
    elif isinstance(device, str):
        return _validate_device(torch.device(device))
    else:
        raise ValueError(f"Invalid device: {device}")