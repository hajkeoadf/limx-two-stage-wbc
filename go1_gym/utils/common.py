from yapf.yapflib.yapf_api import FormatCode
import torch
import random
import numpy as np
import os
import sys
import select


def input_with_timeout(timeout):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        s = sys.stdin.readline()
        try:
            return s.strip()
        except:
            return None
    else:
        return None


def clean_dict_for_formatting(obj):
    """
    Recursively clean dictionary to make it serializable for formatting.
    Converts non-serializable objects (classes, tensors, etc.) to string representations.
    """
    if isinstance(obj, dict):
        return {key: clean_dict_for_formatting(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(clean_dict_for_formatting(item) for item in obj)
    elif isinstance(obj, torch.Tensor):
        # Convert tensor to a simple string representation
        if obj.numel() == 1:
            return f"tensor({obj.item()})"
        else:
            return f"tensor(shape={list(obj.shape)})"
    elif isinstance(obj, type):
        # Handle class objects
        return f"<class '{obj.__module__}.{obj.__name__}'>"
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Basic serializable types
        return obj
    else:
        # For other non-serializable objects, use their string representation
        class_name = obj.__class__.__name__
        module_name = getattr(obj.__class__, '__module__', 'unknown')
        return f"<{class_name} object from {module_name}>"


def format_code(code_text: str):
    """Format the code text with yapf."""
    yapf_style = dict(
        based_on_style='pep8',
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True)
    try:
        code_text, _ = FormatCode(code_text, style_config=yapf_style)
    except:  # noqa: E722
        # If formatting fails (e.g., due to tensor objects in config), just return the original string
        # This is acceptable since the pickled file will have the correct format
        pass

    return code_text

def quaternion_to_rpy(quaternions):
    """
    Note:
        rpy (torch.Tensor): Tensor of shape (N, 3). Range: (-pi, pi)
    """
    assert quaternions.shape[1] == 4, "Input should have shape (N, 4)"
    
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    rpy = torch.zeros((quaternions.shape[0], 3), device=quaternions.device, dtype=quaternions.dtype)
    
    # Compute Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    rpy[:, 0] = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Compute Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    rpy[:, 1] = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.tensor(torch.pi/2, device=quaternions.device, dtype=quaternions.dtype), torch.asin(sinp))
    
    # Compute Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    rpy[:, 2] = torch.atan2(siny_cosp, cosy_cosp)
    
    return rpy

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed