# Utility functions for LoRA
import torch
import loralib as lora
import math
import torch.nn as nn

def reset_lora_weights(model):
    '''
    Reset all low-rank weights in a model modified by LoRA.
    '''
    for module in model.modules():
        if isinstance(module, lora.LoRALayer):  # Check if the module is a LoRA layer
            if module.r > 0:  # If LoRA is active for this layer
                # Reset low-rank weights
                # module.reset_parameters() # this also reset the base weights
                
                # only reset lora weights
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)




def merge_lora_weights(model):
    '''
    Merge LoRA weights into the base weights of the model.
    '''
    for module in model.modules():
        if isinstance(module, lora.LoRALayer):  # Check if the module is a LoRA layer
            if module.r > 0:  # If LoRA is active for this layer
                # Merge low-rank weights into the base weights
                module.weight.data += (module.lora_B @ module.lora_A) * module.scaling
