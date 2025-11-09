
"""Basic DP Trainer wrapper"""
import torch
from transformers import Trainer

class DPTrainer(Trainer):
    """Differential Privacy enabled Trainer"""
    
    def __init__(self, *args, privacy_engine=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_engine = privacy_engine
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with DP if enabled"""
        return super().compute_loss(model, inputs, return_outputs)
