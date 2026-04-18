
import torch
class WarmupCosineScheduler:
    """Custom LR Scehdular
    """

    def __init__(self,
                 optimizer:torch.optim.Optimizer,
                 d_model:int,
                 warmup_steps:int=400):
        self.optimizer= optimizer
        self.d_model=d_model
        self.warmup_steps= warmup_steps
        self._step=0

    def step(self):
        self._step +=1
        lr= self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr']= lr

    def _compute_lr(self)->float:
        step= max(self._step,1)
        return(
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step *  self.warmup_steps ** (-1.5) )
        )
        
        