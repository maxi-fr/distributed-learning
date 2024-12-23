
import torch



class LARS(torch.optim.Optimizer):

    def __init__(self, params, defaults):
        super().__init__(params, defaults)


    def step(self):
        pass
        # update params

class LAMB(torch.optim.Optimizer):

    def __init__(self, params, defaults):
        super().__init__(params, defaults)


    def step(self):
        pass
        # update params