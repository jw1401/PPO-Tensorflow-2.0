from dataclasses import dataclass


@dataclass
class BehavioralCloningParams:

    batch_size_bc: int = 128
    iters_bc: int = 500
    lr: float = 0.001
