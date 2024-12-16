import pytest
from gGA.nn.ansatz import gGASingleOrb, gGAMultiOrb, gGAtomic
import torch

class TestgGASingleOrb: 
    def test_1B_1o(self):
        gs = gGASingleOrb(
            norb=1,
            naux=1,
            intparam={
                "t": torch.eye(2),
                "U": 3.,
                "J": 0.,
                "Jp": 0,
                "Up": 3,
            },
            nspin=1,
        )