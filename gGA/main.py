from gutzwiller import Gutzwiller
import torch


l = torch.eye(2)
l[1,1] = -1
gz = Gutzwiller(
    norb=2,
    t=-torch.eye(4).reshape(1,2,2,2,2),
    U=1.5,
    Up=0.,
    J=0.,
    Jp=0.,
    l=l, 
    R=torch.tensor([[0,0,0]]), 
    kpoints=torch.tensor([[0.,0.,0.]]), 
    kspace=True, 
    Nocc=2,
)


print(gz.forward())


if __name__ == "__main__":
    pass