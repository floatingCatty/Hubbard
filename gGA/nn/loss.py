import torch
import torch.nn as nn
from gGA.data import AtomicDataDict, _keys

# class GhostLoss(nn.Module):
    # def __init__(self):
    #     super(GhostLoss, self).__init__()
    #     self.bdiag = torch.vmap(torch.diag)

    # def forward(self, data: AtomicDataDict.Type, interact=True) -> AtomicDataDict.Type:
    #     T = data[_keys.TOTAL_ENERGY_KEY]
    #     U = data[_keys.INTERACTION_ENERGY_KEY]
    #     variational_density = data[_keys.VARIATIONAL_DENSITY_KEY]
    #     reduced_density_matrix = data[_keys.REDUCED_DENSITY_MATRIX_KEY]
    #     # R = data[_keys.R_MATRIX_KEY]
    #     # RTR = torch.bmm(R.transpose(1,2), R)

    #     loss = T+U

    #     count = 0
    #     count_ele = 0
    #     delta_density = 0
    #     Ntotal_ansatz = 0
    #     for i in range(len(data[_keys.ATOM_TYPE_KEY].flatten())):
    #         rdm = variational_density[i]
    #         mask = ~rdm.eq(0)
    #         count_ele += rdm[mask].numel()
    #         delta_density += ((reduced_density_matrix[count:count+rdm.shape[0], count:count+rdm.shape[0]][mask] - rdm[mask])**2).sum()
    #         count += rdm.shape[0]
    #         Ntotal_ansatz += rdm.trace()
        
    #     loss = loss + 50 * (delta_density / count_ele).sqrt()

    #     return loss

class GhostLoss(nn.Module):
    def __init__(self):
        super(GhostLoss, self).__init__()
        self.mu = 1.0

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        T = data[_keys.TOTAL_ENERGY_KEY]
        U = data[_keys.INTERACTION_ENERGY_KEY]

        return T+U

class GhostLA(nn.Module):
    def __init__(self):
        super(GhostLA, self).__init__()
        self.bdiag = torch.vmap(torch.diag)

    # def forward(self, data: AtomicDataDict.Type, lagrangeD, lagrangeR) -> AtomicDataDict.Type:
        # variational_density = data[_keys.VARIATIONAL_DENSITY_KEY]
        # reduced_density_matrix = data[_keys.REDUCED_DENSITY_MATRIX_KEY]

        # R = data[_keys.R_MATRIX_KEY].clone()
        #R = R / R.norm(dim=1, keepdim=True)
        # RTR = torch.bmm(R.transpose(1,2), R)

        # Rdiag = self.bdiag(self.bdiag(RTR))
        # Roffdiag = (RTR - Rdiag).abs()

        # return lagrangeD * (variational_density - reduced_density_matrix).abs(), lagrangeR * Roffdiag
    
    def forward(self, data: AtomicDataDict.Type, lagrangian) -> AtomicDataDict.Type:

        variational_density = data[_keys.VARIATIONAL_DENSITY_KEY] # from Phi_i
        reduced_density_matrix = data[_keys.REDUCED_DENSITY_MATRIX_KEY] # from \Psi_0

        return torch.block_diag(*lagrangian) * (reduced_density_matrix - variational_density)


        

# class GhostCriteria(nn.Module):
#     def __init__(self):
#         super(GhostCriteria, self).__init__()
#         self.variational_density = 0

#     def forward(self, data: AtomicDataDict.Type) -> float:
#         if isinstance(self.variational_density, int):
#             self.variational_density = data[_keys.VARIATIONAL_DENSITY_KEY].detach().clone()
#             return 100
        
#         with torch.no_grad():

#             new_crit = (self.variational_density - data[_keys.VARIATIONAL_DENSITY_KEY]).abs().max()

#         if new_crit > 1e-11:
#             self.variational_density = data[_keys.VARIATIONAL_DENSITY_KEY].detach().clone()

#         return new_crit.item()

class GhostCriteria(nn.Module):
    def __init__(self):
        super(GhostCriteria, self).__init__()

    def forward(self, model) -> float:
        
        with torch.no_grad():

            # new_crit = min([(p * p.grad).sum().abs()/(p.norm()+p.grad.norm()) for ip, p in enumerate(model.parameters())])
            crit = []
            for p in model.parameters():
                if p.grad is not None:
                    crit.append(p.grad.abs().max().item())

        return max(crit)