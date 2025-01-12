import torch
import torch.nn as nn
from .operators import dm_up, dm_down, get_anni
from gGA.utils import kdsum, ksum

def slater_kanamori(norb, U, Up, J, Jp, l):
    dm_updown = dm_up @ dm_down
    diag_part = U * ksum(
        matrices=[dm_updown] * norb
        )
    
    off_diag_part = torch.zeros_like(diag_part)

    if abs(Up-0) > 1e-7:
        off_diag_part += (Up / 2) * kdsum(
            matricesA=[dm_up]*norb,
            matricesB=[dm_down]*norb
            )
        
        off_diag_part += (Up / 2) * kdsum(
            matricesA=[dm_down]*norb,
            matricesB=[dm_up]*norb
            )
        off_diag_part += (Up / 2) * kdsum(
            matricesA=[dm_up]*norb,
            matricesB=[dm_up]*norb
            )
        
        off_diag_part += (Up / 2) * kdsum(
            matricesA=[dm_down]*norb,
            matricesB=[dm_down]*norb
            )
    
    if abs(J-0) > 1e-7:
    
        off_diag_part -= (J / 2) * kdsum(
            matricesA=[get_anni(0).H.contiguous() @ get_anni(1)]*norb,
            matricesB=[get_anni(1).H.contiguous() @ get_anni(0)]*norb
            )
        
        off_diag_part -= (J / 2) * kdsum(
            matricesA=[get_anni(1).H.contiguous() @ get_anni(0)]*norb,
            matricesB=[get_anni(0).H.contiguous() @ get_anni(1)]*norb
            )
    
    if abs(Jp-0) > 1e-7:

        off_diag_part -= (Jp / 2) * kdsum(
            matricesA=[get_anni(0).H.contiguous() @ get_anni(1).H.contiguous()]*norb,
            matricesB=[get_anni(0) @ get_anni(1)]*norb
            )
    
    if isinstance(l, float) or isinstance(l, int):
        l = torch.eye(norb) * l

    if abs(l).max() > 1e-7:

        diag_part += ksum(
            matrices=[dm_up]*norb,
            coeff=l.diag()
        )

        diag_part += ksum(
            matrices=[dm_down]*norb,
            coeff=l.diag()
        )

        off_diag_part += kdsum(
            matricesA=[get_anni(0).H.contiguous()]*norb,
            matricesB=[get_anni(0)]*norb,
            coeff=l
            )
        
        off_diag_part += kdsum(
            matricesA=[get_anni(1).H.contiguous()]*norb,
            matricesB=[get_anni(1)]*norb,
            coeff=l
            )

    return diag_part + off_diag_part

