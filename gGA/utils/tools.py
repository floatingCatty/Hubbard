import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from gGA.utils.constants import atomic_num_dict, anglrMId, SKBondType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import json
from pathlib import Path
import yaml
import torch.optim as optim
import logging
import random
from ase.neighborlist import neighbor_list
from ase.io.trajectory import Trajectory
import ase
import ssl
import os.path as osp
import urllib
import zipfile
import sys


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _OBJ = TypeVar("_OBJ")
    try:
        from typing import Literal  # python >3.6
    except ImportError:
        from typing_extensions import Literal  # type: ignore
    _ACTIVATION = Literal["relu", "relu6", "softplus", "sigmoid", "tanh", "gelu", "gelu_tf"]
    _PRECISION = Literal["default", "float16", "float32", "float64"]


def float2comlex(dtype):
    if isinstance(dtype, str):
        dtype =  getattr(torch, dtype)
    
    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        raise ValueError("the dtype is not supported! now only float64, float32 is supported!")
    return cdtype


def flatten_dict(dictionary):
    queue = list(dictionary.items())
    f_dict = {}
    while(len(queue)>0):
        k, v = queue.pop()
        if isinstance(v, dict) and len(v.items()) != 0:
            for it in v.items():
                queue.append((k+"-"+it[0], it[1]))
        else:
            f_dict.update({k:v})
    
    return f_dict

def reconstruct_dict(dictionary):
    queue = list(dictionary.items())
    r_dict = {}
    while(len(queue)>0):
        k,v = queue.pop()
        ks = k.split("-")
        s_dict = r_dict
        if len(ks) > 1:
            for ik in ks[:-1]:
                if not s_dict.get(ik, None):
                    s_dict.update({ik:{}})
                s_dict = s_dict[ik]
        s_dict[ks[-1]] = v
    
    return r_dict


def checkdict(dict_prototype, dict_update, checklist):
    flatten_dict_prototype = flatten_dict(dict_prototype)
    flatten_dict_update = flatten_dict(dict_update)
    for cid in checklist:
        if flatten_dict_prototype.get(cid) != flatten_dict_update.get(cid):
            log.error(msg="the {0} in input config is not align with it in checkpoint.".format(cid))
            raise ValueError("the {0} in input config is not align with it in checkpoint.".format(cid))

    return True
    
def update_dict(temp_dict, update_dict, checklist):
    '''
        temp_dict: the dict that need to be update, and some of its value need to be checked in case of wrong update
        update_dict: the dict used to update the templete dict
    '''
    flatten_temp_dict = flatten_dict(temp_dict)
    flatten_update_dict = flatten_dict(update_dict)

    for cid in checklist:
        if flatten_update_dict.get(cid) != flatten_temp_dict.get(cid):
            raise ValueError
    
    flatten_temp_dict.update(flatten_update_dict)
    
    return reconstruct_dict(flatten_temp_dict)

def update_dict_with_warning(dict_input, update_list, update_value):
    flatten_input_dict = flatten_dict(dict_input)

    for cid in update_list:
        idx = update_list.index(cid)
        if flatten_input_dict[cid] != update_value[idx]:
            log.warning(msg="Warning! The value {0} of input config has been changed.".format(cid))
            flatten_input_dict[cid] = update_value[idx]
    
    return reconstruct_dict(flatten_input_dict)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def j_must_have(
    jdata: Dict[str, "_DICT_VAL"], key: str, deprecated_key: List[str] = []
) -> "_DICT_VAL":
    """Assert that supplied dictionary conaines specified key.

    Returns
    -------
    _DICT_VAL
        value that was store unde supplied key

    Raises
    ------
    RuntimeError
        if the key is not present
    """
    if key not in jdata.keys():
        for ii in deprecated_key:
            if ii in jdata.keys():
                log.warning(f"the key {ii} is deprecated, please use {key} instead")
                return jdata[ii]
        else:
            raise RuntimeError(f"json database must provide key {key}")
    else:
        return jdata[key]

    
def get_uniq_symbol(atomsymbols):
    '''>It takes a list of atomic symbols and returns a list of unique atomic symbols in the order of
    atomic number
    
    Parameters
    ----------
    atomsymbols
        a list of atomic symbols, e.g. ['C', 'C','H','H',...]
    
    Returns
    -------
        the unique atom types in the system, and the types are sorted descending order of atomic number.
    
    '''
    atomic_num_dict_r = dict(zip(atomic_num_dict.values(), atomic_num_dict.keys()))
    atom_num = []
    for it in atomsymbols:
        atom_num.append(atomic_num_dict[it])
    # uniq and sort.
    uniq_atom_num = sorted(np.unique(atom_num), reverse=True)
    # assert(len(uniq_atom_num) == len(atomsymbols))
    uniqatomtype = []
    for ia in uniq_atom_num:
        uniqatomtype.append(atomic_num_dict_r[ia])

    return uniqatomtype

def format_readline(line):
    '''In  SK files there are like 5*0 to represent the 5 0 zeros, 0 0 0 0 0. Here we replace the num * values 
    into the format of number of values.
    
    Parameters
    ----------
    line
        the line of text to be formatted
    
    Returns
    -------
        A list of strings.
    
    '''
    lsplit = re.split(',|;| +|\n|\t',line)
    lsplit = list(filter(None,lsplit))
    lstr = []
    for ii in range(len(lsplit)):
        strtmp = lsplit[ii]
        if re.search('\\*',strtmp):
            strspt = re.split('\\*|\n',strtmp)
            strspt = list(filter(None,strspt))
            assert len(strspt) == 2, "The format of the line is not correct! n*value, the value is gone!"
            strfull = int(strspt[0]) * [strspt[1]]
            lstr +=strfull
        else:
            lstr += [strtmp]
    return lstr

def j_loader(filename: Union[str, Path]) -> Dict[str, Any]:
    """Load yaml or json settings file.

    Parameters
    ----------
    filename : Union[str, Path]
        path to file

    Returns
    -------
    Dict[str, Any]
        loaded dictionary

    Raises
    ------
    TypeError
        if the supplied file is of unsupported type
    """
    filepath = Path(filename)
    if filepath.suffix.endswith("json"):
        with filepath.open() as fp:
            return json.load(fp)
    elif filepath.suffix.endswith(("yml", "yaml")):
        with filepath.open() as fp:
            return yaml.safe_load(fp)
    else:
        raise TypeError("config file must be json, or yaml/yml")



def LorentzSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function

        Delta(x) = lim_{sigma to 0}  Lorentzian
    '''

    return 1. / np.pi * sigma**2 / ((x - x0)**2 + sigma**2)


def GaussianSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function

        Delta(x) = lim_{sigma to 0} Gaussian
    '''

    return 1. / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - x0)**2 / (2*sigma**2))


def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def real_hermitian_basis(n):
    # the basis is composed by n diagnoal one hot and n*(n-1)/2 off-diagonal vector with two 1
    basis = []
    for i in range(n):
        bb = torch.zeros(n,n)
        bb[i,i] = 1
        basis.append(bb)
    
    for i in range(n-1):
        for j in range(i+1,n):
            bb = torch.zeros(n,n)
            bb[i,j] = 1
            bb[j,i] = 1
            basis.append(bb/bb.norm())

    return torch.stack(basis)

def real_hermitian_basis_spindeg(n):
    # the basis is composed by n diagnoal one hot and n*(n-1)/2 off-diagonal vector with two 1
    basis = []
    ndeg = n // 2
    for i in range(ndeg):
        bb = torch.zeros(ndeg,ndeg)
        bb[i,i] = 1
        basis.append(bb)
    
    for i in range(ndeg-1):
        for j in range(i+1,ndeg):
            bb = torch.zeros(ndeg,ndeg)
            bb[i,j] = 1
            bb[j,i] = 1
            basis.append(bb)


    basis = torch.stack(basis)
    basis = torch.stack([basis, torch.zeros_like(basis), torch.zeros_like(basis), basis]).reshape(2,2,-1,ndeg,ndeg).permute(2,3,0,4,1).reshape(-1,n,n)

    return basis / basis.norm(dim=(1,2), keepdim=True)

def real_hermitian_basis_spincollin(n):
    # the basis is composed by n diagnoal one hot and n*(n-1)/2 off-diagonal vector with two 1
    basis = []
    ndeg = n // 2
    for i in range(ndeg):
        bb = torch.zeros(ndeg,ndeg)
        bb[i,i] = 1
        basis.append(bb)
    
    for i in range(ndeg-1):
        for j in range(i+1,ndeg):
            bb = torch.zeros(ndeg,ndeg)
            bb[i,j] = 1
            bb[j,i] = 1
            basis.append(bb)
    
    basis = torch.stack(basis)
    basisup = torch.stack([basis, torch.zeros_like(basis), torch.zeros_like(basis), torch.zeros_like(basis)]).reshape(2,2,-1,ndeg,ndeg).permute(2,3,0,4,1).reshape(-1,n,n)
    basisdn = torch.stack([torch.zeros_like(basis), torch.zeros_like(basis), torch.zeros_like(basis), basis]).reshape(2,2,-1,ndeg,ndeg).permute(2,3,0,4,1).reshape(-1,n,n)
    basis = torch.stack([basisup, basisdn]).reshape(-1,n,n)

    return basis / basis.norm(dim=(1,2), keepdim=True)

def real_hermitian_basis_nspin(n, nspin):
    if nspin == 1:
        return real_hermitian_basis_spindeg(n)
    elif nspin == 2:
        return real_hermitian_basis_spincollin(n)
    elif nspin == 4:
        return real_hermitian_basis(n)
    else:
        raise ValueError("The nspin can only select value from 1, 2 and 4.")

def real_trans_basis(n1, n2):
    basis = []
    for i in range(n1):
        for j in range(n2):
            bb = torch.zeros(n1, n2)
            bb[i,j] = 1
            basis.append(bb)

    return torch.stack(basis)

def real_trans_basis_spindeg(n1, n2):
    n1deg = n1//2
    n2deg = n2//2
    basis = real_trans_basis(n1//2, n2//2)
    basis = torch.stack([basis, torch.zeros_like(basis), torch.zeros_like(basis), basis]).reshape(2,2,-1,n1deg,n2deg).permute(2,3,0,4,1).reshape(-1,n1,n2)

    return basis / basis.norm(dim=(1,2), keepdim=True)

def real_trans_basis_spincollin(n1, n2):
    n1deg = n1//2
    n2deg = n2//2
    basis = real_trans_basis(n1//2, n2//2)
    basisup = torch.stack([basis, torch.zeros_like(basis), torch.zeros_like(basis), torch.zeros_like(basis)]).reshape(2,2,-1,n1deg,n2deg).permute(2,3,0,4,1).reshape(-1,n1,n2)
    basisdn = torch.stack([torch.zeros_like(basis), torch.zeros_like(basis), torch.zeros_like(basis), basis]).reshape(2,2,-1,n1deg,n2deg).permute(2,3,0,4,1).reshape(-1,n1,n2)
    basis = torch.stack([basisup, basisdn]).reshape(-1,n1deg,n2deg)

    return basis / basis.norm(dim=(1,2), keepdim=True)

def real_trans_basis_nspin(n1, n2, nspin):
    if nspin == 1:
        return real_trans_basis_spindeg(n1, n2)
    elif nspin == 2:
        return real_trans_basis_spincollin(n1, n2)
    elif nspin == 4:
        return real_hermitian_basis(n1, n2)
    else:
        raise ValueError("The nspin can only select value from 1, 2 and 4.")