from typing import Optional
from torch import Tensor
import torch


def err_null(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    return Mr_.new_zeros([])


def err_l2z(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = (Mr_[..., 2] - Md_[..., 2])  # (1, nM)
    err = (Me_ if w_ is None else Me_*w_).norm()**2
    return err


def err_l2xy(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = (Mr_[..., :2] - Md_[..., :2])
    err = (Me_ if w_ is None else Me_*w_[..., None]).norm()**2
    return err


# hijacking ml2xy for prephasing problem
def err_ml2xy(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    lam1 = 1.0 # 12/6/21: 1.0.  # mxy magnitude error weighting (enforce limits)
    lam2 = 2.0 # 12/6/21: 2.0   # mxy complex error weighting
    lam3 = 0.0                  # mz error weighting
    lam4 = 0.0                  # flip angle error weighting

    #print('Mr_.size(): ', Mr_.size())

    Me_ = Mr_[..., :2].norm(dim=-1) - Md_[..., :2].norm(dim=-1)
    #print('Me_.size(): ', Me_.size())
    #errmag = (Me_ if w_ is None else Me_*w_).norm()**2
    thresh = 0.2;  # max allowable difference in sin(flip)
    errmag = (torch.relu((Me_.abs() - thresh)/thresh)).norm()**2

    Me_ = (Mr_[..., :2] - Md_[..., :2])
    errcplx = (Me_ if w_ is None else Me_*w_[..., None]).norm()**2

    Mez_ = (Mr_[..., 2] - Md_[..., 2])  # (1, nM)
    errmz = (Mez_ if w_ is None else Mez_*w_).norm()**2
    #errmz = torch.pow(torch.relu( Mr_[..., 2] - Md_[..., 2]  - 0.02) + torch.relu(Mr_[..., 2] - Md_[..., 2] + 0.02), 2)
    #print('Mez_.type(): ', Mez_.type())
    #print('Mez_.size(): ', Mez_.size())
    #errmz = torch.relu( Mez_ - 0.02) + torch.relu(Mez_ + 0.02)

    #errflip = torch.pow(torch.relu( Me_ ) + torch.relu(Me_ + 0.02), 2)
    #flipr_ = (Mr_[..., 0] + 1j*Mr_[...,1]).abs()    # (1, nM) sin(flip)
    #flipd_ = (Md_[..., 0] + 1j*Md_[...,1]).abs()    # (1, nM)  sin(flip) (target)
    #Meflip_=  Mrc_ - Mdc_

    err = lam1 * errmag + lam2 * errcplx + lam3 * errmz # + lam4 * errflip

    return err


