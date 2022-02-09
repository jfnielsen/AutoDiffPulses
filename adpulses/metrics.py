from typing import Optional
from torch import Tensor


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
    lam1 = 0.0 # 12/6/21: 1.0.  # mxy magnitude error weighting
    lam2 = 20.0 # 12/6/21: 2.0   # mxy complex error weighting
    lam3 = 0.0                  # mz error weighting

    Me_ = Mr_[..., :2].norm(dim=-1) - Md_[..., :2].norm(dim=-1)
    errmag = (Me_ if w_ is None else Me_*w_).norm()**2

    Me_ = (Mr_[..., :2] - Md_[..., :2])
    errcplx = (Me_ if w_ is None else Me_*w_[..., None]).norm()**2

    Mez_ = (Mr_[..., 2] - Md_[..., 2])  # (1, nM)
    errmz = (Mez_ if w_ is None else Mez_*w_).norm()**2

    err = lam1 * errmag + lam2 * errcplx + lam3 * errmz

    return err


