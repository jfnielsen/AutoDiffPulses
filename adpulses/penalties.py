import torch
from torch import Tensor


def pen_null(rf: Tensor) -> Tensor:
    """
    *INPUTS*
    - `rf`  (1, xy, nT, (nCoils))
    *OUTPUTS*
    - `pen` (1,)
    """
    return rf.new_zeros([])


def pen_l2_rf(rf: Tensor) -> Tensor:
    pen = torch.norm(rf)**2
    return pen


def pen_l2(rf: Tensor, g: Tensor) -> Tensor:
    gmax = 5    # G/cm
    smax = 20   # G/cm/ms
    dt = 10e-3   # ms

    a, b, c = 10.0, 1e-4, 1e-1  # loss term weights

    # RF energy penalty
    standardPulse_energy = 1.3791e-5   # Energy of 1 ms hard 180 (Gauss^2*sec)
    rf_energy = dt*1e-3 * torch.norm(rf)**2
    pen_rf = torch.pow(rf_energy/standardPulse_energy, 8)

    # max gradient penalty (soft thresholding)
    pen_gmax_v = torch.pow(torch.relu(g.abs()-gmax), 2)
    pen_gmax = torch.sum(pen_gmax_v)

    # slew rate penalty
    slew = torch.diff(g)/dt
    pen_slew_v = torch.pow(torch.relu(slew.abs()-smax), 2)
    pen_slew= torch.sum(pen_slew_v)

    pen = a*pen_rf + b*pen_gmax + c*pen_slew

    # print(f"pen_rf, pen_gmax, pen_slew = {pen_rf}, {pen_gmax}, {pen_slew}")

    return pen
