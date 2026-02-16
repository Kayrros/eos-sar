from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from teosar import periodogram
from teosar.periodogram_cl import PeriodogramCL, create_constants, create_variables
from teosar.periodogram_par import PeriodogramPar, PeriodogramTF


@dataclass(frozen=True)
class ThreeParamSimu:
    phi_ps: list[float]
    """
    phi_ps_mat = wrap(Btime * Cv * v_gt + Bperp * Cq * q_gt +  dtemperature * C_eta * eta_gt + noise)
    """
    Btime: list[float]
    Bperp: list[float]
    dtemperature: list[float]
    Cv: float
    Cq: float
    Ceta: float
    v_gt: float
    q_gt: float
    eta_gt: float
    phi_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(dico: dict[str, Any]) -> ThreeParamSimu:
        return ThreeParamSimu(
            dico["phi_ps"],
            dico["Btime"],
            dico["Bperp"],
            dico["dtemperature"],
            dico["Cv"],
            dico["Cq"],
            dico["Ceta"],
            dico["v_gt"],
            dico["q_gt"],
            dico["eta_gt"],
            dico["phi_std"],
        )


# %%
def test_simulate_then_fit():
    with open("./tests/data/threeparamsimu.json", "r") as f:
        dico = json.load(f)

    three_param_simu = ThreeParamSimu.from_dict(dico)

    phi_ps_mat = np.array(three_param_simu.phi_ps, dtype=np.float64).reshape(-1, 1)
    num_dates, num_PS = phi_ps_mat.shape
    Bperp = np.array(three_param_simu.Bperp, dtype=np.float64)
    Btime = np.array(three_param_simu.Btime, dtype=np.float64)
    dtemperature = np.array(three_param_simu.dtemperature, dtype=np.float64)

    weights = np.ones((num_dates,), dtype=np.float64) / num_dates

    # Convert inputs to the format PeriodogramCL expects.
    constants = create_constants(
        num_PS,
        num_dates,
        phi_ps_mat.T,
        [
            -three_param_simu.Cq * Bperp,
            -three_param_simu.Cv * Btime,
            -three_param_simu.Ceta * dtemperature,
        ],
        dtype=np.float64,
    )

    v_test = periodogram.get_test_vals(300, 12)
    q_test = periodogram.get_test_vals(80, 12)
    eta_test = np.linspace(0, 10, 23)

    variables = create_variables([q_test, v_test, eta_test], dtype=np.float64)

    periodo_cl = PeriodogramCL(
        enable_profile=False,
        num_constants_per_sum_term=4,
        interactive_device_selection=False,
    )

    periodo_tf = PeriodogramTF(num_constants_per_sum_term=4, batch_size=1)

    periodo_par = PeriodogramPar(periodo_cl, periodo_tf)
    opt_vars, gammas = periodo_par.find_maximum(constants, variables, weights)

    q_estimated = opt_vars[:, 0]
    v_estimated = opt_vars[:, 1]
    eta_estimated = opt_vars[:, 2]

    v_err = np.abs(v_estimated - three_param_simu.v_gt)
    q_err = np.abs(q_estimated - three_param_simu.q_gt)
    eta_err = np.abs(eta_estimated - three_param_simu.eta_gt)

    theo_prec_v = three_param_simu.phi_std / (
        np.abs(three_param_simu.Cv) * np.std(Btime) * np.sqrt(num_dates)
    )
    theo_prec_q = three_param_simu.phi_std / (
        np.abs(three_param_simu.Cq) * np.std(Bperp) * np.sqrt(num_dates)
    )
    theo_prec_eta = three_param_simu.phi_std / (
        np.abs(three_param_simu.Ceta) * np.std(dtemperature) * np.sqrt(num_dates)
    )

    assert v_err[0] < 3 * theo_prec_v
    assert q_err[0] < 3 * theo_prec_q
    assert eta_err[0] < 3 * theo_prec_eta
