import numpy as np
import pytest
from gpaw.spinorbit import get_L_vlmm


class QuantumState:
    """Quantum State, representing real Y_lm or complex Y_l^m"""
    def __init__(self, l: int, m: int, factor: complex = 1.0):
        assert type(l) is int
        assert type(m) is int
        self.l = l
        self.m = m
        self.factor = factor


def real_to_complex_quantum_state(state: QuantumState):
    """represent the real Y_lm in complex Y_l^m"""
    if state.m == 0:
        return [QuantumState(state.l, state.m, 1.0)]
    elif state.m > 0:
        return [
            QuantumState(state.l, -1 * state.m,
                         1 / np.sqrt(2.0) * state.factor),
            QuantumState(state.l, state.m,
                         (-1.0) ** state.m / np.sqrt(2.0) * state.factor)]
    else:  # state.m < 0:
        return [
            QuantumState(state.l, state.m, 1j / np.sqrt(2.0) * state.factor),
            QuantumState(
                state.l,
                -1 * state.m,
                -1j * (-1.0) ** state.m / np.sqrt(2.0) * state.factor,
            ),
        ]


def matrix_element(bra: QuantumState, ket: QuantumState, operator):
    """<Y_lm' | O | Y_lm >"""
    assert bra.l == ket.l
    result = 0.0
    kets_complex = real_to_complex_quantum_state(ket)
    bras_complex = real_to_complex_quantum_state(bra)
    for b in bras_complex:
        for k in kets_complex:
            operator_factor, newk = operator(k)
            if newk.m != b.m:
                continue
            result += np.conj(b.factor) * newk.factor * operator_factor
    return result


def Lz(ket: QuantumState):
    """L_z acting on complex Y_l^m"""
    return ket.m, QuantumState(ket.l, ket.m, ket.factor)


def Lp(ket: QuantumState):
    """L_+ acting on complex Y_l^m"""
    if ket.m < ket.l:
        result = np.sqrt(ket.l * (ket.l + 1) - ket.m * (ket.m + 1))
    else:
        result = 0
    return result, QuantumState(ket.l, ket.m + 1, ket.factor)


def Lm(ket: QuantumState):
    """L_- acting on complex Y_l^m"""
    if ket.m > -ket.l:
        result = np.sqrt(ket.l * (ket.l + 1) - ket.m * (ket.m - 1))
    else:
        result = 0
    return result, QuantumState(ket.l, ket.m - 1, ket.factor)


def test_lvlmm():

    L_vlmm = get_L_vlmm()

    for l in [0, 1, 2, 3]:
        # print(f"Doing {'spdf'[l]}:")
        result_lz = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
        result_lp = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
        result_lm = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
        for mp in range(-l, l + 1):
            bra = QuantumState(l, mp)  # real Y_lm
            for m in range(-l, l + 1):
                ket = QuantumState(l, m)  # real Y_lm
                result_lz[mp + l, m + l] = matrix_element(bra, ket, Lz)
                result_lp[mp + l, m + l] = matrix_element(bra, ket, Lp)
                result_lm[mp + l, m + l] = matrix_element(bra, ket, Lm)

        result_lx = (result_lp + result_lm) / 2.0
        result_ly = (result_lp - result_lm) / 2j

        L_vlmm[0][l] == pytest.approx(result_lx)
        L_vlmm[1][l] == pytest.approx(result_ly)
        L_vlmm[2][l] == pytest.approx(result_lz)
