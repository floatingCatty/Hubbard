from __future__ import annotations
from . import Operator

def _get_site_operator(
    index: tuple, nsites: int, opstr: str, strength: float = 1.0, is_fermion_down: bool = False
) -> Operator:
    if len(index) == 1 and 0 <= index[0] < nsites:
        index = int(index[0])

    # the spin orbital of quspin is defined as the first nsites for up spin and the next nsites for down spin, e.g.
    # for 2 orbital, nsite=2, the basis will looks like |up0, up1, down0, down1>
    if is_fermion_down:
        index += nsites
    return Operator([[opstr, [[strength, index]]]])


def create_u(nsites, *index) -> Operator:
    r"""
    :math:`c_↑^†` operator
    """
    return _get_site_operator(index, nsites, "+")


def create_d(nsites, *index) -> Operator:
    r"""
    :math:`c_↓^†` operator
    """
    return _get_site_operator(index, nsites, "+", is_fermion_down=True)


def annihilate_u(nsites, *index) -> Operator:
    r"""
    :math:`c_↑` operator
    """
    return _get_site_operator(index, nsites, "-")


def annihilate_d(nsites, *index) -> Operator:
    r"""
    :math:`c_↓` operator
    """
    return _get_site_operator(index, nsites, "-", is_fermion_down=True)


def number_u(nsites, *index) -> Operator:
    r"""
    :math:`n_↑ = c_↑^† c_↑` operator
    """
    return _get_site_operator(index, nsites, "n")


def number_d(nsites, *index) -> Operator:
    r"""
    :math:`n_↓ = c_↓^† c_↓` operator
    """
    return _get_site_operator(index, nsites, "n", is_fermion_down=True)


def S_z(nsites, *index) -> Operator:
    r"""
    :math:`S^z` operator

    .. math::

        S^z =
        \begin{pmatrix}
            1/2 & 0 \\
            0 & -1/2
        \end{pmatrix}
    """
    return 0.5 * number_u(nsites, *index) - 0.5 * number_d(nsites, *index)


def S_p(nsites, *index) -> Operator:
    r"""
    :math:`S^+` operator

    .. math::

        S^+ =
        \begin{pmatrix}
            0 & 1 \\
            0 & 0
        \end{pmatrix}
    """
    return create_u(nsites, *index) * annihilate_d(nsites, *index)


def S_m(nsites, *index) -> Operator:
    r"""
    :math:`S^-` operator

    .. math::

        S^- =
        \begin{pmatrix}
            0 & 0 \\
            1 & 0
        \end{pmatrix}
    """
    return create_d(nsites, *index) * annihilate_u(nsites, *index)
