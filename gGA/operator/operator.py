from __future__ import annotations
from typing import Optional, Tuple, Union
from numbers import Number
from functools import partial
import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_general
from scipy.special import comb


class Operator:
    """Quantum operator"""

    def __init__(self, op_list: list):
        """
        :param op_list:
            The operator represented as a list in the
            `QuSpin format <https://quspin.github.io/QuSpin/generated/quspin.operators.hamiltonian.html#quspin.operators.hamiltonian.__init__>`_

            ``[[opstr1, [strength1, index11, index12, ...]], [opstr2, [strength2, index21, index22, ...]], ...]``

                opstr:
                    a `string <https://quspin.github.io/QuSpin/basis.html>`_ representing the operator type.
                    The convention is chosen the same as ``pauli=0`` in
                    `QuSpin <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.__init__>`_

                strength:
                    interaction strength

                index:
                    the site index that operators act on
        """
        self._op_list = op_list
        self._quspin_op = dict()
        self._quspin_basis = dict()
        self._connectivity = None
        self._is_basis_made = False

    @property
    def op_list(self) -> list:
        """Operator represented as a list in the QuSpin format"""
        return self._op_list

    @property
    def expression(self) -> str:
        """The operator as a human-readable expression"""
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        OP = str.maketrans({"x": "Sˣ", "y": "Sʸ", "z": "Sᶻ", "+": "S⁺", "-": "S⁻"})
        expression = []
        for opstr, interaction in self.op_list:
            for J, *index in interaction:
                expression.append(f"{J:+}")
                for op, i in zip(opstr, index):
                    expression.append(f"{op.translate(OP)}{str(i).translate(SUB)}")
        return " ".join(expression)

    def __repr__(self) -> str:
        return self.expression

    def make_basis(self, nsites, Nparticle):

        key = str(nsites) + "-" + str(Nparticle)
        if self._quspin_basis.get(key) is not None:
            self._basis = self._quspin_basis[key]
        else:
            self._basis = spinful_fermion_basis_general(
                nsites,
                Nparticle,
                simple_symm=False,
                make_basis=True
            )
            self._quspin_basis[key] = self._basis

    @property
    def basis(self):
        return self._basis

    def get_quspin_op(self, nsites, Nparticle, iscomplex=False) -> hamiltonian:
        """
        Obtain the corresponding
        `QuSpin operator <https://quspin.github.io/QuSpin/generated/quspin.operators.hamiltonian.html#quspin.operators.hamiltonian.__init__>`_

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        self.make_basis(nsites, Nparticle)
        
        key = str(nsites) + "-" + str(Nparticle)

        if iscomplex:
            dtype = np.complex128
        else:
            dtype = np.float64

        if key not in self._quspin_op:
            self._quspin_op[key] = hamiltonian(
                static_list=self.op_list,
                dynamic_list=[],
                basis=self.basis,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
                dtype=dtype,
            )

        return self._quspin_op[key]

    def todense(self, nsites, Nparticle, iscomplex=False) -> np.ndarray:
        """
        Obtain the dense matrix representing the operator

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        quspin_op = self.get_quspin_op(nsites, Nparticle, iscomplex=iscomplex)
        return quspin_op.as_dense_format()

    def tosparse(self, nsites, Nparticle, iscomplex=False) -> np.ndarray:
        """
        Obtain the sparse matrix representing the operator

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        quspin_op = self.get_quspin_op(nsites, Nparticle, iscomplex=iscomplex)
        return quspin_op.as_sparse_format()

    def __array__(self) -> np.ndarray:
        return self.todense()

    def diagonalize(
        self,
        nsites, 
        Nparticle,
        k: Union[int, str] = 1,
        iscomplex=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the hamiltonian :math:`H = V D V^†`

        :param symm:
            Symmetry for generating basis.
        :param k:
            A number specifying how many lowest states to obtain.
        :return:
            w:
                Array of k eigenvalues.

            v:
                An array of k eigenvectors. ``v[:, i]`` is the eigenvector corresponding to
                the eigenvalue ``w[i]``.
        """
        
        quspin_op = self.get_quspin_op(nsites, Nparticle, iscomplex=iscomplex)
        # if iscomplex:
        #     dtype = np.complex128
        # else:
        #     dtype = np.float64

        # num = []
        # for updown in Nparticle:
        #     num.append(np.prod(comb([nsites, nsites], updown)))
        # num = np.sum(num)
        # print(num)
        # v0 = np.zeros(int(num), dtype=dtype)
        # v0[0] = 1
        return quspin_op.eigsh(k=k, which="SA")

    def __add__(self, other: Union[Number, Operator]) -> Operator:
        """Add two operators"""
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self

        elif isinstance(other, Operator):
            op_list = self.op_list.copy()
            opstr1 = tuple(op for op, _ in op_list)
            for opstr2, interaction in other.op_list:
                try:
                    index = opstr1.index(opstr2)
                    op_list[index][1] += interaction
                except ValueError:
                    op_list.append([opstr2, interaction])
            return Operator(op_list)

        return NotImplemented

    def __radd__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self + other
        return NotImplemented

    def __iadd__(self, other: Operator) -> Operator:
        return self + other

    def __sub__(self, other: Union[Number, Operator]) -> Operator:
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self
        if isinstance(other, Operator):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return -self
        return NotImplemented

    def __isub__(self, other: Union[Number, Operator]) -> Operator:
        return self - other

    def __mul__(self, other: Union[Number, Operator]) -> Operator:
        if isinstance(other, Number):
            op_list = self.op_list.copy()
            for opstr, interaction in op_list:
                for term in interaction:
                    term[0] *= other
            return Operator(op_list)

        elif isinstance(other, Operator):
            op_list = []
            for opstr1, interaction1 in self.op_list:
                for opstr2, interaction2 in other.op_list:
                    op = [opstr1 + opstr2, []]
                    for J1, *index1 in interaction1:
                        for J2, *index2 in interaction2:
                            op[1].append([J1 * J2, *index1, *index2])
                    op_list.append(op)
            return Operator(op_list)

        return NotImplemented

    def __rmul__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self * other
        return NotImplemented

    def __imul__(self, other: Union[Number, Operator]) -> Operator:
        if isinstance(other, Number):
            for opstr, interaction in self.op_list:
                for term in interaction:
                    term[0] *= other
            return self

    def __neg__(self) -> Operator:
        return (-1) * self

    def __truediv__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self * (1 / other)
        return NotImplemented

    def __itruediv__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self.__imul__(1 / other)
        return NotImplemented
