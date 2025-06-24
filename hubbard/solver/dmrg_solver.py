# Refer to the usage of block2: https://block2.readthedocs.io/en/latest/tutorial/hubbard.html
import numpy as np
try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except:
    print("The block2 is not installed. One should not use DMRG solver.")
from quspin.operators._make_hamiltonian import _consolidate_static
import gc
import numpy as np
from typing import Dict
from hubbard.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u, number_d, number_u, S_z, S_m, S_p
from hubbard.solver._solver import Solver

class DMRG_solver(Solver):
    def __init__(
            self, 
            n_int, 
            n_noint,
            n_elec,
            nspin,
            decouple_bath: bool=False, 
            natural_orbital: bool=False,  
            iscomplex=False, 
            scratch_dir="/tmp/dmrg_tmp", 
            n_threads: int=1,
            mpi: bool=False,
            su2: bool=False,
            kBT: float=0.025,
            mutol: float=1e-4,
            nupdate=4, 
            bond_dim=250, 
            bond_mul=2, 
            n_sweep=20, 
            iprint=0,
            reorder=False,
            eig_cutoff=1e-7
            ) -> None:
        super(DMRG_solver, self).__init__(
            n_int,
            n_noint,
            n_elec,
            nspin,
            decouple_bath, 
            natural_orbital,
            iscomplex
        )

        self.kBT = kBT
        self.mutol = mutol

        # dmrg specific
        self.nupdate = nupdate
        self.bond_dim = bond_dim
        self.bond_mul = bond_mul
        self.n_sweep = n_sweep
        self.eig_cutoff = eig_cutoff
        self.reorder = reorder
        self.iprint = iprint
        self.scratch_dir = scratch_dir
        self.n_threads = n_threads
        self.mpi = mpi
        self.su2 = su2

        if self.su2:
            assert self.nspin == 1, "SU2 symmetry is allowed only when nspin = 1!"


        self.initialize_driver()

        self._t = 0.
        self._intparam = {}

    def initialize_driver(self):
        if self.nspin == 1:
            if self.su2:
                stype = SymmetryTypes.SU2
            else:
                stype = SymmetryTypes.SZ
            if self.iscomplex:
                self.driver = DMRGDriver(
                    scratch=self.scratch_dir, 
                    symm_type=SymmetryTypes.CPX | stype, 
                    n_threads=self.n_threads, 
                    mpi=self.mpi,
                    stack_mem=5368709120
                    )
            else:
                self.driver = DMRGDriver(
                    scratch=self.scratch_dir, 
                    symm_type=stype, 
                    n_threads=self.n_threads, 
                    mpi=self.mpi,
                    stack_mem=5368709120
                    )
                
            self.driver.initialize_system(n_sites=self.n_int+self.n_noint, n_elec=self.n_elec, spin=0)
        else:
            if self.iscomplex:
                self.driver = DMRGDriver(
                    scratch=self.scratch_dir, 
                    symm_type=SymmetryTypes.SGFCPX, 
                    n_threads=self.n_threads, 
                    mpi=self.mpi,
                    stack_mem=5368709120
                    )
            else:
                self.driver = DMRGDriver(
                    scratch=self.scratch_dir, 
                    symm_type=SymmetryTypes.SGF, 
                    n_threads=self.n_threads, 
                    mpi=self.mpi,
                    stack_mem=5368709120
                    )
            self.driver.initialize_system(n_sites=(self.n_noint+self.n_int)*2, n_elec=self.n_elec)

        print("driver threads: ", self.driver.bw.b.Global.threading)
    
    def format_hg(self, h1e, g2e):
        nsites = self.n_int + self.n_noint
        if self.nspin == 1:
            if self.su2:
                h1e = h1e[:nsites, :nsites]
                g2e = g2e[:nsites, :nsites, nsites:, nsites:]
            else:
                h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
                g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

        return h1e, g2e

    def _construct_Hop(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian
        op = super()._construct_Hop(T, intparam)
        nsites = self.n_int + self.n_noint
        
        h1e, g2e = op.get_Hqc(nsites=nsites, symm=True)
        assert np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).max() < 1e-7, \
            "The h1e error, {:.7f}".format(np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).max())

        h1e, g2e = self.format_hg(h1e, g2e)

        if self.reorder:
            reorder_idx = self.driver.orbital_reordering(h1e=h1e, g2e=g2e, method="gaopt")
        else:
            reorder_idx = None

        Hop = self.driver.get_qc_mpo(
            h1e=h1e, 
            g2e=g2e, 
            ecore=0., 
            iprint=0,
            reorder=reorder_idx
            )

        return Hop
        
    def solve(self, T, intparam):
        Hop = self.get_Hop(T=T, intparam=intparam)
        # Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        """
        TODO: Optimize the sweeping algorithm
        """
        if not hasattr(self, "ket") or self.reorder:
            self.ket = self.driver.get_random_mps(tag="KET", bond_dim=self.bond_dim, nroots=1)
        else:
            self.ket = self.driver.compress_mps(self.ket, max_bond_dim=self.bond_dim)

        bond_dims = [self.bond_dim] * self.nupdate + [self.bond_mul*self.bond_dim] * self.nupdate
        noises = [1e-4] * self.nupdate + [1e-5] * self.nupdate + [0]
        thrds = [1e-5] * self.nupdate + [1e-7] * self.nupdate

        self.driver.dmrg(
            Hop, self.ket, n_sweeps=self.n_sweep, bond_dims=bond_dims, 
            noises=noises, thrds=thrds, cutoff=self.eig_cutoff, iprint=self.iprint,
            cached_contraction=True, tol=1e-6
            )
        
        del Hop
        gc.collect()

        return True

    
    def cal_E(self, vec, intonly=False):
        opE = self.get_Eop(intonly=intonly)
        nsites = self.n_int + self.n_noint
        h1e, g2e = opE.get_Hqc(nsites=nsites, symm=True)
        h1e, g2e = self.format_hg(h1e, g2e)

        Hemb = self.driver.get_qc_mpo(
            h1e=h1e, 
            g2e=g2e, 
            ecore=0., 
            iprint=0,
            reorder=self.driver.reorder_idx
            )

        E = self.driver.expectation(vec, Hemb, vec)

        return E
    
    def cal_S2(self, vec, intonly=False):
        nsites = self.n_int + self.n_noint

        S2 = self.get_S2op(intonly=intonly)
        h1e, g2e = S2.get_Hqc(nsites=nsites, symm=True)
        h1e, g2e = self.format_hg(h1e, g2e)

        S2op = self.driver.get_qc_mpo(
            h1e=h1e, 
            g2e=g2e, 
            ecore=0., 
            iprint=0,
            reorder=self.driver.reorder_idx
            )

        return np.array(self.driver.expectation(vec, S2op, vec).real).reshape(1,)
    
    
    def cal_docc(self, vec, intonly=False):
        nsites = self.n_int + self.n_noint

        if intonly:
            DOCC = np.zeros(self.n_int)
            dorb = self.n_int
        else:
            DOCC = np.zeros(nsites)
            dorb = nsites

        for i in range(dorb):
            op = number_u(nsites, i) * number_d(nsites, i)

            h1e, g2e = op.get_Hqc(nsites=nsites, symm=True)
            h1e, g2e = self.format_hg(h1e, g2e)
            
            DCop = self.driver.get_qc_mpo(
                h1e=h1e, 
                g2e=g2e, 
                ecore=0., 
                iprint=0,
                reorder=self.driver.reorder_idx
                )

            DOCC[i] = self.driver.expectation(vec, DCop, vec).real

        return DOCC
    
    def E(self, intonly=False):
        return self.cal_E(self.ket, intonly=intonly)
    
    def S2(self, intonly=False):
        return self.cal_S2(self.ket, intonly=intonly)
    
    def RDM(self):
        nsites = self.n_int + self.n_noint
        RDM = self.driver.get_1pdm(self.ket)
        if self.nspin == 1:
            if self.su2:
                RDM = np.kron(0.5 * RDM, np.eye(2))
            else:
                RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
            # 
        else:
            # SGF mode, the first
            RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)

        if self.decouple_bath:
            RDM = self.recover_decoupled_bath(RDM)

        if self.natural_orbital:
            RDM = self.transmat.conj().T @ RDM @ self.transmat

        return RDM

    def docc(self, intonly=False):
        return self.cal_docc(self.ket, intonly=intonly)
    
    def quspin2block2(self, op_list):

        op_map = {
            "+-": np.array([
                ["CD", "Cd"],
                ["cD", "cd"],
            ]),
            "-+": np.array([
                ["DC", "Dc"],
                ["dC", "dc"],
            ]),
            "nn": np.array(["CDCD","err", "err", "CDcd", 
                              "err", "err", "err", "err", 
                              "err", "err", "err", "err", 
                              "cdCD", "err", "err", "cdcd"]).reshape(2,2,2,2),

            "+-+-": np.array(["CDCD","CDCd", "CDcD", "CDcd", 
                              "CdCD", "CdCd", "CdcD", "Cdcd", 
                              "cDCD", "cDCd", "cDcD", "cDcd", 
                              "cdCD", "cdCd", "cdcD", "cdcd"]).reshape(2,2,2,2),

            "++--": np.array(["CCDD", "CCDd", "CCdD", "CCdd", 
                              "CcDD", "CcDd", "CcdD", "Ccdd", 
                              "cCDD", "cCDd", "cCdD", "cCdd", 
                              "ccDD", "ccDd", "ccdD", "ccdd"]).reshape(2,2,2,2),
            "n": np.array([
                ["CD", "exx"],
                ["exx", "cd"]
                
            ])
        }

        b2oplist = []
        for op in op_list:
            str, idx, t = op # str could be "+-", "-+", "nn", "+-+-", "++--", "n"
            if str == "nn":
                idx = [idx[0],idx[0],idx[1],idx[1]]
            elif str == "n":
                idx = [idx[0], idx[0]]

            if self.nspin == 1:
                spin, idx = list(map(lambda x: x // ((self.naux+1)*self.norb), idx)), list(map(lambda x: x % ((self.naux+1)*self.norb), idx))
            else:
                spin = [0] * len(idx) # spin: 0 for up (C,D), 1 for down [c,d]
            
            spin = tuple(spin)
            b2str = op_map[str][spin]
            b2oplist.append((b2str, idx, t))
        
        return b2oplist
    

# class DMRG_solver_re(object):
#     def __init__(
#             self, 
#             norb, 
#             naux, 
#             nspin,
#             decouple_bath: bool=False, 
#             natural_orbital: bool=False,  
#             iscomplex=False, 
#             scratch_dir="/tmp/dmrg_tmp", 
#             n_threads: int=1,
#             mpi: bool=False,
#             su2: bool=False,
#             kBT: float=0.025,
#             mutol: float=1e-4,
#             nupdate=4, 
#             bond_dim=250, 
#             bond_mul=2, 
#             n_sweep=20, 
#             iprint=0,
#             reorder=False,
#             eig_cutoff=1e-7
#             ) -> None:
        
#         self.norb = norb
#         self.naux = naux
#         self.nspin = nspin
#         self.iscomplex = iscomplex
#         self.kBT = kBT
#         self.mutol = mutol

#         self.nupdate = nupdate
#         self.bond_dim = bond_dim
#         self.bond_mul = bond_mul
#         self.n_sweep = n_sweep
#         self.eig_cutoff = eig_cutoff
#         self.reorder = reorder
#         self.iprint = iprint
#         self.decouple_bath = decouple_bath
#         self.natural_orbital = natural_orbital
#         self.scratch_dir = scratch_dir
#         self.n_threads = n_threads
#         self.mpi = mpi
#         self.su2 = su2

#         if self.su2:
#             assert self.nspin == 1, "SU2 symmetry is allowed only when nspin = 1!"


#         self.initialize_driver()

#         self._t = 0.
#         self._intparam = {}

#     def initialize_driver(self):
#         if self.nspin == 1:
#             if self.su2:
#                 stype = SymmetryTypes.SU2
#             else:
#                 stype = SymmetryTypes.SZ
#             if self.iscomplex:
#                 self.driver = DMRGDriver(
#                     scratch=self.scratch_dir, 
#                     symm_type=SymmetryTypes.CPX | stype, 
#                     n_threads=self.n_threads, 
#                     mpi=self.mpi,
#                     stack_mem=5368709120
#                     )
#             else:
#                 self.driver = DMRGDriver(
#                     scratch=self.scratch_dir, 
#                     symm_type=stype, 
#                     n_threads=self.n_threads, 
#                     mpi=self.mpi,
#                     stack_mem=5368709120
#                     )
#             self.driver.initialize_system(n_sites=self.norb*(self.naux+1), n_elec=self.norb*(self.naux+1), spin=0)
#             # Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]
#             self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
#         else:
#             if self.iscomplex:
#                 self.driver = DMRGDriver(
#                     scratch=self.scratch_dir, 
#                     symm_type=SymmetryTypes.SGFCPX, 
#                     n_threads=self.n_threads, 
#                     mpi=self.mpi,
#                     stack_mem=5368709120
#                     )
#             else:
#                 self.driver = DMRGDriver(
#                     scratch=self.scratch_dir, 
#                     symm_type=SymmetryTypes.SGF, 
#                     n_threads=self.n_threads, 
#                     mpi=self.mpi,
#                     stack_mem=5368709120
#                     )
#             self.driver.initialize_system(n_sites=self.norb*(self.naux+1)*2, n_elec=self.norb*(self.naux+1))
#             # self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
#             self.Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]

#         print("driver threads: ", self.driver.bw.b.Global.threading)

#     def cal_decoupled_bath(self, T: np.array):
#         nsite = self.norb * (1+self.naux)
#         dc_T = T.reshape(nsite, 2, nsite, 2).copy()
#         if self.nspin == 1:
#             assert np.abs(dc_T[:,0,:,0] - dc_T[:,1,:,1]).max() < 1e-7
#             _, eigvec = np.linalg.eigh(dc_T[:,0,:,0][nsite:,nsite:])
#             temp_T = dc_T[:,0,:,0]
#             temp_T[nsite:] = eigvec.conj().T @ temp_T[nsite:]
#             temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvec
#             dc_T[:,0,:,0] = dc_T[:,1,:,1] = temp_T

#             self.transmat = eigvec
        
#         elif self.nspin == 2:
#             _, eigvecup = np.linalg.eigh(dc_T[:,0,:,0][nsite:,nsite:])
#             _, eigvecdown = np.linalg.eigh(dc_T[:,1,:,1][nsite:,nsite:])
#             temp_T = dc_T[:,0,:,0]
#             temp_T[nsite:] = eigvecup.conj().T @ temp_T[nsite:]
#             temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvecup
#             dc_T[:,0,:,0] = temp_T
#             temp_T = dc_T[:,1,:,1].copy()
#             temp_T[nsite:] = eigvecdown.conj().T @ temp_T[nsite:]
#             temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvecdown
#             dc_T[:,1,:,1] = temp_T

#             self.transmat = [eigvecup, eigvecdown]
        
#         else:
#             dc_T = dc_T.reshape(nsite*2, nsite*2)
#             _, eigvec = np.linalg.eigh(dc_T[nsite*2:,nsite*2:])
#             dc_T[nsite*2:] = eigvec.conj().T @ dc_T[nsite*2:]
#             dc_T[:,nsite*2:] = dc_T[:,nsite*2:] @ eigvec

#             self.transmat = eigvec
        
#         return dc_T.reshape(nsite*2, nsite*2)
    
#     def to_natrual_orbital(self, T: np.array, intparams: Dict[str, float]):
#         F,  D, _ = hartree_fock(
#             h_mat=T,
#             n_imp=self.norb,
#             n_bath=self.naux*self.norb,
#             nocc=(self.naux+1)*self.norb, # always half filled
#             max_iter=500,
#             ntol=self.mutol,
#             kBT=self.kBT,
#             tol=1e-9,
#             **intparams,
#             verbose=False,
#         )

#         D = D.reshape((self.naux+1)*self.norb,2,(self.naux+1)*self.norb,2)

#         if self.nspin<4:
#             assert np.abs(D[:,0,:,1]).max() < 1e-9

#         if self.nspin == 1:
#             err = np.abs(D[:,0,:,0]-D[:,1,:,1]).max()
#             print(D[:,0,:,0]-D[:,1,:,1])
#             assert err < 1e-8, "spin symmetry breaking error {}".format(err)

        
#         D = D.reshape((self.naux+1)*self.norb*2, (self.naux+1)*self.norb*2)

#         _, D_meanfield, self.transmat = nao_two_chain(
#                                     h_mat=F,
#                                     D=D,
#                                     n_imp=self.norb,
#                                     n_bath=self.naux*self.norb,
#                                     nspin=self.nspin,
#                                 )
#         new_T = self.transmat @ T @ self.transmat.conj().T

#         return new_T

#     def recover_decoupled_bath(self, T: np.array):
#         nsite = self.norb * (1+self.naux)
#         rc_T = T.reshape(nsite,2,nsite,2).copy()
#         if self.nspin == 1:
#             assert np.abs(rc_T[:,0,:,0] - rc_T[:,1,:,1]).max() < 1e-7
#             temp_T = rc_T[:,0,:,0]
#             temp_T[nsite:] = self.transmat @ temp_T[nsite:]
#             temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat.conj().T
#             rc_T[:,0,:,0] = rc_T[:,1,:,1] = temp_T

#         if self.nspin == 2:
#             temp_T = rc_T[:,0,:,0]
#             temp_T[nsite:] = self.transmat[0] @ temp_T[nsite:]
#             temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat[0].conj().T
#             rc_T[:,0,:,0] = temp_T
#             temp_T = rc_T[:,1,:,1].copy()
#             temp_T[nsite:] = self.transmat[1] @ temp_T[nsite:]
#             temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat[1].conj().T
#             rc_T[:,1,:,1] = temp_T

#         if self.nspin == 4:
#             rc_T = rc_T.reshape(nsite*2, nsite*2)
#             rc_T[nsite*2:] = self.transmat @ rc_T[nsite*2:]
#             rc_T[:,nsite*2:] = rc_T[:,nsite*2:] @ self.transmat.conj().T
        
#         return rc_T.reshape(nsite*2, nsite*2)

#     def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
#         # construct the embedding Hamiltonian
#         if self.decouple_bath:
#             assert T.shape[0] == (self.naux+1) * self.norb * 2
#             # The first (self.norb, self.norb)*2 is the impurity, which is keeped unchange.
#             # We do diagonalization of the other block
#             T = self.cal_decoupled_bath(T=T)
#         elif self.natural_orbital:
#             assert T.shape[0] == (self.naux+1) * self.norb * 2
#             T = self.to_natrual_orbital(T=T, intparams=intparam)

#         intparam = copy.deepcopy(intparam)
#         intparam["t"] = T.copy()
#         self._t = T
#         self._intparam = intparam
#         nsites = self.norb*(self.naux+1)

#         op = Slater_Kanamori(
#                 nsites=nsites,
#                 n_noninteracting=self.norb*self.naux,
#                 **intparam
#             )
        
#         h1e, g2e = op.get_Hqc(nsites=nsites)
#         assert np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).max() < 1e-7, "The h1e error, {:.7f}".format(np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).max())
#         ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#         g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#         g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#         if self.nspin == 1:
#             if self.su2:
#                 h1e = h1e[:nsites, :nsites]
#                 g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#             else:
#                 h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#                 g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

#         if self.reorder:
#             reorder_idx = self.driver.orbital_reordering(h1e=h1e, g2e=g2e, method="gaopt")
#         else:
#             reorder_idx = None

#         Hemb = self.driver.get_qc_mpo(
#             h1e=h1e, 
#             g2e=g2e, 
#             ecore=0., 
#             iprint=0,
#             reorder=reorder_idx
#             )

#         # if self.nspin == 1:
#         #     h1e, g2e = op.get_Hqc(nsites=nsites)
#         #     # check for spin symmetry
#         #     # assert np.abs(h1e[:nsites, :nsites] - h1e[nsites:, nsites:]).mean() < 1e-8
#         #     # assert (np.abs(h1e[:nsites, nsites:]) + np.abs(h1e[nsites:, :nsites])).mean() < 1e-8
#         #     # h1e = h1e[:nsites, :nsites]
#         #     # assert np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).mean() < 1e-8
#         #     # g2e = g2e[:nsites, :nsites, :nsites, :nsites]
#         #     # Hemb = self.driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=0., iprint=0)
#         #     # h1e = h1e + h1e.conj().T
#         #     # h1e *= 0.5
#         #     # assert np.abs(T.reshape(nsites,2,nsites,2).transpose(1,0,3,2).reshape(2*nsites, 2*nsites)-h1e).max() < 1e-7, "The h1e error, {:.7f}".format(np.abs(T.reshape(nsites,2,nsites,2).transpose(1,0,3,2).reshape(2*nsites, 2*nsites)-h1e).max())
#         #     assert np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).max() < 1e-7, "The h1e error, {:.7f}".format(np.abs(g2e[:nsites, :nsites, :nsites, :nsites] - g2e[nsites:, nsites:, nsites:, nsites:]).max())
#         #     ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#         #     g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#         #     g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#         #     if self.su2:
#         #         h1e = h1e[:nsites, :nsites]
#         #         g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#         #     else:
#         #         h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#         #         g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

#         #     Hemb = self.driver.get_qc_mpo(
#         #         h1e=h1e, 
#         #         g2e=g2e, 
#         #         ecore=0., 
#         #         iprint=0
#         #         )
#         # else:
#         #     if not self.reorder:
#         #         oplist = _consolidate_static( # to combine reducible op
#         #                 Slater_Kanamori(
#         #                 nsites=self.norb*(self.naux+1),
#         #                 n_noninteracting=self.norb*self.naux,
#         #                 **intparam
#         #             ).op_list
#         #         )
#         #         # we should notice that the spinorbital are not adjacent in the quspin hamiltonian, 
#         #         # so properties computed from this need to be transformed.
#         #         b = self.driver.expr_builder()
#         #         b2oplist = self.quspin2block2(op_list=oplist)
#         #         for op in b2oplist:
#         #             b.add_term(*op)
#         #         # SZ and SGF (w.r.t. nspin=1,2/4) use different oplist definition, try to differentiate these two
#         #             # C,D,c,d for up_create, up_anni, down_create, down_anni, SGF mode only have CD
#         #         Hemb = self.driver.get_mpo(b.finalize(), iprint=0)

#         #         del b
#         #         # del fcidump
#         #         gc.collect()
#         #     else:
#         #         h1e, g2e = op.get_Hqc(nsites=nsites)

#         #         idx = self.driver.orbital_reordering(h1e=h1e, g2e=g2e)

#         #         Hemb = self.driver.get_qc_mpo(
#         #             h1e=h1e[],
#         #             g2e=g2e, 
#         #             ecore=0., 
#         #             iprint=0
#         #         )


#         return Hemb
    
#     def get_Hemb(self, T, intparam):
#         return self._construct_Hemb(T, intparam)
        
#     def solve(self, T, intparam, return_RDM: bool=False):
#         RDM = None
#         Hemb = self.get_Hemb(T=T, intparam=intparam)
#         nsites = self.norb*(self.naux+1)
#         # Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
#         """
#         TODO: Optimize the sweeping algorithm
#         """
#         if not hasattr(self, "ket") or self.reorder:
#             self.ket = self.driver.get_random_mps(tag="KET", bond_dim=self.bond_dim, nroots=1)
#         else:
#             self.ket = self.driver.compress_mps(self.ket, max_bond_dim=self.bond_dim)

#         bond_dims = [self.bond_dim] * self.nupdate + [self.bond_mul*self.bond_dim] * self.nupdate
#         noises = [1e-4] * self.nupdate + [1e-5] * self.nupdate + [0]
#         thrds = [1e-5] * self.nupdate + [1e-7] * self.nupdate

#         # def f(name, iprint):
#         #     if name == "DMRG::sweep::iter.start":
#         #         import resource, psutil, os
#         #         curr = psutil.Process(os.getpid()).memory_info().rss
#         #         peak = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#         #         print("DMRG::sweep::iter.start: %10s %10s \n" % (self.driver.bw.b.Parsing.to_size_string(curr), self.driver.bw.b.Parsing.to_size_string(peak)), end='', flush=True)

#         #     if name == "DMRG::sweep::iter.end":
#         #         import resource, psutil, os
#         #         curr = psutil.Process(os.getpid()).memory_info().rss
#         #         peak = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#         #         print("DMRG::sweep::iter.end: %10s %10s \n" % (self.driver.bw.b.Parsing.to_size_string(curr), self.driver.bw.b.Parsing.to_size_string(peak)), end='', flush=True)
        
#         # self.driver.set_callback(f)
#         self.driver.dmrg(
#             Hemb, self.ket, n_sweeps=self.n_sweep, bond_dims=bond_dims, 
#             noises=noises, thrds=thrds, cutoff=self.eig_cutoff, iprint=self.iprint,
#             cached_contraction=True, tol=1e-6
#             )
        
#         del Hemb
#         gc.collect()

#         if return_RDM:
#             RDM = self.RDM
#             # RDM = self.driver.get_1pdm(self.ket)
#             # if self.nspin == 1:
#             #     if self.su2:
#             #         RDM = np.kron(0.5 * RDM, np.eye(2))
#             #     else:
#             #         RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
#             # else:
#             #     # SGF mode, the first
#             #     RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)
            
#             if self.decouple_bath:
#                 RDM = self.recover_decoupled_bath(RDM)

#             if self.natural_orbital:
#                 RDM = self.transmat.conj().T @ RDM @ self.transmat
                
#         return RDM
    
#     def cal_E(self, vec):
#         intparam = copy.deepcopy(self._intparam)
#         intparam["t"][self.norb*2:] = 0.
#         intparam["t"][:, self.norb*2:] = 0.
#         opE = Slater_Kanamori(
#                 nsites=self.norb*(self.naux+1),
#                 n_noninteracting=self.norb*self.naux,
#                 **intparam
#             )
#         nsites = (self.naux + 1) * self.norb

#         h1e, g2e = opE.get_Hqc(nsites=nsites)
#         ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#         g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#         g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#         if self.nspin == 1:
#             if self.su2:
#                 h1e = h1e[:nsites, :nsites]
#                 g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#             else:
#                 h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#                 g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

#         Hemb = self.driver.get_qc_mpo(
#             h1e=h1e, 
#             g2e=g2e, 
#             ecore=0., 
#             iprint=0,
#             reorder=self.driver.reorder_idx
#             )
        
#         # if self.nspin == 1:
#         #     h1e, g2e = opE.get_Hqc(nsites=nsites)
#         #     ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#         #     g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#         #     g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#         #     if self.su2:
#         #         h1e = h1e[:nsites, :nsites]
#         #         g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#         #     else:
#         #         h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#         #         g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

#         #     Hemb = self.driver.get_qc_mpo(
#         #         h1e=h1e, 
#         #         g2e=g2e, 
#         #         ecore=0., 
#         #         iprint=0
#         #         )
#         # else:
#         #     oplist = _consolidate_static( # to combine reducible op
#         #         opE.op_list
#         #     )

#         #     b = self.driver.expr_builder()
#         #     b2oplist = self.quspin2block2(op_list=oplist)
#         #     for op in b2oplist:
#         #         b.add_term(*op)
            
#         #     Hemb = self.driver.get_mpo(b.finalize(), iprint=0)

#         E = self.driver.expectation(vec, Hemb, vec)

#         return E
    
#     def cal_S2(self, vec):
#         nsites = (self.naux + 1) * self.norb

#         nsites = self.norb*(self.naux+1)

#         S_m_ = sum(S_m(nsites, i) for i in range(self.norb))
#         S_p_ = sum(S_p(nsites, i) for i in range(self.norb))
#         S_z_ = sum(S_z(nsites, i) for i in range(self.norb))

#         S2 = S_m_ * S_p_ + S_z_ * S_z_ + S_z_
#         # print([i[0] for i in S2.op_list])

#         h1e, g2e = S2.get_Hqc(nsites=nsites)
#         ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#         g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#         g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#         if self.nspin == 1:
#             if self.su2:
#                 h1e = h1e[:nsites, :nsites]
#                 g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#             else:
#                 h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#                 g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

#         S2op = self.driver.get_qc_mpo(
#             h1e=h1e, 
#             g2e=g2e, 
#             ecore=0., 
#             iprint=0,
#             reorder=self.driver.reorder_idx
#             )

#         # if self.nspin == 1:
#         #     h1e, g2e = S2.get_Hqc(nsites=nsites)
#         #     ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#         #     g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#         #     g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#         #     if self.su2:
#         #         h1e = h1e[:nsites, :nsites]
#         #         g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#         #     else:
#         #         h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#         #         g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]

#         #     S2op = self.driver.get_qc_mpo(
#         #         h1e=h1e, 
#         #         g2e=g2e, 
#         #         ecore=0., 
#         #         iprint=0,
#         #         reorder=self.driver.reorder_idx
#         #         )
#         # else:
#         #     oplist = _consolidate_static(S2.op_list)
#         #     b = self.driver.expr_builder()
#         #     b2oplist = self.quspin2block2(op_list=oplist)
#         #     for op in b2oplist:
#         #         b.add_term(*op)
            
#         #     S2op = self.driver.get_mpo(b.finalize(), iprint=0)

#         return np.array(self.driver.expectation(vec, S2op, vec).real).reshape(1,)
    
    
#     def cal_docc(self, vec):
#         nsites = (self.naux + 1) * self.norb

#         DOCC = np.zeros(self.norb)
#         for i in range(self.norb):
#             op = number_u(nsites, i) * number_d(nsites, i)

#             h1e, g2e = op.get_Hqc(nsites=nsites)
#             ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#             g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#             g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()

#             if self.nspin == 1:
#                 if self.su2:
#                     h1e = h1e[:nsites, :nsites]
#                     g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#                 else:
#                     h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#                     g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]
            
#             DCop = self.driver.get_qc_mpo(
#                 h1e=h1e, 
#                 g2e=g2e, 
#                 ecore=0., 
#                 iprint=0,
#                 reorder=self.driver.reorder_idx
#                 )
#             # if self.nspin == 1:
#             #     h1e, g2e = op.get_Hqc(nsites=nsites)
#             #     ofd = 0.5 * (g2e[:nsites,:nsites,nsites:,nsites:] + g2e[nsites:,nsites:,:nsites,:nsites])
#             #     g2e[:nsites,:nsites,nsites:,nsites:] = ofd.copy()
#             #     g2e[nsites:,nsites:,:nsites,:nsites] = ofd.copy()
#             #     if self.su2:
#             #         h1e = h1e[:nsites, :nsites]
#             #         g2e = g2e[:nsites, :nsites, nsites:, nsites:]
#             #     else:
#             #         h1e = [h1e[:nsites, :nsites], h1e[nsites:,nsites:]]
#             #         g2e = [g2e[:nsites, :nsites, :nsites, :nsites], g2e[:nsites, :nsites, nsites:, nsites:], g2e[nsites:, nsites:, nsites:, nsites:]]
#             #     DCop = self.driver.get_qc_mpo(
#             #         h1e=h1e, 
#             #         g2e=g2e, 
#             #         ecore=0., 
#             #         iprint=0
#             #         )
#             # else:
#             #     oplist = _consolidate_static( # to combine reducible op
#             #         op.op_list
#             #     )
#             #     b = self.driver.expr_builder()
#             #     b2oplist = self.quspin2block2(op_list=oplist)
#             #     for op in b2oplist:
#             #         b.add_term(*op)
                
#             #     DCop = self.driver.get_mpo(b.finalize(), iprint=0)

#             DOCC[i] = self.driver.expectation(vec, DCop, vec).real

#         return DOCC
    
#     @property
#     def E(self):
#         return self.cal_E(self.ket)
    
#     @property
#     def S2(self):
#         return self.cal_S2(self.ket)
    
#     @property
#     def RDM(self):
#         nsites = (self.naux + 1) * self.norb
#         RDM = self.driver.get_1pdm(self.ket)
#         if self.nspin == 1:
#             if self.su2:
#                 RDM = np.kron(0.5 * RDM, np.eye(2))
#             else:
#                 RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
#             # 
#         else:
#             # SGF mode, the first
#             RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)

#         if self.decouple_bath:
#             RDM = self.recover_decoupled_bath(RDM)

#         return RDM
    
#     @property
#     def docc(self):
#         return self.cal_docc(self.ket)