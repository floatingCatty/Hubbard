try:
    from pyscf import fci, gto, scf, ao2mo, ci, cc, lib, dmrgscf
    from pyscf.scf import diis
except:
    print("The pyscf & dmrgscf is not installed. One should not use pyscf solver.")
import numpy
import numpy as np
from quspin.operators._make_hamiltonian import _consolidate_static
import copy
from typing import Dict
from hubbard.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u, number_d, number_u, S_z, S_m, S_p
import os

class Pyscf_ccsd(object):
    """ Wrapper for pyscf ccsd solvers
    """
    def __init__(self, ntot, nimp, nbath):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.hsize = 2**ntot
        self.type= 'PySCFCCSD'
        # initialize pyscf solvers

    def build_Hemb(self, D, H1E, LAMBDA, V2E, spin_pen=0.0):
        tmat = numpy.zeros((self.ntot,self.ntot))
        tmat[:self.nimp,:self.nimp] = H1E
        tmat[:self.nimp,self.nimp:] = D.T
        tmat[self.nimp:,self.nimp:] = -LAMBDA
        tmat[self.nimp:,:self.nimp] = D.conj()
        self.h1 = tmat[::2,::2]
        self.h2 = numpy.zeros((self.ntot//2,self.ntot//2,self.ntot//2,self.ntot//2))
        self.h2[:self.nimp//2,:self.nimp//2,:self.nimp//2,:self.nimp//2] = V2E[::2,::2,1::2,1::2] # spin symmetric

    def solve_Hemb(self, num_eig=10, verbose=0, restrict=True):
        self.restrict = restrict # restrict CCSD? True: for spin symmetry
        if restrict:
            mol = gto.M()
            mol.nelectron = self.ntot//2
            mol.incore_anyway = True
            mf = scf.RHF(mol)
            mf.max_cycle = 1000
            mf.conv_tol = 1e-8#1e-12
            mf.diis_space = 20
            mf.diis_start_cycle = 5
            mf.chkfile = 'hf.chk'
            if os.path.isfile('hf.chk'):
                mol = lib.chkfile.load_mol('hf.chk')
                #mf = scf.HF(mol)
                mf.__dict__.update(lib.chkfile.load('hf.chk', 'scf'))
            mf.get_hcore = lambda *args: self.h1
            mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
            mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
            mf.init_guess = '1e'
            mf.diis = diis.EDIIS
            mf = mf.run()
            self.C = mf.mo_coeff
            self.mycc = cc.RCCSD(mf)
            self.mycc.conv_tol = 1e-8
            self.mycc.conv_tol_normt = 1e-5
            self.mycc.max_cycle = 10000
            self.mycc.diis_space = 20#15
            self.mycc.diis_start_cycle = 5
            self.mycc.diis = diis.EDIIS
            self.mycc.iterative_damping = 0.05
            #self.mycc.diis = False
            self.mycc.diis_file = 'ccdiis.h5'
            if os.path.isfile('ccdiis.h5'):
                self.mycc.restore_from_diis_('ccdiis.h5')
            #self.eccsd, t1, t2 = self.mycc.kernel()
            self.eccsd, t1, t2 = self.mycc.ccsd()
            #self.mycc.nroots = 3
            #e, v = self.mycc.ipccsd()
            #self.mycc.solve_lambda()
            self.e0 = self.eccsd + mf.energy_tot()
        else:
            mol = gto.M()
            mol.nelectron = self.ntot//2
            mol.incore_anyway = True
            mf = scf.UHF(mol)
            mf.max_cycle = 1000
            mf.conv_tol = 1e-8
            mf.diis_space = 20
            mf.init_guess_breaksym = True
            mf.get_hcore = lambda *args: self.h1
            mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
            mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
            mf.init_guess = 'minao'
            mf = mf.run()
            self.Cup, self.Cdn = mf.mo_coeff
            self.mycc = cc.UCCSD(mf)
            self.mycc.conv_tol = 1e-8
            self.mycc.conv_tol_normt = 1e-5
            self.mycc.max_cycle = 500
            self.mycc.diis_space = 20
            self.mycc.diis_start_cycle = 4
            #self.mycc.diis = False
            self.mycc.iterative_damping = 0.001
            self.mycc.diis = diis.ADIIS
            #self.eccsd, t1, t2 = self.mycc.kernel()
            self.eccsd, t1, t2 = self.mycc.ccsd()
            #self.mycc.nroots = 3
            #e, v = self.mycc.ipccsd()
            #self.mycc.solve_lambda()
            self.e0 = self.eccsd + mf.energy_tot()

    def calc_density_matrix(self):
        if self.restrict:
            dmup = self.mycc.make_rdm1()/2.
            dmup = numpy.einsum('ai,bj,ij->ab',self.C, self.C, dmup)
            #print('dmup=')
            #print(dmup)
            #assert(numpy.allclose(dmup,dmdn))
            dm = numpy.kron(dmup,numpy.eye(2))
            self.dm = dm
            return dm
        else:
            dmup, dmdn = self.mycc.make_rdm1()
            dmup = numpy.einsum('ai,bj,ij->ab',self.Cup, self.Cup, dmup)
            dmdn = numpy.einsum('ai,bj,ij->ab',self.Cdn, self.Cdn, dmdn)
            #print('dmup=')
            #print(dmup)
            #print('dmdn=')
            #print(dmdn)
            dm = (dmup + dmdn)/2. # average over spin to recover spin symmetry. Don't use it with magnetism
            dm = numpy.kron(dm,numpy.eye(2))
            self.dm = dm
            return dm

    def compute_E1loc(self, nimp):
        '''
        Compute local energy including local one and two-body term from a given wavefunction.
        Input:
        Return:
          Eloc: float. Total local energy.
        '''
        #return self.gs_wf.conj().T.dot((self.Htwo).dot(self.gs_wf))
        return numpy.trace(self.h1[:nimp,:nimp].dot(self.dm[:nimp,:nimp].T))

    def compute_E2loc(self):
        eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        etwo = self.e0 - eone
        return etwo

    def calc_double_occ(self,idx):
        print('double occupancy not implemented')
        return 0.25

class Pyscf_dmrg(Pyscf_ccsd):
    """ Wrapper to pyscf dmrg
    """
    def __init__(self, ntot, nimp, nbath, maxM):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.hsize = 2**ntot
        self.maxM = maxM
        self.type= 'PySCFDMRG'
        print('maxM=',maxM)
        # initialize pyscf solvers

    @staticmethod
    def gauge_transform(h1e, ntot, nimp):
        from scipy.linalg import eig, eigh
        h1e_trans = numpy.zeros((ntot,ntot), dtype=numpy.float64)
        u_trans = numpy.eye(ntot, dtype=numpy.float64)
        h1e_trans[:nimp,:nimp] = h1e[:nimp,:nimp]
        # enforce spin symmtery
        evals, tmp = eigh(h1e[nimp:,nimp:])
        u_trans[nimp:,nimp:] = tmp
        h1e_trans = u_trans.conj().T.dot(h1e).dot(u_trans)
        print('h1e_trans=')
        print(h1e_trans)
        return h1e_trans, u_trans

    def solve_Hemb(self, num_eig=10, verbose=0, sweep_iter = [0,10,20], sweep_epsilon = [5e-3,1e-3,5e-4], maxM=500):
        mol = gto.M()
        mol.nelectron = self.ntot//2
        mol.incore_anyway = True
        mf = scf.RHF(mol)
        mf.max_cycle = 1000
        mf.conv_tol = 1e-8#1e-12
        #mf.diis_space = 5#10
        #mf.diis_start_cycle = 5
        mf.diis = False
        mf.DIIS = diis.ADIIS
        #self.h1, self.u_trans = self.gauge_transform(self.h1, self.ntot//2, self.nimp//2)
        mf.get_hcore = lambda *args: self.h1
        mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
        mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
        #mf._eri = self.h2
        #mf.init_guess = '1e'
        mf = mf.run()
        self.C = mf.mo_coeff

        #dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
        #dmrgscf.settings.MPIPREFIX = ''
        #dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
        #dmrgscf.settings.MPIPREFIX = 'mpirun -n 4 --bind-to none'

        self.nmo = mf.mo_coeff.shape[1]
        self.nelec = mol.nelec # this part report error when nelectron is odd
        self.cisolver = dmrgscf.DMRGSCF(mf, self.nmo, self.nelec, maxM=self.maxM, tol=1e-8)#shci.SHCISCF(mf, ntot//2, ntot//2)
        #self.cisolver.fcisolver.runtimeDir = lib.param.TMPDIR
        #self.cisolver.fcisolver.scratchDirectory = lib.param.TMPDIR
        self.cisolver.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4))
        print('using threads:',self.cisolver.fcisolver.threads)
        self.cisolver.fcisolver.memory = 50 #int(mol.max_memory / 1000) # mem in GB
        #self.cisolver.canonicalization = True
        #self.cisolver.natorb = True
        self.e0 = self.cisolver.kernel()[0]

    def calc_density_matrix(self):
        dm1 = self.cisolver.fcisolver.make_rdm1(0, self.nmo, self.nelec)/2.
        dmup = numpy.einsum('ai,bj,ij->ab', self.C, self.C, dm1)
        #print('dmup=')
        #print(dmup)
        #print('dmdn=')
        #print(dmdn)
        #assert(numpy.allclose(dmup,dmdn))
        #dmup = self.u_trans.conj().dot( dmup ).dot(self.u_trans.T)
        dm = numpy.kron(dmup,numpy.eye(2))
        self.dm = dm
        return dm

    def calc_double_occ(self,idx):
        #tmp = fci.addons.des_a(self.fcivec, self.ntot//2, (self.ntot//4  ,self.ntot//4  ), idx//2)
        #tmp = fci.addons.cre_a(tmp        , self.ntot//2, (self.ntot//4-1,self.ntot//4  ), idx//2)
        #tmp = fci.addons.des_b(tmp        , self.ntot//2, (self.ntot//4  ,self.ntot//4  ), idx//2)
        #tmp = fci.addons.cre_b(tmp        , self.ntot//2, (self.ntot//4  ,self.ntot//4-1), idx//2)
        #return numpy.dot(tmp.flatten(), self.fcivec.flatten())
        return 0.25

    def compute_E2loc(self):
        eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        etwo = self.e0 - eone
        return etwo


class PYSCF_solver(object):
    def __init__(
            self, 
            norb, 
            naux, 
            nspin,
            decouple_bath: bool=False, 
            natural_orbital: bool=False,  
            iscomplex=False, 
            scratch_dir="/tmp/dmrg_tmp", 
            n_threads: int=1,
            mpi: bool=False,
            kBT: float=0.025,
            mutol: float=1e-4,
            nupdate=4, 
            bond_dim=250, 
            bond_mul=2, 
            n_sweep=20, 
            eig_cutoff=1e-7
            ) -> None:
        
        self.norb = norb
        self.naux = naux
        self.nspin = nspin
        self.iscomplex = iscomplex

        self.nupdate = nupdate
        self.bond_dim = bond_dim
        self.bond_mul = bond_mul
        self.n_sweep = n_sweep
        self.eig_cutoff = eig_cutoff
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.scratch_dir = scratch_dir
        self.n_threads = n_threads

        self._t = 0.
        self._intparam = {}

    def cal_decoupled_bath(self, T: np.array):
        nsite = self.norb * (1+self.naux)
        dc_T = T.reshape(nsite, 2, nsite, 2).copy()
        if self.nspin == 1:
            assert np.abs(dc_T[:,0,:,0] - dc_T[:,1,:,1]).max() < 1e-7
            _, eigvec = np.linalg.eigh(dc_T[:,0,:,0][nsite:,nsite:])
            temp_T = dc_T[:,0,:,0]
            temp_T[nsite:] = eigvec.conj().T @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvec
            dc_T[:,0,:,0] = dc_T[:,1,:,1] = temp_T

            self.transmat = eigvec
        
        elif self.nspin == 2:
            _, eigvecup = np.linalg.eigh(dc_T[:,0,:,0][nsite:,nsite:])
            _, eigvecdown = np.linalg.eigh(dc_T[:,1,:,1][nsite:,nsite:])
            temp_T = dc_T[:,0,:,0]
            temp_T[nsite:] = eigvecup.conj().T @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvecup
            dc_T[:,0,:,0] = temp_T
            temp_T = dc_T[:,1,:,1].copy()
            temp_T[nsite:] = eigvecdown.conj().T @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvecdown
            dc_T[:,1,:,1] = temp_T

            self.transmat = [eigvecup, eigvecdown]
        
        else:
            dc_T = dc_T.reshape(nsite*2, nsite*2)
            _, eigvec = np.linalg.eigh(dc_T[nsite*2:,nsite*2:])
            dc_T[nsite*2:] = eigvec.conj().T @ dc_T[nsite*2:]
            dc_T[:,nsite*2:] = dc_T[:,nsite*2:] @ eigvec

            self.transmat = eigvec
        
        return dc_T.reshape(nsite*2, nsite*2)

    def recover_decoupled_bath(self, T: np.array):
        nsite = self.norb * (1+self.naux)
        rc_T = T.reshape(nsite,2,nsite,2).copy()
        if self.nspin == 1:
            assert np.abs(rc_T[:,0,:,0] - rc_T[:,1,:,1]).max() < 1e-7
            temp_T = rc_T[:,0,:,0]
            temp_T[nsite:] = self.transmat @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat.conj().T
            rc_T[:,0,:,0] = rc_T[:,1,:,1] = temp_T

        if self.nspin == 2:
            temp_T = rc_T[:,0,:,0]
            temp_T[nsite:] = self.transmat[0] @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat[0].conj().T
            rc_T[:,0,:,0] = temp_T
            temp_T = rc_T[:,1,:,1].copy()
            temp_T[nsite:] = self.transmat[1] @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat[1].conj().T
            rc_T[:,1,:,1] = temp_T

        if self.nspin == 4:
            rc_T = rc_T.reshape(nsite*2, nsite*2)
            rc_T[nsite*2:] = self.transmat @ rc_T[nsite*2:]
            rc_T[:,nsite*2:] = rc_T[:,nsite*2:] @ self.transmat.conj().T
        
        return rc_T.reshape(nsite*2, nsite*2)

    def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian
        if self.decouple_bath:
            assert T.shape[0] == (self.naux+1) * self.norb * 2
            # The first (self.norb, self.norb)*2 is the impurity, which is keeped unchange.
            # We do diagonalization of the other block
            T = self.cal_decoupled_bath(T=T)

        intparam = copy.deepcopy(intparam)
        intparam["t"] = T.copy()
        self._t = T
        self._intparam = intparam

        oplist = _consolidate_static( # to combine reducible op
                Slater_Kanamori(
                nsites=self.norb*(self.naux+1),
                n_noninteracting=self.norb*self.naux,
                **intparam
            ).op_list
        )

        nsite = self.norb * (1+self.naux)
        h1e, h2e = self.quspin2pyscf(oplist)
        # transform T and intparams, i.e. oplist to the h1e and h2e.
        assert self.nspin == 1, "Currently, PYSCF DMRG solver is best suited for nspin=1 case."
        h1 = h1e[::2,::2]
        h2 = numpy.zeros((nsite,nsite,nsite,nsite))
        h2[:self.norb,:self.norb,:self.norb,:self.norb] = h2e[::2,::2,1::2,1::2] # spin symmetric

        # get the pyscf object
        mol = gto.M()
        mol.nelectron = nsite
        mol.incore_anyway = True
        mf = scf.RHF(mol)
        mf.max_cycle = 1000
        mf.conv_tol = 1e-8#1e-12
        #mf.diis_space = 5#10
        #mf.diis_start_cycle = 5
        mf.diis = False
        mf.DIIS = diis.ADIIS

        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(nsite)
        mf._eri = ao2mo.restore(8, h2, nsite) # 8-fold symmetry
        #mf._eri = self.h2
        #mf.init_guess = '1e'

        return mf
    
    def get_Hemb(self, T, intparam):
        return self._construct_Hemb(T, intparam)
        
    def solve(self, T, intparam, return_RDM: bool=False):
        RDM = None
        Hemb = self.get_Hemb(T=T, intparam=intparam)
        nsites = self.norb*(self.naux+1)
        # Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]

        mf = mf.run()
        self.C = mf.mo_coeff

        self.nmo = mf.mo_coeff.shape[1]
        self.nelec = nsites
        self.cisolver = dmrgscf.DMRGSCF(mf, self.nmo, self.nelec, maxM=self.bond_dim, tol=1e-8)#shci.SHCISCF(mf, ntot//2, ntot//2)
        #self.cisolver.fcisolver.runtimeDir = lib.param.TMPDIR
        #self.cisolver.fcisolver.scratchDirectory = lib.param.TMPDIR
        self.cisolver.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", self.n_threads))
        self.cisolver.fcisolver.memory = 50 #int(mol.max_memory / 1000) # mem in GB
        #self.cisolver.canonicalization = True
        #self.cisolver.natorb = True
        self.e0 = self.cisolver.kernel()[0]


        if return_RDM:
            RDM = self.driver.get_1pdm(self.ket)
            if self.nspin == 1:
                RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
            else:
                # SGF mode, the first
                RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)
            
            if self.decouple_bath:
                RDM = self.recover_decoupled_bath(RDM)
                
        return RDM
    
    def cal_E(self, vec):
        intparam = copy.deepcopy(self._intparam)
        intparam["t"][self.norb*2:] = 0.
        intparam["t"][:, self.norb*2:] = 0.

        oplist = _consolidate_static( # to combine reducible op
                Slater_Kanamori(
                nsites=self.norb*(self.naux+1),
                n_noninteracting=self.norb*self.naux,
                **intparam
            ).op_list
        )

        b = self.driver.expr_builder()
        b2oplist = self.quspin2block2(op_list=oplist)
        for op in b2oplist:
            b.add_term(*op)
        
        Hemb = self.driver.get_mpo(b.finalize(), iprint=0)

        E = self.driver.expectation(vec, Hemb, vec)

        return E
    
    def cal_S2(self, vec):
        nsites = (self.naux + 1) * self.norb

        nsites = self.norb*(self.naux+1)

        S_m_ = sum(S_m(nsites, i) for i in range(self.norb))
        S_p_ = sum(S_p(nsites, i) for i in range(self.norb))
        S_z_ = sum(S_z(nsites, i) for i in range(self.norb))

        S2 = S_m_ * S_p_ + S_z_ * S_z_ + S_z_

        oplist = _consolidate_static(S2.op_list)
        b = self.driver.expr_builder()
        b2oplist = self.quspin2block2(op_list=oplist)
        for op in b2oplist:
            b.add_term(*op)
        
        S2op = self.driver.get_mpo(b.finalize(), iprint=0)

        return np.array(self.driver.expectation(vec, S2op, vec).real).reshape(1,)
    
    
    def cal_docc(self, vec):
        nsites = (self.naux + 1) * self.norb

        DOCC = np.zeros(self.norb)
        for i in range(self.norb):
            oplist = _consolidate_static( # to combine reducible op
                (number_u(nsites, i) * number_d(nsites, i)).op_list
            )
            b = self.driver.expr_builder()
            b2oplist = self.quspin2block2(op_list=oplist)
            for op in b2oplist:
                b.add_term(*op)
            
            DCop = self.driver.get_mpo(b.finalize(), iprint=0)

            DOCC[i] = self.driver.expectation(vec, DCop, vec).real

        return DOCC
    
    @property
    def E(self):
        return self.cal_E(self.ket)
    
    @property
    def S2(self):
        return self.cal_S2(self.ket)
    
    @property
    def RDM(self):
        nsites = (self.naux + 1) * self.norb
        RDM = self.driver.get_1pdm(self.ket)
        if self.nspin == 1:
            RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
        else:
            # SGF mode, the first
            RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)

        if self.decouple_bath:
            RDM = self.recover_decoupled_bath(RDM)

        return RDM
    
    @property
    def docc(self):
        return self.cal_docc(self.ket)
    
    def quspin2pyscf(self, op_list):
        h1e = np.zeros(((self.naux + 1) * self.norb * 2, (self.naux + 1) * self.norb * 2))
        h2e = np.zeros(((self.naux + 1) * self.norb * 2, (self.naux + 1) * self.norb * 2, (self.naux + 1) * self.norb * 2, (self.naux + 1) * self.norb * 2))
        for op in op_list:
            str, idx, t = op # str could be "+-", "nn", "+-+-", "++--", "n"
            if str == "nn":
                idx = [idx[0],idx[0],idx[1],idx[1]]
            elif str == "n":
                idx = [idx[0], idx[0]]
            idx = tuple(idx)

            if len(idx) == 2:
                h1e[idx] = t
            elif len(idx) == 4:
                h2e[idx] = t
            else:
                raise ValueError("The index length is not correct.")
        
        return h1e, h2e

