import numpy as np
import copy
from typing import Dict
from gGA.operator import Slater_Kanamori
from quantax.operator import create_u, create_d, annihilate_u, annihilate_d, Operator
import jax.numpy as jnp
import quantax as qtx
from tqdm import tqdm
import equinox as eqx
from gGA.nao.ghf import generalized_hartree_fock # not useful yet
from gGA.nao.hf import hartree_fock
from gGA.nao.tonao import nao_two_chain

class NQS_solver(object):
    def __init__(
            self, 
            norb, 
            naux, 
            nspin, 
            decouple_bath: bool=False, 
            natural_orbital: bool=False, 
            iscomplex=False, 
            nblocks=4,
            channels=2,
            kernel_size=3,
            d=1,
            h=1,
            Nsamples=1000, # number of samples in training
            mftol=1e-2, # variance tol for energy in mean-field optimization
            mfepmax=5000, # max epoch for mean-field wave-function training
            mfepmin=600, # min epoch for mean-field wave-function training (but it would be break if Etol is reached)
            nnepmax=5000,
            Etol=1e-3, # variance tol for energy minimization
            Ptol=4e-5, # error tol for property evaluation
            ) -> None:

        self.norb = norb
        self.naux = naux
        self.nspin = nspin
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.iscomplex = iscomplex
        self.Nsamples = Nsamples
        self.mftol = mftol
        self.mfepmax = mfepmax
        self.mfepmin = mfepmin
        self.Etol = Etol
        self.Ptol = Ptol
        self.nblocks = nblocks
        self.channels = channels
        self.kernel_size = kernel_size
        self.nnepmax = nnepmax
        self.d = d
        self.h = h


        if self.iscomplex:
            qtx.global_defs.set_default_dtype(jnp.complex128)
            self.dtype = jnp.complex128
        else:
            qtx.global_defs.set_default_dtype(jnp.float64)
            self.dtype=jnp.float64

        self._t = 0.
        self._intparam = {}

        if decouple_bath:
            n_coupled = norb
            n_decoupled = naux * norb
        else:
            n_coupled = (naux + 1) * norb
            n_decoupled = 0

        self.lattice = qtx.sites.Cluster(
            n_coupled=n_coupled,
            n_decoupled=n_decoupled, # total site will be n_coupled+n_decoupled
            Nparticle=((naux + 1) * norb // 2, (naux + 1) * norb // 2),
            is_fermion=True,
            double_occ=True, # whether allowing double occupation
        )

        self.mf_model = qtx.model.Pfaffian(dtype=self.dtype,)
        self.nn_model = qtx.model.RBM_Dense(
            features=self.channels,
            dtype=self.dtype
        )

        self.mf_state = qtx.state.Variational(
            self.mf_model, # 4 spin-up, 4 spin-down
            max_parallel=32768, # maximum forward batch on each machine
        )

        self.mf_sampler = qtx.sampler.NeighborExchange(self.mf_state,Nsamples)

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
    
    def to_natrual_orbital(self, T: np.array, intparams: Dict[str, float]):
        F,  D, _ = hartree_fock(
            h_mat=T,
            n_imp=self.norb,
            n_bath=self.naux*self.norb,
            nocc=(self.naux+1)*self.norb, # always half filled
            max_iter=100,
            ntol=1e-8,
            **intparams
        )

        D = D.reshape((self.naux+1)*self.norb,2,(self.naux+1)*self.norb,2)

        if self.nspin<4:
            assert np.abs(D[:,0,:,1]).max() < 1e-9

        if self.nspin == 1:
            err = np.abs(D[:,0,:,0]-D[:,1,:,1]).max()
            assert err < 1e-9, "spin symmetry breaking error {}".format(err)
        
        D = D.reshape((self.naux+1)*self.norb*2, (self.naux+1)*self.norb*2)

        new_T, D_meanfield, self.transmat = nao_two_chain(
                                    h_mat=F,
                                    D=D,
                                    n_imp=self.norb,
                                    n_bath=self.naux*self.norb,
                                    nspin=self.nspin,
                                )

        return new_T


    def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian
        if self.decouple_bath:
            assert T.shape[0] == (self.naux+1) * self.norb * 2
            # The first (self.norb, self.norb)*2 is the impurity, which is keeped unchange.
            # We do diagonalization of the other block
            T = self.cal_decoupled_bath(T=T)
        elif self.natural_orbital:
            assert T.shape[0] == (self.naux+1) * self.norb * 2
            T = self.to_natrual_orbital(T=T)


        intparam = copy.deepcopy(intparam)
        intparam["t"] = T.copy()
        self._t = T
        self._intparam = intparam

        _Hemb = Slater_Kanamori(
            nsites=self.norb*(self.naux+1),
            n_noninteracting=self.norb*self.naux,
            **intparam
        ) 

        self._Hemb = Operator(op_list=_Hemb._op_list)
        # we should notice that the spinorbital are not adjacent in the quspin hamiltonian, 
        # so properties computed from this need to be transformed.

        return self._Hemb
    
    def get_Hemb(self, T, intparam):
        if np.abs(self._t - T).max() > 1e-8 or self._intparam != intparam:
            return self._construct_Hemb(T, intparam)
        else:
            return self._Hemb
        
    def solve(self, T, intparam, return_RDM: bool=False):
        RDM = None
        Hemb = self.get_Hemb(T=T, intparam=intparam)
        # assert self.nspin == 1, "Currently, the quantax only support one known spin pairs."

        # first do the optimization of a mean-field determinant

        # new samples proposed by electron hopping
        tdvp = qtx.optimizer.TDVP(self.mf_state,Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-6))
        for i in tqdm(range(self.mfepmax), desc="Meanfield Pfaffian training: "):
            samples = self.mf_sampler.sweep()
            step = tdvp.get_step(samples)
            self.mf_state.update(step*0.01)
            
            VarE = tdvp.VarE
            if i > self.mfepmin:
                break

            if VarE < self.Etol:
                break
            else:
                if (i+1) % 250 == 0:
                    print("Current E var: {:.4f}".format(VarE))

        #TODO: The strategy to reuse the states of last iteration can be improved, do improve this.
        if VarE < self.mftol: # this suggest the mean-field is good:
            if VarE > self.Etol: # this suggest mean-field is not converged
                # wait for pfaffian
                # self.mf_state.save("/tmp/model.eqx")
                # F = eqx.tree_deserialise_leaves("/tmp/model.eqx",self.nn_model.layers[-1].F)
                # self.nn_model = eqx.tree_at(lambda tree: tree.layers[-1].F, self.nn_model, F)

                model = qtx.model.NeuralJastrow(self.nn_model, self.mf_model)
                self.nn_state = qtx.state.Variational(model,max_parallel=32768)
                self.nn_sampler = qtx.sampler.NeighborExchange(self.nn_state,self.Nsamples)
                tdvp = qtx.optimizer.TDVP(self.nn_state,self._Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-6))

                for i in tqdm(range(self.nnepmax), desc="NN Wave Function training (w mf): "):
                    samples = self.nn_sampler.sweep()
                    step = tdvp.get_step(samples)
                    self.nn_state.update(step*0.01)

                    VarE = tdvp.VarE

                    if VarE < self.Etol:
                        break
                    else:
                        if (i+1) % 250 == 0:
                            print("Current E var: {:.4f}".format(VarE))

                converged_model = "nn"
            else: # this means mf wave function have converged
                converged_model = "mf"
        else: # this means mf wave function is not good
            model = qtx.model.NeuralJastrow(self.nn_model, self.mf_model)
            self.nn_state = qtx.state.Variational(model,max_parallel=32768)
            self.nn_sampler = qtx.sampler.NeighborExchange(self.nn_state,self.Nsamples)
            tdvp = qtx.optimizer.TDVP(self.nn_state,self._Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-6))

            for i in tqdm(range(self.nnepmax), desc="NN Wave Function training (w/o mf): "):
                samples = self.nn_sampler.sweep()
                step = tdvp.get_step(samples)
                self.nn_state.update(step*0.01)

                VarE = tdvp.VarE

                if VarE < self.Etol:
                    break
                else:
                    if (i+1) % 250 == 0:
                        print("Current E var: {:.4f}".format(VarE))

            converged_model = "nn"
            
        print("Convergent variance of energy: {:.4f}".format(VarE))

        if converged_model == "nn":
            sampler = self.nn_sampler
            state = self.nn_state
        else:
            sampler = self.mf_sampler
            state = self.mf_state
        
        if return_RDM:
            nsites = self.norb*(self.naux+1)
            RDM = self.cal_RDM(state=state, sampler=sampler)

            if self.decouple_bath:
                RDM = self.recover_decoupled_bath(RDM)
            
            if self.natural_orbital:
                RDM = self.transmat.conj().T @ RDM @ self.transmat

        return RDM
    
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
    
    def cal_RDM(self, state, sampler):
        nsites = self.norb*(self.naux+1)
        # # compute RDM
        # tol = 1e-3
        # sampler.reset()
        Nsamples = 100000
        sampler = qtx.sampler.NeighborExchange(state,Nsamples, thermal_steps=nsites*100)
        converge = False
        count = 0.
        mean = np.zeros((nsites, 2, nsites, 2)) + 0j
        var = np.zeros(mean.shape)

        while not converge:
            samples = sampler.sweep()

            for a in range(nsites):
                for s in range(2):
                    for b in range(a, nsites):
                        if a == b:
                            start = s
                        else:
                            start = 0

                        for s_ in range(start,2):
                            if s == 0 and s_ == 0:
                                op = create_u(a) * annihilate_u(b)
                            elif s == 1 and s_ == 1:
                                op = create_d(a) * annihilate_d(b)
                            elif s == 0 and s_ == 1:
                                op = create_u(a) * annihilate_d(b)
                            else:
                                op = create_d(a) * annihilate_u(b)
                                
                            vmean, vvar = op.expectation(state, samples, return_var=True)
                            var[a,s,b,s_] = var[a,s,b,s_] + np.abs(mean[a,s,b,s_]) ** 2
                            var[b,s_,a,s] = var[a,s,b,s_].copy()
                            mean[a,s,b,s_] = count/(count+1) * mean[a,s,b,s_] + 1/(count+1) * vmean
                            mean[b,s_,a,s] = np.conjugate(mean[a,s,b,s_]).copy()
                            var[a,s,b,s_] = count/(count+1) * var[a,s,b,s_] + 1/(count+1) * (vvar + np.abs(vmean)**2) - np.abs(mean[a,s,b,s_]) ** 2
                            var[b,s_,a,s] = var[a,s,b,s_].copy()

            varmax = var.max()
            varmax = np.sqrt(varmax / (Nsamples * (count+1)))
            if varmax < self.Ptol:
                converge = True

            count += 1

            if count % 100 == 0:
                print("--- varmax: {:.4f}".format(varmax))
                print(np.linalg.eigvalsh(mean[:,0,:,0]), np.linalg.eigvalsh(mean[:,1,:,1]))

        # print("RDM convergence error: {:.4f}".format(max(vars)))
        # print(vars)
        if self.nspin <= 2:
            mean[:,0,:,1] = mean[:,1,:,0] = 0.

        mean = mean.reshape(2*nsites, 2*nsites)

        return mean