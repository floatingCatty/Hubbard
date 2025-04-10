import numpy as np
import copy
from typing import Dict
from gGA.operator import Slater_Kanamori, S_z, S_m, S_p
import io
try:
    from quantax.operator import create_u, create_d, annihilate_u, annihilate_d, Operator, number_u, number_d
    import quantax as qtx
    import jax.numpy as jnp
    import jax
    import equinox as eqx
    from tqdm import tqdm
except:
    if jax.process_index() == 0:
        print("The quantax & jax & equinox is not installed. One should not use NQS solver.")
from gGA.nao.hf import hartree_fock, compute_random_energy
from gGA.nao.tonao import nao_two_chain
from time import time
from .cluster import Cluster
from .graph_net import GTran


class NQS_solver(object):
    def __init__(
            self, 
            norb, 
            naux, 
            nspin, 
            decouple_bath: bool=False, 
            natural_orbital: bool=False, 
            iscomplex=False, 
            kBT: float=0.025,
            mutol: float=1e-4,
            # NQS setting
            nblocks=3,
            d_emb=4,
            hidden_channels=8,
            out_channels=8,
            ffn_hidden=[8],
            heads=4,
            Nsamples=1000, # number of samples in training
            mfepmax=500, # max epoch for mean-field wave-function training
            nnepmax=2000,
            Nmf=100000,
            Nnn=50000,
            Np=50000,
            Etol=1e-3, # variance tol for energy minimization
            Ptol=1e-4, # error tol for property evaluation, should be smaller than at least 5e-4 if expecting 1e-4 convergence
            ) -> None:

        self.norb = norb
        self.naux = naux
        self.nspin = nspin
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.iscomplex = iscomplex
        self.Nsamples = Nsamples
        self.mfepmax = mfepmax
        self.Etol = Etol
        self.Ptol = Ptol
        self.nblocks = nblocks
        self.out_channels = out_channels
        self.heads = heads
        self.nnepmax = nnepmax
        self.mutol = mutol
        self.kBT = kBT
        self.Nnn = Nnn
        self.Np = Np
        self.Nmf = Nmf

        self.device_count = jax.device_count()

        if self.iscomplex:
            qtx.global_defs.set_default_dtype(jnp.complex128)
            self.dtype = jnp.float64
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

        self.lattice = Cluster(
            n_coupled=n_coupled,
            n_decoupled=n_decoupled, # total site will be n_coupled+n_decoupled
            Nparticle=((naux + 1) * norb // 2, (naux + 1) * norb // 2),
            is_fermion=True,
            double_occ=True, # whether allowing double occupation
        )

        # self.nn_model = qtx.model.RBM_Dense(
        #     features=self.channels,
        #     dtype=self.dtype
        # )

        self.mf_model = qtx.model.Pfaffian(dtype=self.dtype,)
        self.mf_state = qtx.state.Variational(
            self.mf_model, # 4 spin-up, 4 spin-down
            max_parallel=[self.Nmf*self.device_count,self.Nmf*self.device_count,self.Nmf*self.device_count], # maximum forward batch on each machine
        )

        self.mf_sampler = qtx.sampler.NeighborExchange(self.mf_state,self.Nsamples)
        nn_model = GTran(
            nblocks=nblocks,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            ffn_hidden=ffn_hidden,
            d_emb=d_emb,
            heads=heads,
            final_activation=lambda x: x,
            dtype=self.dtype,
        )

        self.nn_model = qtx.model.HiddenPfaffian(pairing_net=nn_model, dtype=self.dtype)
        

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
            max_iter=500,
            ntol=self.mutol,
            kBT=self.kBT,
            tol=1e-6,
            **intparams,
            verbose=False,
        )

        D = D.reshape((self.naux+1)*self.norb,2,(self.naux+1)*self.norb,2)

        if self.nspin<4:
            assert np.abs(D[:,0,:,1]).max() < 1e-9

        if self.nspin == 1:
            err = np.abs(D[:,0,:,0]-D[:,1,:,1]).max()
            assert err < 1e-9, "spin symmetry breaking error {}".format(err)
        
        D = D.reshape((self.naux+1)*self.norb*2, (self.naux+1)*self.norb*2)

        _, D_meanfield, self.transmat = nao_two_chain(
                                    h_mat=F,
                                    D=D,
                                    n_imp=self.norb,
                                    n_bath=self.naux*self.norb,
                                    nspin=self.nspin,
                                )
        new_T = self.transmat @ T @ self.transmat.conj().T

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
            T = self.to_natrual_orbital(T=T, intparams=intparam)


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
        
    def solve(self, T, intparam, return_RDM: bool=False, return_S2: bool=False):
        RDM = None
        Hemb = self.get_Hemb(T=T, intparam=intparam)
        # assert self.nspin == 1, "Currently, the quantax only support one known spin pairs."

        # first do the optimization of a mean-field determinant
        # new samples proposed by electron hopping
        Einf = compute_random_energy(
            nocc=(self.naux+1)*self.norb, 
            n_bath=self.naux*self.norb, 
            n_imp=self.norb, 
            h_mat=T,
            **self._intparam
            )
        
        _, _, Ehf = hartree_fock(
            h_mat=T,
            n_imp=self.norb,
            n_bath=self.naux*self.norb,
            nocc=(self.naux+1)*self.norb, # always half filled
            max_iter=500,
            ntol=self.mutol,
            kBT=self.kBT,
            tol=1e-6,
            **self._intparam,
            verbose=False,
        )
        if jax.process_index() == 0:
            print("--- Einf: {:.4f}, Ehf: {:.4f}".format(Einf, Ehf))


        tdvp = qtx.optimizer.TDVP(self.mf_state,Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-6), kazcmarz_mu=0.5)
        if jax.process_index() == 0:
            iterator = tqdm(range(self.mfepmax), desc="Meanfield Pfaffian training: ")
        else:
            iterator = range(self.mfepmax)
        for i in iterator:
            samples = self.mf_sampler.sweep()
            step = tdvp.get_step(samples)
            self.mf_state.update(step*0.01)

            VarE = tdvp.VarE
            E = tdvp.energy
            N = self.lattice.N

            Vscore = VarE*N / (E - Einf)**2

            if Vscore < self.Etol:
                break
            else:
                if (i+1) % 250 == 0:
                    if jax.process_index() == 0:
                        print("Current Evar: {:.4f}, Vscore: {:.4f}, Etot: {:.4f}".format(VarE, Vscore, E))
        #TODO: The strategy to reuse the states of last iteration can be improved, do improve this.
        if Vscore > self.Etol: # this suggest mean-field is not converged
            # wait for pfaffian

            # if jax.process_index() == 0:
                # self.mf_state.save("/tmp/model.eqx")
            # F = eqx.tree_deserialise_leaves("/tmp/model.eqx",self.nn_model.layers[-1].F)

            # serialized_state = eqx.tree_serialise_leaves(self.mf_state)
            # buffer = io.BytesIO(serialized_state)
            # F = eqx.tree_deserialise_leaves(buffer,self.nn_model.layers[-1].F)
            self.nn_model = eqx.tree_at(lambda tree: tree.layers[-1].F, self.nn_model, self.mf_state.model.F)
            self.nn_state = qtx.state.Variational(self.nn_model,max_parallel=[self.Nnn * self.device_count,self.Nnn * self.device_count,self.Nnn * self.device_count])
            self.nn_sampler = qtx.sampler.NeighborExchange(self.nn_state,self.Nsamples)
            tdvp = qtx.optimizer.TDVP(self.nn_state,Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-8), kazcmarz_mu=0.5)

            if jax.process_index() == 0:
                iterator = tqdm(range(self.nnepmax), desc="Neural Pfaffian training: ")
            else:
                iterator = range(self.nnepmax)

            for i in iterator:
                # start = time()
                samples = self.nn_sampler.sweep()
                # end = time() - start
                # print("Time for sampling: ", end- start)
                step = tdvp.get_step(samples)
                self.nn_state.update(step*0.01)
                # print("Time for update: ", time() - end)

                VarE = tdvp.VarE
                E = tdvp.energy
                N = self.lattice.N

                Vscore = VarE*N / (E - Einf)**2
                print(Vscore)

                if Vscore < self.Etol:
                    break
                else:
                    if (i+1) % 250 == 0:
                        if jax.process_index() == 0:
                            print("Current Evar: {:.4f}, Vscore: {:.4f}, Etot: {:.4f}".format(VarE, Vscore, E))

            converged_model = "nn"
        else: # this means mf wave function have converged
            converged_model = "mf"
        
        if jax.process_index() == 0:
            print("#### Convergent variance of energy: {:.4f}, energy: {:.4f}".format(Vscore, E))

        if converged_model == "nn":
            self.state = self.nn_state
        else:
            self.state = self.mf_state
        
        Pout = self.Psample(state=self.state, RDM=return_RDM, S2=return_S2)
        if return_RDM:
            # self._RDM = self.cal_RDM(state=state)
            self._RDM = Pout["RDM"]
            if self.decouple_bath:
                self._RDM = self.recover_decoupled_bath(self._RDM)
            
            if self.natural_orbital:
                self._RDM = self.transmat.conj().T @ self._RDM @ self.transmat
        if return_S2:
            self._S2 = Pout["S2"]

        return self._RDM
    
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
    
    def Psample(self, state, RDM=False, S2=False, E=False, DOCC=False):

        if not RDM and not S2 and not E and not DOCC:
            return True
        
        nsites = self.norb*(self.naux+1)
        Nsamples = self.Np * self.device_count
        start = time()
        sampler = qtx.sampler.NeighborExchange(state, Nsamples, thermal_steps=nsites*50)
        end = time()
        if jax.process_index() == 0:
            print("time for sampler thermalization: ", end-start)
        converge = {}
        count = 0.

        if RDM:
            rdm_mean = np.zeros((nsites, 2, nsites, 2)) + 0j
            rdm_var = np.ones(rdm_mean.shape)
            rdm_varmax = rdm_var.max()
            converge["RDM"] = False
        if S2:
            s2_mean = 0.
            s2_var = 1.
            s2_varmax = s2_var
            S2 = self._build_S2()
            converge["S2"] = False

        while not all(list(converge.values())):
            start = time()
            samples = sampler.sweep()
            end = time()

            if jax.process_index() == 0:
                print("time for sample: ", end-start)

            if RDM and rdm_varmax > self.Ptol:
                rdm_mean, rdm_var = self._cal_RDM(state=state, samples=samples, mean=rdm_mean, var=rdm_var, count=count)
                rdm_varmax = rdm_var.max()
                rdm_varmax = np.sqrt(rdm_varmax / (Nsamples * (count+1)))
                print("Sample Iter.{}: Current RDM stdmax: {:.6f}".format(count, rdm_varmax))
            else:
                converge["RDM"] = True
            
            if S2 and s2_varmax > self.Ptol:
                s2_mean, s2_var = self._cal_S2(S2=S2, state=state, samples=samples, mean=s2_mean, var=s2_var, count=count)
                s2_varmax = np.sqrt(s2_varmax / (Nsamples * (count+1)))
                print("Sample Iter.{}: Current S2 stdmax: {:.6f}".format(count, s2_varmax))
            else:
                converge["S2"] = True

            count += 1

        # postprocessing
        out = {}
        if RDM:
            if self.nspin == 1:
                sym = 0.5 * (rdm_mean[:,1,:,1] + rdm_mean[:,0,:,0])
                rdm_mean[:,0,:,1] = rdm_mean[:,1,:,0] = 0
                rdm_mean[:,1,:,1] = sym
                rdm_mean[:,0,:,0] = sym
            elif self.nspin == 2:
                rdm_mean[:,0,:,1] = rdm_mean[:,1,:,0] = 0
            rdm_mean = rdm_mean.reshape(2*nsites, 2*nsites)
            # clip the possible negative value and value larger than one.
            vals, vecs = np.linalg.eigh(rdm_mean)
            if np.sum(vals<0) > 0:
                if vals[vals<0].min() < -3*self.Ptol: 
                    """according to probability, the mean value deviate by 3 x std only have probability 0.03%, 2x for 95%, 1x for 68%"""
                    raise ValueError("#### The RDM calculation does not converge !, decrease Ptol or improve sampler!")
                
            if np.sum(vals>1) > 0:
                if vals[vals>1].max() > (1+3*self.Ptol):
                    raise ValueError("#### The RDM calculation does not converge !, decrease Ptol or improve sampler!")

            if np.sum(vals<0) > 0 or np.sum(vals>0) > 0:
                vals[vals<0] = self.Ptol
                vals[vals>1] = 1-self.Ptol
                rdm_mean = (vecs*vals[None,:]) @ vecs.conj().T

            out["RDM"] = rdm_mean
        
        if S2:
            assert s2_mean > -3*self.Ptol, "The S2 expecation cannot be more negative than the tolerant margin, try improve the sampler!"

            out["S2"] = s2_mean

        return out
        
    def _cal_RDM(self, state, samples, mean, var, count):
        nsites = self.norb*(self.naux+1)
        start_eval = time()

        for a in range(nsites):
            for b in range(a, nsites):
                if self.nspin == 1:
                    ss_set = [(0,0), (1,1)]
                elif self.nspin == 2:
                    ss_set = [(0,0), (1,1)]
                else:
                    if a == b:
                        ss_set = [(0,0),(1,1),(0,1)]
                    else:
                        ss_set = [(0,0),(1,1),(0,1),(1,0)]

                for s,s_ in ss_set:
                    if s == 0 and s_ == 0:
                        if a == b:
                            op = number_u(a)
                        else:
                            op = create_u(a) * annihilate_u(b)
                    elif s == 1 and s_ == 1:
                        if a == b:
                            op = number_d(a)
                        else:
                            op = create_d(a) * annihilate_d(b)
                    elif s == 0 and s_ == 1:
                        op = create_u(a) * annihilate_d(b)
                    else:
                        op = create_d(a) * annihilate_u(b)
                    
                    vmean, vvar = op.expectation(state, samples, return_var=True)
                    # vmean = state @ op @ state
                    var[a,s,b,s_] = var[a,s,b,s_] + np.abs(mean[a,s,b,s_]) ** 2
                    var[b,s_,a,s] = var[a,s,b,s_].copy()
                    mean[a,s,b,s_] = count/(count+1) * mean[a,s,b,s_] + 1/(count+1) * vmean
                    mean[b,s_,a,s] = mean[a,s,b,s_].conj().copy()
                    var[a,s,b,s_] = count/(count+1) * var[a,s,b,s_] + 1/(count+1) * (vvar + np.abs(vmean)**2) - np.abs(mean[a,s,b,s_]) ** 2
                    var[b,s_,a,s] = var[a,s,b,s_].copy()

        if jax.process_index() == 0:
            print("--- time for evaluation: ", time()-start_eval)

        return mean, var
    
    def _build_S2(self):
        nsites = self.norb*(self.naux+1)

        S_m_ = sum(S_m(nsites, i) for i in range(self.norb))
        S_p_ = sum(S_p(nsites, i) for i in range(self.norb))
        S_z_ = sum(S_z(nsites, i) for i in range(self.norb))

        S2 = S_m_ * S_p_ + S_z_ * S_z_ + S_z_

        S2 = Operator(op_list=S2._op_list)

        return S2
    
    def _cal_S2(self, S2, state, samples, mean, var, count):
        
        start_eval = time()
        vmean, vvar = S2.expectation(state, samples, return_var=True)

        var = var + np.abs(mean) ** 2
        mean = count/(count+1) * mean + 1/(count+1) * vmean
        var = count/(count+1) * var + 1/(count+1) * (vvar + np.abs(vmean)**2) - np.abs(mean) ** 2

        if jax.process_index() == 0:
            print("--- time for evaluation: ", time()-start_eval)

        return mean, var
    
    # def cal_RDM(self, state, mean, var):
    #     """
    #         1. Why the spin exchange hopping have non-zero expecation in a spin conserved system?
    #         2. spin symmtrization broken.
    #         3. 
    #     """
    #     # state = state.todense()
    #     # state.normalize()
    #     nsites = self.norb*(self.naux+1)
    #     # # compute RDM
    #     # tol = 1e-3
    #     # sampler.reset()
    #     Nsamples = 50000 * jax.device_count()
    #     start = time()
    #     sampler = qtx.sampler.NeighborExchange(state, Nsamples, thermal_steps=nsites*50)
    #     end = time()
    #     if jax.process_index() == 0:
    #         print("time for sampler thermalization: ", end-start)
    #     converge = False
    #     count = 0.
    #     mean = np.zeros((nsites, 2, nsites, 2)) + 0j
    #     var = np.zeros(mean.shape)

    #     while not converge:
    #         start = time()
    #         samples = sampler.sweep()
    #         end = time()
    #         if jax.process_index() == 0:
    #             print("time for sample: ", end-start)

    #         for a in range(nsites):
    #             for b in range(a, nsites):
    #                 if self.nspin == 1:
    #                     ss_set = [(0,0), (1,1)]
    #                 elif self.nspin == 2:
    #                     ss_set = [(0,0), (1,1)]
    #                 else:
    #                     if a == b:
    #                         ss_set = [(0,0),(1,1),(0,1)]
    #                     else:
    #                         ss_set = [(0,0),(1,1),(0,1),(1,0)]

    #                 for s,s_ in ss_set:
    #                     if s == 0 and s_ == 0:
    #                         if a == b:
    #                             op = number_u(a)
    #                         else:
    #                             op = create_u(a) * annihilate_u(b)
    #                     elif s == 1 and s_ == 1:
    #                         if a == b:
    #                             op = number_d(a)
    #                         else:
    #                             op = create_d(a) * annihilate_d(b)
    #                     elif s == 0 and s_ == 1:
    #                         op = create_u(a) * annihilate_d(b)
    #                     else:
    #                         op = create_d(a) * annihilate_u(b)
                        
    #                     vmean, vvar = op.expectation(state, samples, return_var=True)
    #                     # vmean = state @ op @ state
    #                     var[a,s,b,s_] = var[a,s,b,s_] + np.abs(mean[a,s,b,s_]) ** 2
    #                     var[b,s_,a,s] = var[a,s,b,s_].copy()
    #                     mean[a,s,b,s_] = count/(count+1) * mean[a,s,b,s_] + 1/(count+1) * vmean
    #                     mean[b,s_,a,s] = mean[a,s,b,s_].conj().copy()
    #                     var[a,s,b,s_] = count/(count+1) * var[a,s,b,s_] + 1/(count+1) * (vvar + np.abs(vmean)**2) - np.abs(mean[a,s,b,s_]) ** 2
    #                     var[b,s_,a,s] = var[a,s,b,s_].copy()

    #         if jax.process_index() == 0:
    #             print("--- time for evaluation: ", time()-end)
    #         varmax = var.max()
    #         varmax = np.sqrt(varmax / (Nsamples * (count+1)))
    #         if varmax < self.Ptol:
    #             converge = True

    #         if count % 5 == 0:
    #             if jax.process_index() == 0:
    #                 print("--- varmax: {:.5f}".format(varmax))
    #             if self.nspin == 1:
    #                 if jax.process_index() == 0:
    #                     print(np.linalg.eigvalsh(0.5*(mean[:,0,:,0]+mean[:,1,:,1])))
    #             elif self.nspin == 2:
    #                 if jax.process_index() == 0:
    #                     print(np.linalg.eigvalsh(mean[:,0,:,0]), np.linalg.eigvalsh(mean[:,1,:,1]))
    #             else:
    #                 if jax.process_index() == 0:
    #                     print(np.linalg.eigvalsh(mean.reshape(2*nsites, 2*nsites)))

    #         count += 1

    #     if self.nspin == 1:
    #         sym = 0.5 * (mean[:,1,:,1] + mean[:,0,:,0])
    #         mean[:,0,:,1] = mean[:,1,:,0] = 0
    #         mean[:,1,:,1] = sym
    #         mean[:,0,:,0] = sym
    #     elif self.nspin == 2:
    #         mean[:,0,:,1] = mean[:,1,:,0] = 0
    #     mean = mean.reshape(2*nsites, 2*nsites)
    #     # clip the possible negative value and value larger than one.
    #     vals, vecs = np.linalg.eigh(mean)
    #     if np.sum(vals<0) > 0:
    #         if vals[vals<0].min() < -2*self.Ptol:
    #             raise ValueError("#### The RDM calculation does not converge !, decrease Ptol")
            
    #     if np.sum(vals>1) > 0:
    #         if vals[vals>1].max() > (1+2*self.Ptol):
    #             raise ValueError("#### The RDM calculation does not converge !, decrease Ptol")

    #     if np.sum(vals<0) > 0 or np.sum(vals>0) > 0:
    #         vals[vals<0] = 1e-4
    #         vals[vals>1] = 1-1e-4
    #         mean = (vecs*vals[None,:]) @ vecs.conj().T
        
    #     return mean

    @property
    def RDM(self):
        if hasattr(self, "_RDM"):
            return self._RDM
        else:
            if hasattr(self, "state"):
                return self.Psample(state=self.state, RDM=True)["RDM"]
            else:
                raise RuntimeError("The solver have not solve any model!")
    
    @property
    def S2(self):
        if hasattr(self, "_S2"):
            return self._S2
        else:
            if hasattr(self, "state"):
                return self.Psample(state=self.state, S2=True)["S2"]
            else:
                raise RuntimeError("The solver have not solve any model!")
            