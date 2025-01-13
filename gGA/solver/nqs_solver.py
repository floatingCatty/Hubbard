import numpy as np
import copy
from typing import Dict
from gGA.operator import Slater_Kanamori
from quantax.operator import create_u, create_d, annihilate_u, annihilate_d, Operator
import jax.numpy as jnp
import quantax as qtx
from tqdm import tqdm
import equinox as eqx

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

    def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian

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
        assert self.nspin == 1, "Currently, the quantax only support one known spin pairs."

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
                tdvp = qtx.optimizer.TDVP(self.nn_state,self._Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-12))

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
            tdvp = qtx.optimizer.TDVP(self.nn_state,self._Hemb,solver=qtx.optimizer.auto_pinv_eig(rtol=1e-12))

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
            RDM = self.cal_RDM(state=state, sampler=sampler)

        return RDM
    
    def cal_RDM(self, state, sampler):
        nsites = self.norb*(self.naux+1)
        # # compute RDM
        # tol = 1e-3
        RDM = np.zeros((nsites, 2, nsites, 2)) + 0j
        # sampler = qtx.sampler.NeighborExchange(state,Nsamples, thermal_steps=nsites*2*20)
        vars = []

        for a in range(nsites):
            for s in range(2):
                for b in range(a, (self.naux+1) * self.norb):
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

                        converge = False

                        old_mean = 0.
                        count = 0.
                        while not converge:
                            samples = sampler.sweep()
                            vmean = op.expectation(state, samples)
                            new_mean = count/(count+1) * old_mean + 1/(count+1) * vmean

                            if abs(new_mean - old_mean) < self.Ptol:
                                break
                            else:
                                old_mean = new_mean

                            count += 1

                        RDM[a, s, b, s_] = new_mean
                        RDM[b, s_, a, s] = jnp.conjugate(new_mean)

                        # vars.append(vvar/(Nsamples**0.5))
                        # print(vvar/(Nsamples**0.5))
                        
                        # while not converge:
                        #     samples = sampler.sweep()
                        #     vmean, vvar = op.expectation(state, samples, return_var=True)
                        #     # vmean = jnp.mean(vs)
                        #     # vvar = jnp.mean(jnp.abs(vs) ** 2) - jnp.abs(vmean) ** 2
                        #     print(vvar/(Nsamples**0.5))

                        #     if vvar/(Nsamples**0.5) > tol:
                        #         Nsamples *= 2
                        #         sampler = qtx.sampler.NeighborExchange(state,Nsamples)
                        #     else:
                        #         converge = True

                        #     if Nsamples > 1e6:
                        #         raise RuntimeError("The expecation of RDM does not converge with tol={:.4f} when Nsample topped, The var is {:.4f}".format(tol, vvar))

                        # if converge:
                        #     RDM[a, s, b, s_] = vmean
                        #     RDM[b, s_, a, s] = vmean
                        # else:
                        #     raise RuntimeError("The expecation of RDM does not converge with tol={:.4f}. The var is {:.4f}".format(tol, vvar))
        # print("RDM convergence error: {:.4f}".format(max(vars)))
        # print(vars)
        return RDM.reshape(2*nsites, 2*nsites)