
# this is a hartree fock solution of a single site impurity model
# Here we consider the Slater-Kanamori Hamiltonian
# crystal field maybe added on to t_imp term.
def hf(t_imp, V_ib, t_bath, U, Up, J, Jp):
    # t_imp: impurity hopping
    # V_ib: impurity-bath hopping
    # t_bath: bath hopping
    # U_imp: impurity interaction

    nimp = t_imp.shape[0] // 2
    nbath = t_bath.shape[0] // 2
    assert V_ib.shape == (nimp*2, nbath*2)


    # if J and Jp is 0, use the usual basis
    # if not, use the Bogoliubov-de Gennes (BdG) framework for the inpurity part, normal basis to the bath part

    
    pass