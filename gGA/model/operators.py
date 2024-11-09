import torch
from itertools import combinations_with_replacement
from gGA.utils.kronsum import sparse_kron

#0, up, down, updown

anni_up = torch.tensor([
    [0, 1, 0, 0],
    [0, 0.,0, 0],
    [0, 0, 0,-1],
    [0, 0, 0, 0]
])

anni_down = torch.tensor([
    [0, 0, 1, 0],
    [0, 0.,0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])


def get_anni(s):
    if s == 0:
        return anni_up
    elif s == 1:
        return anni_down

dm_up = anni_up.H @ anni_up
dm_down = anni_down.H @ anni_down

# def generate_basis(norb, nocc):
#     state_labels = [0, 1, 2, 3]
#     state_electrons = {0: 0, 1: 1, 2: 1, 3: 2}
#     basis_states = []

#     def recursive_generate(orbital_index, current_config, current_electrons):
#         if orbital_index == norb:
#             if current_electrons == nocc:
#                 basis_states.append(current_config.copy())
#             return

#         for state in state_labels:
#             electrons = state_electrons[state]
#             new_total = current_electrons + electrons

#             if new_total > nocc:
#                 continue  # Prune branches exceeding electron count

#             current_config.append(state)
#             recursive_generate(orbital_index + 1, current_config, new_total)
#             current_config.pop()  # Backtrack

#     recursive_generate(0, [], 0)

#     return basis_states

def generate_product_basis_indices(norb_A, norb_B, nocc_total):
    state_labels = [0, 1, 2, 3]
    state_electrons = {0: 0, 1: 1, 2: 1, 3: 2}

    def generate_basis_indices_and_occupations(norb, nocc_max):
        basis_indices_and_occupations = []
        index = 0
        stack = []
        # Each item in stack: (orbital_index, current_electrons)
        stack.append((0, 0))  # Start with the first orbital, zero electrons

        while stack:
            orbital_index, current_electrons = stack.pop()
            if orbital_index == norb:
                # We have a complete configuration
                basis_indices_and_occupations.append((index, current_electrons))
                index += 1
                continue
            for state in state_labels:
                electrons = state_electrons[state]
                new_total = current_electrons + electrons
                if new_total > nocc_max:
                    continue  # Prune branches exceeding maximum electron count
                # Early pruning based on possible electrons remaining
                remaining_orbitals = norb - (orbital_index + 1)
                min_possible_electrons = new_total + remaining_orbitals * min(state_electrons.values())
                max_possible_electrons = new_total + remaining_orbitals * max(state_electrons.values())
                if min_possible_electrons > nocc_max or max_possible_electrons < 0:
                    continue  # Prune branches that can't sum to valid electron counts
                stack.append((orbital_index + 1, new_total))
        return basis_indices_and_occupations

    def build_occ_to_indices_map(basis_indices_and_occupations):
        occ_to_indices = {}
        for index, nocc in basis_indices_and_occupations:
            occ_to_indices.setdefault(nocc, []).append(index)
        return occ_to_indices

    # Generate basis indices and occupation numbers for subsystems A and B
    # For A and B, nocc_max is the maximum possible occupation number (limited by nocc_total or norb * 2)
    nocc_max_A = min(nocc_total, norb_A * 2)
    nocc_max_B = min(nocc_total, norb_B * 2)

    basis_A = generate_basis_indices_and_occupations(norb_A, nocc_max_A)
    basis_B = generate_basis_indices_and_occupations(norb_B, nocc_max_B)

    # Build occupation number to indices mapping
    occ_to_indices_A = build_occ_to_indices_map(basis_A)
    occ_to_indices_B = build_occ_to_indices_map(basis_B)

    # Now, generate the pairs of indices (i, j) where the total occupation number is nocc_total
    indices_pairs = []
    for nocc_A_sub in occ_to_indices_A:
        nocc_B_sub = nocc_total - nocc_A_sub
        if nocc_B_sub in occ_to_indices_B:
            indices_A = occ_to_indices_A[nocc_A_sub]
            indices_B = occ_to_indices_B[nocc_B_sub]
            # Generate all combinations of indices
            for i in indices_A:
                for j in indices_B:
                    indices_pairs.append((i, j))
    return indices_pairs


def generate_basis(norb, nocc):
    state_labels = [0, 1, 2, 3]
    state_electrons = {0: 0, 1: 1, 2: 1, 3: 2}
    basis_states = []

    stack = []
    # Initialize the stack with the starting state
    # We use a tuple (orbital_index, current_config, current_electrons)
    stack.append((0, [], 0))

    while stack:
        orbital_index, current_config, current_electrons = stack.pop()

        # Early pruning: if the minimum possible electrons from here exceed nocc, skip
        remaining_orbitals = norb - orbital_index
        min_possible_electrons = current_electrons + remaining_orbitals * min(state_electrons.values())
        max_possible_electrons = current_electrons + remaining_orbitals * max(state_electrons.values())

        if min_possible_electrons > nocc or max_possible_electrons < nocc:
            continue  # Prune branches that can't possibly sum to nocc

        if orbital_index == norb:
            if current_electrons == nocc:
                basis_states.append(list(current_config))  # Use tuple for efficiency
            continue

        for state in state_labels:
            electrons = state_electrons[state]
            new_total = current_electrons + electrons

            if new_total > nocc:
                continue  # Prune branches exceeding electron count

            # Instead of copying the list, we pass the new element separately
            stack.append((orbital_index + 1, current_config + [state], new_total))

    return basis_states


def generate_product_basis(norb_A, nocc_A, norb_B, nocc_B):
    state_labels = [0, 1, 2, 3]
    state_electrons = {0: 0, 1: 1, 2: 1, 3: 2}

    total_norb = norb_A + norb_B
    total_nocc = nocc_A + nocc_B

    basis_states = []

    stack = []
    # Initialize the stack with the starting state
    # (orbital_index, current_config, total_electrons, electrons_in_A, electrons_in_B)
    stack.append((0, [], 0, 0, 0))

    while stack:
        orbital_index, current_config, total_electrons, electrons_in_A, electrons_in_B = stack.pop()

        # Early pruning based on total electrons
        remaining_orbitals = total_norb - orbital_index
        min_possible_electrons = total_electrons + remaining_orbitals * min(state_electrons.values())
        max_possible_electrons = total_electrons + remaining_orbitals * max(state_electrons.values())

        if min_possible_electrons > total_nocc or max_possible_electrons < total_nocc:
            continue  # Prune branches that can't sum to total_nocc

        # Early pruning based on electrons in subsystem A
        if orbital_index < norb_A:
            remaining_orbitals_A = norb_A - orbital_index
            min_electrons_A = electrons_in_A + remaining_orbitals_A * min(state_electrons.values())
            max_electrons_A = electrons_in_A + remaining_orbitals_A * max(state_electrons.values())

            if min_electrons_A > nocc_A or max_electrons_A < nocc_A:
                continue  # Prune branches that can't sum to nocc_A

        # Early pruning based on electrons in subsystem B
        else:
            remaining_orbitals_B = norb_A + norb_B - orbital_index
            min_electrons_B = electrons_in_B + remaining_orbitals_B * min(state_electrons.values())
            max_electrons_B = electrons_in_B + remaining_orbitals_B * max(state_electrons.values())

            if min_electrons_B > nocc_B or max_electrons_B < nocc_B:
                continue  # Prune branches that can't sum to nocc_B

        if orbital_index == total_norb:
            if total_electrons == total_nocc and electrons_in_A == nocc_A and electrons_in_B == nocc_B:
                basis_states.append(list(current_config))
            continue

        for state in state_labels:
            electrons = state_electrons[state]
            new_total_electrons = total_electrons + electrons

            # Prune if total electrons exceed the limit
            if new_total_electrons > total_nocc:
                continue

            if orbital_index < norb_A:
                new_electrons_in_A = electrons_in_A + electrons
                new_electrons_in_B = electrons_in_B
                # Prune if electrons in A exceed the limit
                if new_electrons_in_A > nocc_A:
                    continue
            else:
                new_electrons_in_B = electrons_in_B + electrons
                new_electrons_in_A = electrons_in_A
                # Prune if electrons in B exceed the limit
                if new_electrons_in_B > nocc_B:
                    continue

            # Append the new state to the stack
            stack.append((
                orbital_index + 1,
                current_config + [state],
                new_total_electrons,
                new_electrons_in_A,
                new_electrons_in_B
            ))

    return basis_states


def generate_basis_minimized(norb, nocc):

    # Possible states per orbital and their electron counts
    state_labels = [0, 1, 2, 3]
    state_electrons = {0: 0, 1: 1, 2: 1, 3: 2}

    # Generate all possible multisets (combinations with replacement)
    basis_states = []
    for states in combinations_with_replacement(state_labels, norb):
        # Calculate the total number of electrons
        total_electrons = sum(state_electrons[state] for state in states)
        if total_electrons == nocc:
            basis_states.append(list(states))

    basis_states = torch.tensor(basis_states)
    indices = (4 ** torch.arange(basis_states.shape[1]-1, -1, -1)).unsqueeze(0) * basis_states
    indices = torch.sum(indices, dim=1)

    return basis_states, indices

def states_to_indices(basis_states):
    if not isinstance(basis_states, torch.Tensor):
        basis_states = torch.tensor(basis_states)
    indices = (4 ** torch.arange(basis_states.shape[1]-1, -1, -1).to(basis_states.device)).unsqueeze(0) * basis_states
    indices = torch.sum(indices, dim=1)

    return indices


def annihilation(oidx, spin, state):
    if state is None:
        return None, 0
    
    new_state = list(state)
    transitions = {
        (1, 'up'): (0, 1),      # From spin-up to empty
        (2, 'down'): (0, 1),    # From spin-down to empty
        (3, 'up'): (2, -1),   # Remove spin-up from double occupancy
        (3, 'down'): (1, 1) # Remove spin-down from double occupancy
    }

    if transitions.get((state[oidx], spin)) is not None:
        st, factor = transitions[(state[oidx], spin)]
        new_state[oidx] = st
        
        return new_state, factor
    else:
        return None, 0
    
def creation(oidx, spin, state):

    if state is None:
        return None, 0
    
    new_state = list(state)
    transitions = {
        (0, 'up'): (1, 1),      # From empty to spin-up
        (0, 'down'): (2, 1),    # From empty to spin-down
        (1, 'down'): (3, 1),      # From spin-up to double occupancy
        (2, 'up'): (3, 1)     # From spin-down to double occupancy
    }

    if transitions.get((state[oidx], spin)) is not None:
        st, factor = transitions[(state[oidx], spin)]
        new_state[oidx] = st
        
        return new_state, factor
    else:
        return None, 0

def anni_creates(oidx, ojdx, ispin, jspin, state):
    new_state, factor1 = annihilation(oidx, ispin, state)
    new_state, factor2 = creation(ojdx, jspin, new_state)

    return new_state, factor1 * factor2


def annihilation_operator(oidx, spin, states):
    row = []
    col = []
    data = []
    state_to_index = {tuple(state): idx for idx, state in enumerate(states)}
    for old_state in states:
        new_state, factor = annihilation(oidx, spin, old_state)
        if new_state is not None and state_to_index.get(tuple(new_state)) is not None:
            row.append(state_to_index[tuple(new_state)])
            col.append(state_to_index[tuple(old_state)])
            data.append(factor)

    return torch.sparse_coo_tensor(torch.tensor([row, col]), torch.tensor(data), (len(states), len(states)))

def anni_creates_operator(oidx, ojdx, ispin, jspin, states):
    row = []
    col = []
    data = []
    state_to_index = {tuple(state): idx for idx, state in enumerate(states)}
    for old_state in states:
        new_state, factor = anni_creates(oidx, ojdx, ispin, jspin, old_state)
        if new_state is not None and state_to_index.get(tuple(new_state)) is not None:
            row.append(state_to_index[tuple(new_state)])
            col.append(state_to_index[tuple(old_state)])
            data.append(factor)

    return torch.sparse_coo_tensor(torch.tensor([row, col]), torch.tensor(data), (len(states), len(states)))

def generate_annihilation_creations(max_norb):
    operators = {}

    for norb in range(1, max_norb+1):
        for i in range(norb):
            for s in range(2):
                mat_list = []
                for k in range(norb):
                    if k == i and s == 0:
                        mat_list.append(anni_up.to_sparse())
                    elif k == i and s == 1:
                        mat_list.append(anni_down.to_sparse())
                    else:
                        mat_list.append(torch.eye(4).to_sparse())
                
                operators[(norb,i,s)] = mat_list[0]
                for k in range(1,norb):
                    operators[(norb,i,s)] = sparse_kron(operators[(norb,i,s)].coalesce(), mat_list[k])
    
    return operators

annis = generate_annihilation_creations(9)