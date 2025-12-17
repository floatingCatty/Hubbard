import numpy as np
from dataclasses import dataclass
from typing import List, Iterable, Tuple

@dataclass
class LatticeBond:
    s1: int
    s2: int
    x1: float
    y1: float
    x2: float
    y2: float
    bond_type: str



def nnn_square_lattice(nx: int, ny: int, yperiodic: bool = False) -> List[LatticeBond]:
    """
    Construct a square lattice with nearest-neighbor (nn) and
    next-nearest-neighbor (nnn) bonds.

    Site indices s1, s2 are 0-based in this Python version.
    """
    # only allow y-periodic if Ny > 2
    yperiodic = yperiodic and (ny > 2)

    N = nx * ny


    lattice: List[LatticeBond] = []

    for n in range(1, N + 1):
        x = (n - 1) // ny + 1
        y = (n - 1) % ny + 1

        # --- Nearest neighbors (nn) ---

        # Right neighbor in x-direction
        if x < nx:
            s1 = n - 1
            s2 = n + ny - 1
            lattice.append(
                LatticeBond(
                    s1=s1,
                    s2=s2,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + 1),
                    y2=float(y),
                    bond_type="nn",
                )
            )

        # Vertical neighbors
        if ny > 1:
            # Upwards in y-direction
            if y < ny:
                s1 = n - 1
                s2 = n + 1 - 1
                lattice.append(
                    LatticeBond(
                        s1=s1,
                        s2=s2,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x),
                        y2=float(y + 1),
                        bond_type="nn",
                    )
                )

            # Periodic wrap in y
            if yperiodic and y == 1:
                s1 = n - 1
                s2 = n + ny - 1 - 1
                lattice.append(
                    LatticeBond(
                        s1=s1,
                        s2=s2,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x),
                        y2=float(y + ny - 1),
                        bond_type="nn",
                    )
                )

        # --- Next-nearest neighbors (nnn) ---

        if x < nx and ny > 1:
            if y == 1:
                # bottom row
                if yperiodic:
                    # diagonal up-right with wrap in y
                    s1 = n - 1
                    s2 = n + 2 * ny - 1 - 1
                    lattice.append(
                        LatticeBond(
                            s1=s1,
                            s2=s2,
                            x1=float(x),
                            y1=float(y),
                            x2=float(x + 1),
                            y2=float(y + ny - 1),
                            bond_type="nnn",
                        )
                    )
                # diagonal up-right without wrap
                s1 = n - 1
                s2 = n + ny + 1 - 1
                lattice.append(
                    LatticeBond(
                        s1=s1,
                        s2=s2,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + 1),
                        y2=float(y + 1),
                        bond_type="nnn",
                    )
                )

            elif y == ny:
                # top row
                # diagonal down-right
                s1 = n - 1
                s2 = n + ny - 1 - 1
                lattice.append(
                    LatticeBond(
                        s1=s1,
                        s2=s2,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + 1),
                        y2=float(y - 1),
                        bond_type="nnn",
                    )
                )

                if yperiodic:
                    # diagonal down-right with wrap
                    s1 = n - 1
                    s2 = n + 1 - 1
                    lattice.append(
                        LatticeBond(
                            s1=s1,
                            s2=s2,
                            x1=float(x),
                            y1=float(y),
                            x2=float(x + 1),
                            y2=float(y - ny + 1),
                            bond_type="nnn",
                        )
                    )

            else:
                # middle rows: two diagonals
                # down-right
                s1 = n - 1
                s2 = n + ny - 1 - 1
                lattice.append(
                    LatticeBond(
                        s1=s1,
                        s2=s2,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + 1),
                        y2=float(y - 1),
                        bond_type="nnn",
                    )
                )

                # up-right
                s1 = n - 1
                s2 = n + ny + 1 - 1
                lattice.append(
                    LatticeBond(
                        s1=s1,
                        s2=s2,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + 1),
                        y2=float(y + 1),
                        bond_type="nnn",
                    )
                )

    return lattice


def iter_neighbors_and_types(
    lattice: Iterable[LatticeBond],
) -> Iterable[Tuple[Tuple[int, int], str]]:
    """
    Yield ((s1, s2), bond_type) for each bond in the lattice.
    """
    for bond in lattice:
        yield (bond.s1, bond.s2), bond.bond_type

lattice = nnn_square_lattice(4, 3, yperiodic=True)