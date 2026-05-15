from math import nan


class PoissonSolver:
    def solve(self,
              vHt,
              rhot) -> float:
        raise NotImplementedError

    def dipole_layer_correction(self) -> float:
        raise NotImplementedError


class PoissonSolverWrapper(PoissonSolver):
    def __init__(self, solver):
        self.description = solver.get_description()
        self.solver = solver

    def __str__(self):
        return self.description

    def solve(self,
              vHt,
              rhot) -> float:
        self.solver.solve(vHt.data, rhot.data)
        return nan

    def dipole_layer_correction(self) -> float:
        try:
            return self.solver.correction
        except AttributeError:
            raise NotImplementedError
