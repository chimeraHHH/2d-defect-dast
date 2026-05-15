from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.move_wfs import move_wave_functions


class PWFDIBZWaveFunctions(IBZWaveFunctions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_wave_functions = move_wave_functions

    def has_wave_functions(self):
        return self.wfs_qs[0][0].psit_nX.data is not None
