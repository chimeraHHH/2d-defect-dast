from dataclasses import dataclass

import numpy as np
from gpaw.mpi import MPIComm, broadcast


@dataclass
class NLOData:
    w_sk: np.ndarray
    f_skn: np.ndarray
    E_skn: np.ndarray
    p_skvnn: np.ndarray
    comm: MPIComm

    def write(self, filename):
        if self.comm.rank == 0:
            np.savez(filename, w_sk=self.w_sk, f_skn=self.f_skn,
                     E_skn=self.E_skn, p_skvnn=self.p_skvnn)
        self.comm.barrier()

    @classmethod
    def load(cls, filename, comm):
        """
        Load the data

        Input:
            filename        NLO data filename
        Output:
            p_kvnn          The mometum matrix elements, dimension (nk,3,nb,nb)
        """

        # Load the data to the master
        if comm.rank == 0:
            nlo = np.load(filename)
        else:
            nlo = dict.fromkeys(['w_sk', 'f_skn', 'E_skn', 'p_skvnn'])
        comm.barrier()

        return cls(nlo['w_sk'], nlo['f_skn'],
                   nlo['E_skn'], nlo['p_skvnn'], comm)

    def distribute(self):
        """
        Distribute the data among cores

        Input:
            arr_list        A list of numpy array (the first two should be s,k)
        Output:
            k_info          A  dictionary of data with key of s,k index
        """

        arr_list = [self.w_sk, self.f_skn, self.E_skn, self.p_skvnn]

        # Check the array shape
        comm = self.comm
        size = comm.size
        rank = comm.rank
        if rank == 0:
            nk = 0
            arr_shape = []
            for ii, arr in enumerate(arr_list):
                ar_shape = arr.shape
                arr_shape.append(ar_shape)
                if nk == 0:
                    ns = ar_shape[0]
                    nk = ar_shape[1]
                else:
                    assert ar_shape[1] == nk, 'Wrong shape for array.'
        else:
            arr_shape = None
            nk = None
            ns = None
        arr_shape = broadcast(arr_shape, root=0, comm=comm)
        nk = broadcast(nk, root=0, comm=comm)
        ns = broadcast(ns, root=0, comm=comm)

        # Distribute the data of k-points between cores
        k_info = {}

        # Loop over k points
        for s1 in range(ns):
            for kk in range(nk):
                if rank == 0:
                    if kk % size == rank:
                        k_info[s1 * nk + kk] = [arr[s1, kk]
                                                for arr in arr_list]
                    else:
                        for ii, arr in enumerate(arr_list):
                            data_k = np.array(arr[s1, kk], dtype=complex)
                            comm.send(
                                data_k, kk % size, tag=ii * nk + kk)
                else:
                    if kk % size == rank:
                        dataset = []
                        for ii, cshape in enumerate(arr_shape):
                            data_k = np.empty(cshape[2:], dtype=complex)
                            comm.receive(data_k, 0, tag=ii * nk + kk)
                            dataset.append(data_k)
                        k_info[s1 * nk + kk] = dataset

        return k_info
