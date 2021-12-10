from sicore import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import copy


class Biclustering:
    '''
        constractor
        data in R^{n*d}
        selection_num in R^{2}
        sigma in R^{n*n} default is I_n
        xi in R^{d*d} default is I_d
    '''
    def __init__(self, data, selection_num, sigma=None, xi=None):
        self.x = data
        self.vec_x =  np.ravel(data)
        self.k = selection_num
        self.n = data.shape[0]
        self.d = data.shape[1]
        if self.n < self.k[0] or self.d < self.k[1]:
            raise ValueError(
                "selection_num are high."
            )
        self.itr = 0
        if sigma is None:
            self.Sigma = np.identity(self.n)
        else:
            self.Sigma = sigma
        if xi is None:
            self.Xi = np.identity(self.d)
        else:
            self.Xi = xi
        self.obs_ans, self.obs_intervals = self._algorithm()


    def _getsubmatrix(self, data, i, j):
        res = data[np.ix_(list(i), list(j))]
        return res


    def _gete(self, size, i):
        # if not isinstance(i, np.int64) and len(i) > size:
        #     raise ValueError(
        #         "len(i) > size"
        #     )
        if isinstance(i, np.int64) or isinstance(i, int):
            i = [i]
        # print(i)
        res = np.zeros(size)
        np.put(res, list(i), 1)
        return res


    def _algorithm(self, input=None):
        # 探索ポリトープ数確認用
        self.itr += 1
        # polytope
        # b vec_x <= 0, b in bs
        bs = []
        p_i = []
        p_j = []
        Is = []
        Js = []

        if input is not None:
            vec_data = input
            data = input.reshape([self.n, self.d])
        else:
            data = self.x
            vec_data = np.ravel(data)

        Is.append([])
        # J^0
        j_sum = np.sum(data, axis=0)
        j_sum_arg = np.argsort(j_sum)[::-1]
        J = set(j_sum_arg[:self.k[1]])
        Js.append(J)

        not_J = set([elm for elm in j_sum_arg if elm not in J])
        tmp = np.ones(self.n)
        for j in J:
            j_kron = np.kron(
                        tmp, self._gete(self.d, j)
                    )
            for not_j in not_J:
                bs.append(
                    csr_matrix(
                        np.kron(
                            tmp, self._gete(self.d, not_j)
                        ) -
                        j_kron 
                    )
                )

        L = 0
        while True:
            i_sum = np.sum(self._getsubmatrix(data, range(self.n), J), axis=1)
            i_sum = np.argsort(i_sum)[::-1]
            I = set(i_sum[:self.k[0]])
            Is.append(I)

            j_sum = np.sum(self._getsubmatrix(data, I, range(self.d)), axis=0)
            j_sum = np.argsort(j_sum)[::-1]
            J = set(j_sum[:self.k[1]])
            Js.append(J)

            if(set(p_i) == set(I) and set(p_j) == set(J)):
                break
            else:
                p_i = I
                p_j = J
                L += 1

        # SI
        # Homotopy法で呼び出す場合はetaの更新をせず，観測時のものを使う
        if input is None:
            self.e1 = np.kron(
                self._gete(self.n, I),
                self._gete(self.d, J)
            ) / (len(I) * len(J))
            self.e2 = (
                (
                    np.ones(self.n * self.d) - 
                    np.kron(
                        self._gete(self.n, I),
                        self._gete(self.d, J)
                    )
                ) / 
                (
                    self.n * self.d - len(I) * len(J)
                )
            ) 
            self.eta = (self.e1 - self.e2)
        else:
            pass

        si = SelectiveInferenceNormSE(vec_data, np.kron(self.Sigma, self.Xi), self.eta)

        ittr = 0

        for b in bs:
            si.add_selection_event(b=b.getrow(0).toarray()[0])
            ittr += 1
        
        # set b
        for l in range(len(Js)-1):
            tmp = set(np.arange(self.n)) - set(Is[l+1])
            e_J = self._gete(self.d, Js[l])
            for i in Is[l+1]:
                e_i = self._gete(self.n, i)
                kron_e_i_e_J = np.kron(e_i , e_J)
                for not_i in tmp:
                    b = (np.kron(self._gete(self.n, not_i), e_J) - kron_e_i_e_J)
                    si.add_selection_event(b=b)

        for l in range(len(Js)-1):
            tmp = set(np.arange(self.d)) - set(Js[l+1])
            e_I = self._gete(self.n, Is[l+1])
            for j in Js[l+1]:
                e_j = self._gete(self.d, j)
                kron_e_I_e_j = np.kron(e_I, e_j)
                for not_j in tmp:
                    b = (np.kron(e_I, self._gete(self.d, not_j)) - kron_e_I_e_j)
                    si.add_selection_event(b=b)

        self.stat = copy.copy(si.stat)

        return [I, J], si.get_intervals()


    def test(self, is_oc=False):
        I = self.obs_ans[0]
        J = self.obs_ans[1]

        if is_oc:
            si = SelectiveInferenceNormSE(self.vec_x, np.kron(self.Sigma, self.Xi), self.eta)
            si.add_interval(self.obs_intervals)
            try:
                return self.obs_intervals, si.test(tail="right")
            except ZeroDivisionError:
                print(self.stat, self.obs_intervals, si.norm_intervals)
                return self.obs_intervals, si.test(tail="right", dps=1e5)
        else:
            si2 = SelectiveInferenceNorm(self.vec_x, np.kron(self.Sigma, self.Xi), self.eta)
            si2.add_interval(self.obs_intervals)
            si2.parametric_search2(algorithm=self._algorithm, min_tail=-100, max_tail=100, tol=1e-5)
            try:
                p = si2.test(
                    model_selector=lambda res : set(res[0]) == set(I) and set(res[1]) == set(J),
                    tail="right"
                )
                self.stat = si2.stat
                return si2.norm_intervals, p
            except ZeroDivisionError:
                print(self.stat, self.obs_intervals, si2.norm_intervals)
                p = si2.test(
                    model_selector=lambda res : set(res[0]) == set(I) and set(res[1]) == set(J),
                    tail="right",
                    dps=1e5
                )
                self.stat = si2.stat
                return si2.norm_intervals, p
            except ValueError as e:
                self.stat = si2.stat
                if si2.stat < si2.norm_intervals[0][0]:
                    return si2.norm_intervals, 1
                if si2.stat > si2.norm_intervals[-1][1]:
                    return si2.norm_intervals, 0
                else:
                    raise ValueError(
                        str(e)
                    )

    
    def naive_test(self):
        naive = NaiveInferenceNorm(self.vec_x, np.kron(self.Sigma, self.Xi), self.eta)
        return naive.test(tail="double")

    
    def path_data(self, p) -> np.ndarray:
        '''
        実数pを与えた時のデータz+cpを返す
        '''
        si = SelectiveInferenceNorm(self.vec_x, np.kron(self.Sigma, self.Xi), self.eta)
        c = si.c
        z = si.z
        return np.array(z + c * p)