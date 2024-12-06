import qutip as qt
from qhash import Hashing


def test_vs_qutip():
    n_modes = 2
    n_exc = 6
    hash_inst = Hashing(n_exc, n_modes)
    a_0_qt = qt.tensor(qt.destroy(n_exc + 1), qt.qeye(n_exc + 1))
    a_1_qt = qt.tensor(qt.qeye(n_exc + 1), qt.destroy(n_exc + 1))
    a_0_hash = hash_inst.a_operator(0)
    a_1_hash = hash_inst.a_operator(1)
    states_list = hash_inst.basis_vectors()
    for a_qt, a_hash in zip([a_0_qt, a_1_qt], [a_0_hash, a_1_hash]):
        for idx_1, state_1 in enumerate(states_list):
            for idx_2, state_2 in enumerate(states_list):
                state_1_qt = qt.basis([n_exc + 1, n_exc + 1], list(state_1.astype(int)))
                state_2_qt = qt.basis([n_exc + 1, n_exc + 1], list(state_2.astype(int)))
                hash_result = a_hash[idx_1, idx_2]
                qt_result = state_1_qt.dag() * a_qt * state_2_qt
                assert hash_result == qt_result
