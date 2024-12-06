# qhash
Create quantum operators using a global excitation number cutoff

In this example, we want to simulate a system with 6 modes and keeping up to 3 total excitations. In the normal way of tensoring up individual subsystems, if we kept 3+1 states in each mode, this requires a Hilbert-space size of `4^6=4096`. Here, by instead keeping only 3 total excitations in the system we require a Hilbert-space size of only 84.  
```python
from qhash import Hashing

num_excitations = 3
num_modes = 6
two_mode_hash = Hashing(num_excitations, num_modes)
hilbert_dim = two_mode_hash.hilbert_dim()  # 84
```
We can now extract the raising and lowering operators of each mode by specifying the mode index
```python
a_0 = two_mode_hash.a_operator(0)
a_1 = two_mode_hash.a_operator(1)
```
And so on. These `ndarray` objects can now be passed to your favorite tool for performing quantum simulations. The ordering of the basis vectors can be found from calling
```python
basis_vectors = two_mode_hash.basis_vectors()
```
which returns a list of the basis vectors.
