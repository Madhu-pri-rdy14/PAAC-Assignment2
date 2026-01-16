import torch
import numpy as np
from torch.utils.data import Dataset

def random_density_matrix(n_qubits):
    """Generate a random valid density matrix."""
    dim = 2 ** n_qubits
    x = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = x @ x.conj().T
    rho /= np.trace(rho)
    return rho.astype(np.complex64)

def simulate_measurement(rho, n_shots=100):
    """Simulate random Pauli measurements."""
    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))

    I = np.eye(2, dtype=np.complex64)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

    paulis = [X, Y, Z]
    data_sequence = []

    if n_qubits == 1:
        for _ in range(n_shots):
            basis_idx = np.random.randint(0, 3)
            M = paulis[basis_idx]

            projector_plus = (I + M) / 2
            prob_plus = np.real(np.trace(projector_plus @ rho))

            outcome = 1 if np.random.rand() < prob_plus else -1
            data_sequence.append([basis_idx, outcome])

    return np.array(data_sequence, dtype=np.float32)

class QuantumDataset(Dataset):
    def __init__(self, num_samples=1000, n_qubits=1, n_shots=50):
        self.data = []
        self.targets = []

        print(f"Generating {num_samples} random quantum states...")
        for _ in range(num_samples):
            rho = random_density_matrix(n_qubits)
            measurements = simulate_measurement(rho, n_shots)
            self.data.append(measurements)
            self.targets.append(rho)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.targets[idx])
        return x, y
