# QuantumML_Py

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Qiskit](https://img.shields.io/badge/Qiskit-0.40%2B-purple?style=flat-square&logo=qiskit)
![Pennylane](https://img.shields.io/badge/PennyLane-0.30%2B-darkgreen?style=flat-square&logo=pennylane)
![License](https://img.shields.io/github/license/Eation5/QuantumML_Py?style=flat-square)

## Overview

QuantumML_Py is a Python library dedicated to exploring and implementing Quantum Machine Learning (QML) algorithms. It provides a high-level interface for building quantum circuits, training variational quantum algorithms (VQAs), and integrating with classical machine learning workflows. Leveraging frameworks like Qiskit and PennyLane, this project aims to make quantum computing accessible for ML practitioners and researchers.

## Features

- **Variational Quantum Eigensolver (VQE)**: Implementations for finding ground states of Hamiltonians.
- **Quantum Approximate Optimization Algorithm (QAOA)**: Tools for solving combinatorial optimization problems.
- **Quantum Neural Networks (QNNs)**: Build and train quantum circuits as neural network layers.
- **Integration with Classical ML**: Combine quantum and classical models for hybrid algorithms.
- **Visualization Tools**: Plotting utilities for quantum states, circuits, and optimization landscapes.
- **Simulator & Hardware Agnostic**: Run algorithms on local simulators or connect to quantum hardware.

## Installation

To get started with QuantumML_Py, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Eation5/QuantumML_Py.git
cd QuantumML_Py
pip install -r requirements.txt
```

## Usage

Here's a quick example of how to use QuantumML_Py to implement a simple VQE:

```python
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.opflow import Z, I, PauliSumOp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import QuantumInstance

# 1. Define the Hamiltonian (e.g., for H2 molecule in STO-3G basis)
# H = -1.05 * (I ^ I) + 0.39 * (I ^ Z) + 0.39 * (Z ^ I) + 0.01 * (Z ^ Z) + 0.18 * (X ^ X)
H = PauliSumOp.from_list([
    ("II", -1.05),
    ("IZ", 0.39),
    ("ZI", 0.39),
    ("ZZ", 0.01),
    ("XX", 0.18)
])

# 2. Define the Ansatz (a parameterized quantum circuit)
def custom_ansatz(num_qubits, reps, rotation_blocks, entanglement_blocks):
    circuit = QuantumCircuit(num_qubits)
    for _ in range(reps):
        for i in range(num_qubits):
            circuit.append(rotation_blocks[i], [i])
        for i in range(num_qubits - 1):
            circuit.append(entanglement_blocks[i], [i, i+1])
    return circuit

# Example: Simple ansatz with Ry rotations and CNOT entanglements
num_qubits = 2
reps = 1
rotation_blocks = ["Ry(theta[0])", "Ry(theta[1])"]
entanglement_blocks = ["cx"]

# For simplicity, let's use a pre-defined ansatz from Qiskit for this example
from qiskit.circuit.library import EfficientSU2
ansatz = EfficientSU2(num_qubits, reps=1, entanglement=\'linear\', su2_gates=["ry"])

# 3. Define the optimizer
optimizer = SLSQP(maxiter=100)

# 4. Set up the quantum instance (simulator)
quantum_instance = QuantumInstance(Aer.get_backend(\'statevector_simulator\'))

# 5. Run VQE
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
result = vqe.compute_minimum_eigenvalue(operator=H)

print(f"VQE Result: {result.eigenvalue.real:.4f}")
print(f"Optimal parameters: {result.optimal_parameters}")

# Compare with exact classical solution
exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(operator=H)
print(f"Exact Result: {exact_result.eigenvalue.real:.4f}")
```

## Project Structure

```
QuantumML_Py/
├── README.md
├── requirements.txt
├── setup.py
├── quantumml_py/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── vqe.py
│   │   └── qaoa.py
│   ├── circuits/
│   │   └── ansatzes.py
│   ├── utils/
│   │   └── visualization.py
│   └── data/
│       └── datasets.py
└── tests/
    ├── __init__.py
    └── test_vqe.py
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to get started.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries, please open an issue on GitHub or contact Matthew Wilson at [matthew.wilson.ai@example.com](mailto:matthew.wilson.ai@example.com).
