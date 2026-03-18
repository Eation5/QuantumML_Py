import numpy as np
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import QuantumInstance
from qiskit import Aer

class QuantumVQE:
    """
    A wrapper class for the Variational Quantum Eigensolver (VQE) algorithm.
    This class simplifies the process of setting up and running VQE on a given Hamiltonian.
    """
    def __init__(self, ansatz, optimizer=None, quantum_instance=None):
        """
        Initialize the QuantumVQE with an ansatz, optimizer, and quantum instance.
        
        Args:
            ansatz (QuantumCircuit): The parameterized quantum circuit (ansatz).
            optimizer (Optimizer, optional): The classical optimizer to use. Defaults to SLSQP.
            quantum_instance (QuantumInstance, optional): The backend to run the algorithm on. 
                                                           Defaults to statevector_simulator.
        """
        self.ansatz = ansatz
        self.optimizer = optimizer if optimizer else SLSQP(maxiter=100)
        self.quantum_instance = quantum_instance if quantum_instance else \
            QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.vqe = VQE(ansatz=self.ansatz, optimizer=self.optimizer, 
                       quantum_instance=self.quantum_instance)

    def run(self, hamiltonian):
        """
        Run the VQE algorithm on the provided Hamiltonian.
        
        Args:
            hamiltonian (OperatorBase): The Hamiltonian of the system.
            
        Returns:
            VQEResult: The result object containing the minimum eigenvalue and optimal parameters.
        """
        print(f"Running VQE on Hamiltonian with {hamiltonian.num_qubits} qubits...")
        result = self.vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        return result

    def get_optimal_circuit(self, result):
        """
        Get the ansatz circuit with the optimal parameters found by VQE.
        
        Args:
            result (VQEResult): The result object from a VQE run.
            
        Returns:
            QuantumCircuit: The optimized quantum circuit.
        """
        return self.ansatz.assign_parameters(result.optimal_parameters)

class ExactSolver:
    """
    A classical solver for finding the exact minimum eigenvalue of a Hamiltonian.
    Used for benchmarking quantum algorithms.
    """
    @staticmethod
    def solve(hamiltonian):
        """
        Solve the Hamiltonian exactly using classical eigensolver.
        
        Args:
            hamiltonian (OperatorBase): The Hamiltonian of the system.
            
        Returns:
            float: The exact minimum eigenvalue.
        """
        from qiskit.algorithms import NumPyMinimumEigensolver
        exact_solver = NumPyMinimumEigensolver()
        result = exact_solver.compute_minimum_eigenvalue(operator=hamiltonian)
        return result.eigenvalue.real
