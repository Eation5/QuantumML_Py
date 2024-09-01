import numpy as np
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import PauliSumOp
from qiskit.circuit import QuantumCircuit, Parameter
from typing import Union, List, Optional, Callable

class QuantumVQE:
    """
    A comprehensive wrapper class for the Variational Quantum Eigensolver (VQE) algorithm.
    This class provides enhanced functionalities for setting up, running, and analyzing VQE results
    on a given Hamiltonian, including support for different optimizers, initial parameters,
    and callback functions for tracking optimization progress.
    """
    def __init__(self,
                 ansatz: QuantumCircuit,
                 optimizer: Optional[Union[SLSQP, COBYLA, L_BFGS_B]] = None,
                 quantum_instance: Optional[QuantumInstance] = None,
                 initial_parameters: Optional[Union[np.ndarray, List[float]]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None):
        """
        Initialize the QuantumVQE with an ansatz, optimizer, quantum instance, and optional parameters.
        
        Args:
            ansatz (QuantumCircuit): The parameterized quantum circuit (ansatz).
            optimizer (Optimizer, optional): The classical optimizer to use. Defaults to SLSQP.
            quantum_instance (QuantumInstance, optional): The backend to run the algorithm on. 
                                                           Defaults to statevector_simulator.
            initial_parameters (np.ndarray or List[float], optional): Initial parameters for the ansatz.
                                                                      If None, random parameters are used.
            callback (Callable, optional): A callback function to be called after each optimization step.
                                           It receives (evaluation_count, parameters, mean, std).
        """
        self.ansatz = ansatz
        self.optimizer = optimizer if optimizer else SLSQP(maxiter=100)
        self.quantum_instance = quantum_instance if quantum_instance else \
            QuantumInstance(Aer.get_backend(
                'statevector_simulator', 
                shots=1024, 
                optimization_level=1
            ))
        self.initial_parameters = initial_parameters
        self.callback = callback

        # Initialize VQE algorithm
        self.vqe = VQE(ansatz=self.ansatz,
                       optimizer=self.optimizer,
                       quantum_instance=self.quantum_instance,
                       initial_point=self._get_initial_point(),
                       callback=self.callback)

    def _get_initial_point(self) -> np.ndarray:
        """
        Determines the initial parameters for the VQE optimization.
        If initial_parameters are provided, they are used. Otherwise, random parameters are generated.
        """
        if self.initial_parameters is not None:
            if len(self.initial_parameters) != self.ansatz.num_parameters:
                raise ValueError("Length of initial_parameters must match the number of ansatz parameters.")
            return np.asarray(self.initial_parameters)
        else:
            # Generate random initial parameters if not provided
            return np.random.rand(self.ansatz.num_parameters) * 2 * np.pi

    def run(self, hamiltonian: PauliSumOp):
        """
        Run the VQE algorithm on the provided Hamiltonian.
        
        Args:
            hamiltonian (PauliSumOp): The Hamiltonian of the system, represented as a PauliSumOp.
            
        Returns:
            VQEResult: The result object containing the minimum eigenvalue, optimal parameters, 
                       and other optimization details.
        """
        print(f"Running VQE on Hamiltonian with {hamiltonian.num_qubits} qubits using {type(self.optimizer).__name__} optimizer...")
        result = self.vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        print(f"VQE optimization finished. Optimal eigenvalue: {result.eigenvalue.real:.6f}")
        return result

    def get_optimal_circuit(self, result) -> QuantumCircuit:
        """
        Get the ansatz circuit with the optimal parameters found by VQE.
        
        Args:
            result (VQEResult): The result object from a VQE run.
            
        Returns:
            QuantumCircuit: The optimized quantum circuit with assigned parameters.
        """
        if result.optimal_parameters is None:
            raise ValueError("VQE result does not contain optimal parameters. Ensure VQE was run successfully.")
        return self.ansatz.assign_parameters(result.optimal_parameters)

    @staticmethod
    def analyze_result(result):
        """
        Analyzes and prints key information from the VQE result.
        """
        print("\n--- VQE Result Analysis ---")
        print(f"Optimal eigenvalue: {result.eigenvalue.real:.6f}")
        print(f"Optimal parameters: {result.optimal_parameters}")
        print(f"Number of evaluations: {result.cost_function_evals}")
        if hasattr(result, 'optimizer_evals'):
            print(f"Optimizer evaluations: {result.optimizer_evals}")
        if hasattr(result, 'optimizer_time'):
            print(f"Optimizer time: {result.optimizer_time:.2f} seconds")
        print("---------------------------")

class ExactSolver:
    """
    A classical solver for finding the exact minimum eigenvalue of a Hamiltonian.
    Used for benchmarking quantum algorithms against exact solutions.
    """
    @staticmethod
    def solve(hamiltonian: PauliSumOp) -> float:
        """
        Solve the Hamiltonian exactly using a classical eigensolver.
        
        Args:
            hamiltonian (PauliSumOp): The Hamiltonian of the system.
            
        Returns:
            float: The exact minimum eigenvalue.
        """
        from qiskit.algorithms import NumPyMinimumEigensolver
        exact_solver = NumPyMinimumEigensolver()
        result = exact_solver.compute_minimum_eigenvalue(operator=hamiltonian)
        if result.eigenvalue is None:
            raise ValueError("Exact solver failed to compute eigenvalue.")
        return result.eigenvalue.real

# Example Usage (for demonstration, not part of the class)
# if __name__ == '__main__':
#     # Define a simple Hamiltonian (e.g., for H2 molecule in STO-3G basis)
#     # This is a placeholder; real Hamiltonians are more complex.
#     from qiskit.opflow import Z, I
#     hamiltonian = -1.05 * (Z ^ I) + 0.39 * (I ^ Z) + 0.011 * (Z ^ Z) + 0.18 * (I ^ I)

#     # Define a simple ansatz (e.g., UCCSD or a custom circuit)
#     num_qubits = hamiltonian.num_qubits
#     ansatz = QuantumCircuit(num_qubits)
#     for i in range(num_qubits):
#         ansatz.ry(Parameter(f'theta_{i}'), i)
#     ansatz.cx(0, 1)

#     # Define a callback function to observe optimization progress
#     intermediate_data = {'eval_count': [], 'parameters': [], 'means': [], 'stds': []}
#     def store_intermediate_result(eval_count, parameters, mean, std):
#         intermediate_data['eval_count'].append(eval_count)
#         intermediate_data['parameters'].append(parameters)
#         intermediate_data['means'].append(mean)
#         intermediate_data['stds'].append(std)
#         print(f"Callback: Eval {eval_count}, Mean: {mean:.6f}")

#     # Initialize and run VQE
#     vqe_solver = QuantumVQE(ansatz=ansatz, callback=store_intermediate_result)
#     vqe_result = vqe_solver.run(hamiltonian)

#     # Analyze results
#     QuantumVQE.analyze_result(vqe_result)

#     # Get optimal circuit
#     optimal_circuit = vqe_solver.get_optimal_circuit(vqe_result)
#     print("\nOptimal Circuit:")
#     print(optimal_circuit.draw())

#     # Solve exactly for comparison
#     exact_energy = ExactSolver.solve(hamiltonian)
#     print(f"\nExact minimum eigenvalue: {exact_energy:.6f}")
#     print(f"Difference (VQE - Exact): {vqe_result.eigenvalue.real - exact_energy:.6f}")
