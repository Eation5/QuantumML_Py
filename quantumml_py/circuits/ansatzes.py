from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, UCCSD
from qiskit.opflow import PauliSumOp
from typing import List, Optional, Union

class VariationalAnsatz:
    """
    A class for building various parameterized quantum circuits (ansatzes) for variational algorithms.
    It supports common ansatz types like EfficientSU2, RealAmplitudes, and UCCSD, as well as custom designs.
    """
    def __init__(self, num_qubits: int, reps: int = 1):
        """
        Initialize the VariationalAnsatz with the number of qubits and repetitions.
        
        Args:
            num_qubits (int): The number of qubits in the circuit.
            reps (int): The number of repetitions of the variational block.
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        if reps <= 0:
            raise ValueError("Number of repetitions must be a positive integer.")

        self.num_qubits = num_qubits
        self.reps = reps
        self._circuit: Optional[QuantumCircuit] = None
        self._parameters: Optional[ParameterVector] = None

    @property
    def circuit(self) -> QuantumCircuit:
        if self._circuit is None:
            raise RuntimeError("Ansatz circuit has not been built yet. Call a build method first.")
        return self._circuit

    @property
    def parameters(self) -> ParameterVector:
        if self._parameters is None:
            raise RuntimeError("Ansatz parameters have not been initialized. Call a build method first.")
        return self._parameters

    def build_efficient_su2(self, su2_gates: Optional[List[str]] = None, entanglement: str = 'linear') -> QuantumCircuit:
        """
        Builds an EfficientSU2 ansatz with parameterized single-qubit gates and entangling gates.
        
        Args:
            su2_gates (list, optional): A list of single-qubit gates to use (e.g., ["ry", "rz"]). Defaults to ["ry"].
            entanglement (str): The entanglement pattern to use ("linear", "full", "circular").
            
        Returns:
            QuantumCircuit: The built EfficientSU2 circuit.
        """
        if su2_gates is None:
            su2_gates = ["ry"]
        
        self._circuit = EfficientSU2(num_qubits=self.num_qubits, reps=self.reps,
                                     su2_gates=su2_gates, entanglement=entanglement)
        self._parameters = self._circuit.parameters
        return self._circuit

    def build_real_amplitudes(self, entanglement: str = 'linear') -> QuantumCircuit:
        """
        Builds a RealAmplitudes ansatz, a hardware-efficient circuit with Y-rotations and CNOTs.

        Args:
            entanglement (str): The entanglement pattern to use ("linear", "full", "circular").

        Returns:
            QuantumCircuit: The built RealAmplitudes circuit.
        """
        self._circuit = RealAmplitudes(num_qubits=self.num_qubits, reps=self.reps, entanglement=entanglement)
        self._parameters = self._circuit.parameters
        return self._circuit

    def build_uccsd_ansatz(self, 
                           num_particles: int, 
                           num_spin_orbitals: int, 
                           qubit_mapping: str = 'parity', 
                           two_qubit_reduction: bool = True) -> QuantumCircuit:
        """
        Builds a Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.
        This ansatz is commonly used for quantum chemistry problems.

        Args:
            num_particles (int): The number of particles (electrons) in the system.
            num_spin_orbitals (int): The number of spin orbitals.
            qubit_mapping (str): The mapping from fermionic to qubit operators (
                                 'jordan_wigner', 'parity', 'bravyi_kitaev').
            two_qubit_reduction (bool): Whether to apply two-qubit reduction.

        Returns:
            QuantumCircuit: The built UCCSD circuit.
        """
        if not (num_particles > 0 and num_spin_orbitals > num_particles):
            raise ValueError("Invalid num_particles or num_spin_orbitals for UCCSD.")

        self._circuit = UCCSD(num_particles=num_particles,
                              num_spin_orbitals=num_spin_orbitals,
                              qubit_mapping=qubit_mapping,
                              two_qubit_reduction=two_qubit_reduction)
        self._parameters = self._circuit.parameters
        return self._circuit

    def build_custom_ansatz(self, single_qubit_gates: List[str] = ['ry'], entangling_gate: str = 'cx') -> QuantumCircuit:
        """
        Builds a custom ansatz with specified single-qubit gates and entangling gates.
        
        Args:
            single_qubit_gates (List[str]): A list of single-qubit gates to apply (e.g., ['rx', 'ry', 'rz']).
            entangling_gate (str): The entangling gate to use (e.g., 'cx', 'cz').
            
        Returns:
            QuantumCircuit: The custom-built quantum circuit.
        """
        self._circuit = QuantumCircuit(self.num_qubits)
        num_params_per_layer = self.num_qubits * len(single_qubit_gates)
        total_params = num_params_per_layer * (self.reps + 1) # +1 for initial layer
        
        self._parameters = ParameterVector('theta', length=total_params)
        param_idx = 0

        for rep in range(self.reps + 1):
            # Single-qubit gates layer
            for i in range(self.num_qubits):
                for gate_name in single_qubit_gates:
                    if param_idx < total_params:
                        param = self._parameters[param_idx]
                        if gate_name == 'rx':
                            self._circuit.rx(param, i)
                        elif gate_name == 'ry':
                            self._circuit.ry(param, i)
                        elif gate_name == 'rz':
                            self._circuit.rz(param, i)
                        else:
                            raise ValueError(f"Unsupported single-qubit gate: {gate_name}")
                        param_idx += 1
            
            # Entanglement layer (if not the last repetition)
            if rep < self.reps:
                for i in range(self.num_qubits - 1):
                    if entangling_gate == 'cx':
                        self._circuit.cx(i, i + 1)
                    elif entangling_gate == 'cz':
                        self._circuit.cz(i, i + 1)
                    else:
                        raise ValueError(f"Unsupported entangling gate: {entangling_gate}")
        
        return self._circuit

    def get_circuit(self) -> QuantumCircuit:
        """
        Get the built quantum circuit.
        
        Returns:
            QuantumCircuit: The built quantum circuit.
        """
        if self._circuit is None:
            raise RuntimeError("Ansatz circuit has not been built yet. Call a build method first.")
        return self._circuit

    def get_parameters(self) -> ParameterVector:
        """
        Get the parameters of the built quantum circuit.
        
        Returns:
            ParameterVector: The parameters of the quantum circuit.
        """
        if self._parameters is None:
            raise RuntimeError("Ansatz parameters have not been initialized. Call a build method first.")
        return self._parameters

# Example Usage (for demonstration, not part of the class)
# if __name__ == '__main__':
#     # Example 1: EfficientSU2
#     print("\n--- EfficientSU2 Ansatz ---")
#     ansatz_builder_es2 = VariationalAnsatz(num_qubits=2, reps=2)
#     es2_circuit = ansatz_builder_es2.build_efficient_su2(entanglement='full')
#     print(es2_circuit.draw())
#     print(f"Number of parameters: {len(ansatz_builder_es2.parameters)}")

#     # Example 2: RealAmplitudes
#     print("\n--- RealAmplitudes Ansatz ---")
#     ansatz_builder_ra = VariationalAnsatz(num_qubits=3, reps=1)
#     ra_circuit = ansatz_builder_ra.build_real_amplitudes(entanglement='circular')
#     print(ra_circuit.draw())
#     print(f"Number of parameters: {len(ansatz_builder_ra.parameters)}")

#     # Example 3: Custom Ansatz
#     print("\n--- Custom Ansatz ---")
#     ansatz_builder_custom = VariationalAnsatz(num_qubits=2, reps=1)
#     custom_circuit = ansatz_builder_custom.build_custom_ansatz(single_qubit_gates=['rx', 'rz'], entangling_gate='cz')
#     print(custom_circuit.draw())
#     print(f"Number of parameters: {len(ansatz_builder_custom.parameters)}")

#     # Example 4: UCCSD Ansatz (requires a molecular problem context)
#     # For a simple demonstration, we'll just build the circuit structure.
#     print("\n--- UCCSD Ansatz (Structure Only) ---")
#     ansatz_builder_uccsd = VariationalAnsatz(num_qubits=4) # num_qubits should match num_spin_orbitals
#     try:
#         uccsd_circuit = ansatz_builder_uccsd.build_uccsd_ansatz(num_particles=2, num_spin_orbitals=4)
#         print(uccsd_circuit.draw())
#         print(f"Number of parameters: {len(ansatz_builder_uccsd.parameters)}")
#     except Exception as e:
#         print(f"Could not build UCCSD ansatz (this is expected if Qiskit Nature is not fully configured): {e}")
