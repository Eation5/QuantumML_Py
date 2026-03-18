from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class VariationalAnsatz:
    """
    A class for building parameterized quantum circuits (ansatzes) for variational algorithms.
    """
    def __init__(self, num_qubits, reps=1):
        """
        Initialize the VariationalAnsatz with the number of qubits and repetitions.
        
        Args:
            num_qubits (int): The number of qubits in the circuit.
            reps (int): The number of repetitions of the variational block.
        """
        self.num_qubits = num_qubits
        self.reps = reps
        self.parameters = ParameterVector('theta', length=0)
        self.circuit = QuantumCircuit(num_qubits)

    def build_efficient_su2(self, su2_gates=['ry'], entanglement='linear'):
        """
        Build an EfficientSU2 ansatz with parameterized single-qubit gates and entangling gates.
        
        Args:
            su2_gates (list): A list of single-qubit gates to use in the circuit.
            entanglement (str): The entanglement pattern to use ('linear', 'full', 'circular').
            
        Returns:
            QuantumCircuit: The built EfficientSU2 circuit.
        """
        from qiskit.circuit.library import EfficientSU2
        self.circuit = EfficientSU2(num_qubits=self.num_qubits, reps=self.reps, 
                                     su2_gates=su2_gates, entanglement=entanglement)
        self.parameters = self.circuit.parameters
        return self.circuit

    def build_custom_ansatz(self):
        """
        Build a custom ansatz with parameterized Ry rotations and linear CNOT entanglement.
        
        Returns:
            QuantumCircuit: The custom-built quantum circuit.
        """
        num_params = self.num_qubits * (self.reps + 1)
        self.parameters = ParameterVector('theta', length=num_params)
        
        param_idx = 0
        # Initial rotations
        for i in range(self.num_qubits):
            self.circuit.ry(self.parameters[param_idx], i)
            param_idx += 1
            
        # Repetitions of entanglement and rotation
        for _ in range(self.reps):
            # Entanglement layer
            for i in range(self.num_qubits - 1):
                self.circuit.cx(i, i + 1)
                
            # Rotation layer
            for i in range(self.num_qubits):
                self.circuit.ry(self.parameters[param_idx], i)
                param_idx += 1
                
        return self.circuit

    def get_circuit(self):
        """
        Get the built quantum circuit.
        
        Returns:
            QuantumCircuit: The built quantum circuit.
        """
        return self.circuit
