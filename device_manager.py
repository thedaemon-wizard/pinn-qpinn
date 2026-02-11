"""Quantum Device Manager for CPU/GPU/QPU backend management"""
import os
import json
import logging
import warnings
import numpy as np
import pennylane as qml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

_logger = logging.getLogger('benchmark.device_manager')

# Try importing Braket plugin
try:
    import boto3
    from braket.aws import AwsDevice, AwsSession
    import braket.pennylane_plugin
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    warnings.warn("Amazon Braket PennyLane plugin not installed.")

@dataclass
class QPUConfig:
    """Configuration for a specific QPU device"""
    arn: str
    shots: Optional[int]
    poll_timeout_seconds: int = 432000
    poll_interval_seconds: int = 1
    noise_params: Optional[Dict[str, float]] = None
    region: Optional[str] = None

#================================================
# QuantumDeviceManager
#================================================

class QuantumDeviceManager:
    """
    Manages quantum device creation and configuration for both simulators and QPUs
    Handles Amazon Braket QPU connections with proper error handling
    """

    def __init__(self, config_file: str = "braket_config.json"):
        """
        Initialize the Quantum Device Manager

        Args:
            config_file: Path to JSON configuration file
        """
        self.config = self._load_config(config_file)
        self.aws_session = None
        self.device_cache = {}
        self.current_device = None
        self.current_device_type = "simulator"

        # Initialize AWS session if credentials are provided
        if BRAKET_AVAILABLE:
            self._initialize_aws_session()

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Return default configuration for simulator mode"""
        return {
            "simulator_devices": {
                "local": {
                    "backend": "default.mixed",
                    "shots": None,
                    "noise_params": {
                        "depolarizing_1q": 0.001,
                        "depolarizing_2q": 0.01,
                        "amplitude_damping": 0.0001,
                        "phase_damping": 0.0001,
                        "readout_error": 0.01
                    }
                }
            },
            "execution_settings": {
                "max_parallel": 4,
                "use_error_mitigation": True,
                "zero_noise_extrapolation": {
                    "enabled": True,
                    "scale_factors": [1.0, 1.5, 2.0]
                }
            }
        }

    def _initialize_aws_session(self):
        """Initialize AWS session for Braket access"""
        try:
            aws_config = self.config.get("aws_settings", {})
            credentials = aws_config.get("credentials", {})

            if credentials.get("aws_access_key_id"):
                # Use provided credentials
                session = boto3.Session(
                    aws_access_key_id=credentials.get("aws_access_key_id"),
                    aws_secret_access_key=credentials.get("aws_secret_access_key"),
                    aws_session_token=credentials.get("aws_session_token"),
                    region_name=aws_config.get("region", "us-east-1")
                )
                self.aws_session = AwsSession(boto_session=session)
            else:
                # Use default credentials
                self.aws_session = AwsSession()

        except Exception as e:
            _logger.warning(f"Could not initialize AWS session: {e}")
            _logger.warning("Falling back to simulator mode")
            self.aws_session = None

    def create_device(self, device_name: str, n_qubits: int,
                     shots: Optional[int] = None,
                     use_qpu: bool = False) -> qml.device:
        """
        Create a quantum device (simulator or QPU)

        Args:
            device_name: Name of the device (from config)
            n_qubits: Number of qubits
            shots: Number of shots for sampling
            use_qpu: Whether to use QPU instead of simulator

        Returns:
            PennyLane device object
        """
        device_key = f"{device_name}_{n_qubits}_{shots}_{use_qpu}"

        # Check cache
        if device_key in self.device_cache:
            return self.device_cache[device_key]

        if use_qpu and BRAKET_AVAILABLE and self.aws_session:
            device = self._create_qpu_device(device_name, n_qubits, shots)
            self.current_device_type = "qpu"
        else:
            device = self._create_simulator_device(device_name, n_qubits, shots)
            self.current_device_type = "simulator"

        # Cache the device
        self.device_cache[device_key] = device
        self.current_device = device

        return device

    def _create_qpu_device(self, device_name: str, n_qubits: int,
                          shots: Optional[int] = None) -> qml.device:
        """Create a QPU device via Amazon Braket"""
        qpu_config = self.config.get("qpu_devices", {}).get(device_name)

        if not qpu_config:
            _logger.warning(f"QPU {device_name} not found in config, using simulator")
            return self._create_simulator_device("local", n_qubits, shots)

        try:
            # Get S3 settings
            aws_config = self.config.get("aws_settings", {})
            s3_destination = (
                aws_config.get("s3_bucket", "amazon-braket-results"),
                aws_config.get("s3_prefix", "qpinn-experiments")
            )

            # Check device availability
            braket_device = AwsDevice(qpu_config["arn"], aws_session=self.aws_session)
            status = braket_device.status

            if status != "ONLINE":
                _logger.warning(f"QPU {device_name} is {status}, using simulator instead")
                return self._create_simulator_device("local", n_qubits, shots)

            # Create PennyLane device
            device = qml.device(
                "braket.aws.qubit",
                device_arn=qpu_config["arn"],
                wires=n_qubits,
                shots=shots or qpu_config.get("shots", 1000),
                s3_destination_folder=s3_destination,
                poll_timeout_seconds=qpu_config.get("poll_timeout_seconds", 432000),
                poll_interval_seconds=qpu_config.get("poll_interval_seconds", 1),
                aws_session=self.aws_session,
                parallel=self.config.get("execution_settings", {}).get("max_parallel", 10) > 1,
                max_parallel=self.config.get("execution_settings", {}).get("max_parallel", 10)
            )

            _logger.info(f"Successfully connected to QPU: {device_name} ({qpu_config['arn']})")
            return device

        except Exception as e:
            _logger.error(f"Error creating QPU device: {e}")
            _logger.warning("Falling back to simulator")
            return self._create_simulator_device("local", n_qubits, shots)

    def _create_simulator_device(self, device_name: str, n_qubits: int,
                                 shots: Optional[int] = None) -> qml.device:
        """Create a simulator device"""
        sim_config = self.config.get("simulator_devices", {}).get(device_name, {})

        if device_name == "local" or not BRAKET_AVAILABLE:
            # Use PennyLane's default mixed simulator
            if shots is not None:
                device = qml.device("default.mixed", wires=n_qubits, shots=shots)
            else:
                device = qml.device("default.mixed", wires=n_qubits)
        else:
            # Use Braket simulator
            try:
                device = qml.device(
                    "braket.aws.qubit",
                    device_arn=sim_config.get("arn"),
                    wires=n_qubits,
                    shots=shots or sim_config.get("shots"),
                    aws_session=self.aws_session
                )
            except:
                # Fallback to local simulator
                device = qml.device("default.mixed", wires=n_qubits, shots=shots)

        _logger.info(f"Using simulator: {device_name}")
        return device

    def get_noise_params(self, device_name: str) -> Dict[str, float]:
        """Get noise parameters for a specific device"""
        # Check QPU devices first
        qpu_config = self.config.get("qpu_devices", {}).get(device_name)
        if qpu_config and qpu_config.get("noise_params"):
            return qpu_config["noise_params"]

        # Check simulator devices
        sim_config = self.config.get("simulator_devices", {}).get(device_name)
        if sim_config and sim_config.get("noise_params"):
            return sim_config["noise_params"]

        # Return default noise parameters
        return {
            "depolarizing_1q": 0.001,
            "depolarizing_2q": 0.01,
            "amplitude_damping": 0.0001,
            "phase_damping": 0.0001,
            "readout_error": 0.01,
            "T1": 0.00001,
            "T2": 0.000005
        }

    def apply_hardware_noise(self, wire: int, noise_params: Dict[str, float],
                            gate_type: str = "1q"):
        """
        Apply realistic hardware noise to a qubit
        Uses noise parameters from device configuration

        Args:
            wire: Qubit index
            noise_params: Noise parameters dictionary
            gate_type: Type of gate ("1q" or "2q")
        """
        # Apply depolarizing noise
        if gate_type == "1q":
            error_rate = noise_params.get("depolarizing_1q", 0.001)
        else:
            error_rate = noise_params.get("depolarizing_2q", 0.01)

        if error_rate > 0:
            qml.DepolarizingChannel(error_rate, wires=wire)

        # Apply amplitude damping (T1 decay)
        t1_rate = noise_params.get("amplitude_damping", 0.0001)
        if t1_rate > 0:
            qml.AmplitudeDamping(t1_rate, wires=wire)

        # Apply phase damping (T2 decay)
        t2_rate = noise_params.get("phase_damping", 0.0001)
        if t2_rate > 0:
            qml.PhaseDamping(t2_rate, wires=wire)

    def check_device_status(self, device_name: str) -> Dict[str, Any]:
        """
        Check the status of a QPU device

        Returns:
            Dictionary with device status information
        """
        if not BRAKET_AVAILABLE or not self.aws_session:
            return {"status": "SIMULATOR_MODE", "available": True}

        qpu_config = self.config.get("qpu_devices", {}).get(device_name)
        if not qpu_config:
            return {"status": "NOT_CONFIGURED", "available": False}

        try:
            device = AwsDevice(qpu_config["arn"], aws_session=self.aws_session)

            # Get device properties
            properties = device.properties.dict()

            return {
                "status": device.status,
                "available": device.status == "ONLINE",
                "name": device.name,
                "provider": device.provider_name,
                "qubits": properties.get("paradigm", {}).get("qubitCount", 0),
                "connectivity": properties.get("paradigm", {}).get("connectivity", {}),
                "native_gates": properties.get("paradigm", {}).get("nativeGateSet", []),
                "queue_depth": device.queue_depth().quantum_tasks if hasattr(device, 'queue_depth') else "N/A"
            }
        except Exception as e:
            return {"status": "ERROR", "available": False, "error": str(e)}

    def estimate_circuit_runtime(self, n_gates: int, n_qubits: int,
                                 shots: int, device_name: str) -> float:
        """
        Estimate runtime for a quantum circuit on a specific device

        Returns:
            Estimated runtime in seconds
        """
        if self.current_device_type == "simulator":
            # Rough estimate for simulators
            return 0.001 * n_gates * shots / 1000
        else:
            # Rough estimate for QPUs (includes queue time)
            noise_params = self.get_noise_params(device_name)
            gate_time = 0.0001  # ~100 microseconds per gate
            measurement_time = 0.001  # ~1ms per measurement

            circuit_time = n_gates * gate_time + measurement_time
            total_time = circuit_time * shots

            # Add overhead for QPU scheduling
            overhead = 10.0  # 10 seconds overhead

            return total_time + overhead

    def get_error_mitigation_config(self) -> Dict[str, Any]:
        """Get error mitigation configuration"""
        return self.config.get("execution_settings", {}).get("zero_noise_extrapolation", {})

    def cleanup(self):
        """Cleanup resources"""
        self.device_cache.clear()
        if self.aws_session:
            # Close AWS session if needed
            pass
