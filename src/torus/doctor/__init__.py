"""Machine capability checks for ``torus doctor`` (P0.6)."""

from torus.doctor.eligibility import ReaderStatus, assess_reader, assess_readers
from torus.doctor.extras import EXTRA_MARKERS, detect_extras
from torus.doctor.hardware import GpuInfo, HardwareInfo, probe_hardware
from torus.doctor.report import DoctorReport, format_report, run_doctor
from torus.doctor.weights import model_dir, weights_present

__all__ = [
    "EXTRA_MARKERS",
    "DoctorReport",
    "GpuInfo",
    "HardwareInfo",
    "ReaderStatus",
    "assess_reader",
    "assess_readers",
    "detect_extras",
    "format_report",
    "model_dir",
    "probe_hardware",
    "run_doctor",
    "weights_present",
]
