"""polcheck: regression testing for robots that run learned policies."""

from polcheck.measures import RunData, measure
from polcheck.recorder import Contact, Recorder
from polcheck.schema import SimulatorInfo

__version__ = "0.0.1"

__all__ = ["Contact", "Recorder", "RunData", "SimulatorInfo", "measure"]
