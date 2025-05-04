from enum import Enum


class NewsType(str, Enum):
    R_10K = "10-K report"
    R_10Q = "10-Q report"
    NI = "ni43-101 technical report"
    PEA = "preliminary economic assessment release"
    FS = "feasibility study release"
    PFS = "pre-feasibility study release"
    DRILLING = "drilling results"
    RESOURCE = "resource update"
    FINANCIAL = "financial results"
    PROJECT = "project update"
    CORPORATE = "corporate update"
    ACQUISITION = "acquisition"
    JOINT_VENTURE = "joint venture"
    OTHER = "other"