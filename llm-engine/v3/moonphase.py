
from dataclasses import dataclass

@dataclass
class MoonPhase:
    '''Data object to be passed out from the api to the caller'''
    phase: float
    year: int
    month: int
    day: int