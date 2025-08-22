from enum import Enum

class CategoryType(Enum):
    COMP = "COMPUTE"
    MEM = "MEMORY"
    NET = "NETWORK"

    def __str__(self) -> str: 
        return self.value