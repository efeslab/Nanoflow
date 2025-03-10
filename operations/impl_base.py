from typing import List

class OperationImpl:
    category_tag = None
    def __init__(self):
        self.impl_tag = None
    
    @staticmethod
    def list_tags(self) -> List[str]:
        return [""]
    
    def config(self, impl_tag, parameter_map = {}):
        self.impl_tag = impl_tag
        
    def run(self, *args, **kwargs):
        pass