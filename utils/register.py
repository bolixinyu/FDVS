from typing import Any


_dict = {}

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
    
    def __call__(self, target):
        return self.registe(target)
    
    @classmethod
    def registe(self, target):
        def add_item(key, value):
            _dict[key] = value
            return value
        if callable(target):
            return add_item(target.__name__, target)
        else:
            return lambda x: add_item(target, x)
    
    def __setitem__(self, k,v) -> None:
        _dict[k] = v

    def __getitem__(self, k) -> Any:
        return _dict[k]
    
    def __contains__(self, key):
        return key in _dict

    def __str__(self):
        return str(_dict)

    def keys(self):
        return _dict.keys()

    def values(self):
        return _dict.values()

    def items(self):
        return _dict.items()
    