from collections import OrderedDict


class Module:
    def __init__(self) -> None:
        super().__setattr__('_parameters',OrderedDict())

    def register_parameters(self,name, param):
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        
        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param
    

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
    
        raise AttributeError("Attribute doesn't exsits")
        

    def  __setattr__(self, name, value):
        if hasattr(value,'is_parameter') and value.is_parameter:
            self.register_parameters(name, value)
        
        elif isinstance(value, Module):
            self.register_modules(name,value)
        else:
            super().__setattr__(name, value)

    def forward(*args):
        pass

    def __call__(self,*args):
        return self.forward(*args)

    def parameters(self):
        for value in  self._parameters.values():
            yield value

    def __str__(self) -> str:
        return self.__class__.__qualname__
    def __repr__(self) -> str:
        return self.__class__.__qualname__

class Network:
    def __init__(self) -> None:
        super().__setattr__('_modules', OrderedDict())
    
    def register_modules(self,name, module):
        if '_modules' not in self.__dict__:
            raise AttributeError(
                "cannot assign module before Module.__init__() call")
        
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))

        if module is None:
            self._modules[name] = None
        else:
            self._modules[name] = module
    
    def  __setattr__(self, name, value):
     
        if isinstance(value, Module):
            self.register_modules(name,value)
        else:
            super().__setattr__(name, value)

    def forward(self,*args):
        pass

    def parameters(self):
        for module in  self._modules.values():
            for param in module.parameters():
                yield param
        
    def __call__(self, *args) :
        return self.forward(*args)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("Attribute doesn't exsits")       
    
    def __str__(self) -> str:
        return '\n'.join('{} : {}'.format(name, val) for name, val in self._modules.items())