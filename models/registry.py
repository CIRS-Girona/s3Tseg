# ----------------------------------------------------------------------
# Adapted from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py
# ----------------------------------------------------------------------

import inspect

class Registry:
    """A registry to map strings to functions.
    Registered functions will be added to `self._model_entrypoints`,
    whose key is the assigned name and value is the function itself.
    Example:
            >>> @encoder_entrypoints.register('dummy_encoder')
            >>> def build_encoder(config):
            >>>     return DummyEncoder(
            >>>         ···
            >>>     )
            >>> encoder = encoder_entrypoints.get('dummy_encoder')(config)
    """

    def __init__(self, name):
        self._name = name
        self._model_entrypoints = dict()

    def __len__(self):
        return len(self._model_entrypoints)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._model_entrypoints})'
        return format_str
    
    @property
    def name(self):
        return self._name

    @property
    def model_entrypoints(self):
        return self._model_entrypoints
    
    def get(self, key):
        return self._model_entrypoints.get(key)
    
    def register(self, name, force=False):
        if not isinstance(name, str):
            raise TypeError(f'name must be an instance of str, but got {type(name)}')
        def wrapper(fn):
            if not inspect.isfunction(fn):
                raise TypeError(f'expected to register a function, but got {type(fn)}')
            if not force and name in self._model_entrypoints:
                raise KeyError(f'{name} is already registered in {self.name}')
            self._model_entrypoints[name] = fn
            return fn
        return wrapper


encoder_entrypoints = Registry('encoder_entrypoints')
decoder_entrypoints = Registry('decoder_entrypoints')
