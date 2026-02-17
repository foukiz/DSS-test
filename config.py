import os.path as path
import yaml
import sys

from torch.optim import Optimizer

from functools import reduce
from typing import Any
from types import ModuleType


class Config:

    def __init__(self, conf_dic=None, conf_file=None, instantiate=True):
        if conf_dic is None and conf_file is None: return 
        if conf_dic is not None:
            self.config = conf_dic
        elif conf_file is not None:
            self.config = self.make_config_out_of_file(conf_file)
        for k, dic in self.config.items():
            if k.lower() == 'dataset': self.dataset = self.format_dic(dic)
            if k.lower() == 'model': self.model = self.format_dic(dic)
            if k.lower() == 'train': self.train = self.format_dic(dic)
            if k.lower() == 'project': self.project = dic
        if instantiate:
            self.instantiate_activation()
            self.instantiate_loss()
            self.instantiate_metrics()

    def make_config_out_of_file(self, conf_file):
        ext = path.splitext(conf_file)[-1]
        extensions = ['.yaml']
        if ext not in extensions:
            err_str = "{} is not a correct file extension, accepted files are".format(conf_file)
            for i, k in enumerate(extensions):
                if i == len(extensions) - 1 and i > 0: err_str += " and {}".format(k)
                else: err_str += " {},".format(k)
            raise KeyError(err_str)
        
        if ext == '.yaml': return self.make_config_out_of_yaml(conf_file)

    def make_config_out_of_yaml(self, conf_file):
        with open(conf_file, 'r') as f:
            dic = yaml.safe_load(f)
        return dic

    def format_dic(self, dic):
        d = {k.lower(): v for k, v in dic.items()}
        for k, v in d.items():
            if isinstance(v, str) and v.startswith('$'):
                d[k] = self.get_object_from_modules(v)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, str) and item.startswith('$'):
                        v[i] = self.get_object_from_modules(item)
            if isinstance(v, dict):
                for kk, vv, in v.items():
                    if isinstance(vv, str) and vv.startswith('$'):
                        v[kk] = self.get_object_from_modules(vv)
        return d
        
    def instantiate_optimizer(self, params):
        if not self.train['optimizer_config']: self.train['optimizer'] = self.train['optimizer'](params=params)
        else: self.train['optimizer'] = self.train['optimizer'](params=params, **self.train['optimizer_config'])
        
    def instantiate_scheduler(self):
        optimizer = self.train['optimizer']
        if not isinstance(optimizer, Optimizer):
            raise AttributeError("Error when trying to instantiate the torch scheduler ; "
                                 "optimizer has not been initialized yet")
        if not self.train['scheduler_config']: self.train['scheduler'] = self.train['scheduler'](optimizer=optimizer)
        else: self.train['scheduler'] = self.train['scheduler'](optimizer=optimizer, **self.train['scheduler_config'])

    def instantiate_loss(self):
        if not self.train['loss_config']: self.train['loss_fn'] = self.train['loss_fn']()
        else: self.train['loss_fn'] = self.train['loss_fn'](**self.train['loss_config'])

    def instantiate_activation(self):
        if 'activation' in self.model.keys() and self.model['activation'] is not None:
            if ('activation_config' in self.model.keys()) and (self.model['activation_config']):
                self.model['activation'] = self.model['activation'](**self.model['activation_config'])
            else:
                self.model['activation'] = self.model['activation']()
        if 'activation_final' in self.model.keys() and self.model['activation_final'] is not None:
            if ('activation_final_config' in self.model.keys()) and (self.model['activation_final_config']):
                self.model['activation_final'] = self.model['activation_final'](**self.model['activation_final_config'])
            else:
                self.model['activation_final'] = self.model['activation_final']()

    def instantiate_metrics(self):
        if self.train['metrics'] is not None:
            for k, v in self.train['metrics'].items():
                if self.train['metrics_config'] is not None:
                    self.train['metrics'][k] = v(**self.train['metrics_config'])
                else: self.train['metrics'][k] = v()

    def get_device(self):
        try:
            return self.train['torch_device']
        except AttributeError:
            raise
        except KeyError:
            return 'cpu'
        
    def get_object_from_modules(
        self, function_name: Any, sys_modules: ModuleType = sys.modules["__main__"]
    ) -> Any:
        """
        Given a string and a module, recovers the function
        from the modules if it's defined in the modules and the string
        starts with $.
        $ is the signifier for function name, strings not
        starting with $ are interpreted as normal strings not function
        names.
        function names with . are recursively iterated and imported. e.g.
        tensorflow.keras.losses.SparseCategoricalCrossentropy will:
        - get tensorflow from sys_modules
        - get keras from tensorflow
        - get losses from keras
        - get SparseCategoricalCrossentropy from losses

        Args:
            function_name (Any): name of function to be found in sys_modules. Strings that don't start with $ or non-strings are ignored
            and returned as is
            sys_modules: Module in which to search function name. Defaults to entry point

        Returns:
            Any: if function name is a string starting with $: the module of sys_modules named function_name
                Else: function_name
        """
        if isinstance(function_name, str) and function_name.startswith("$"):
            get_class = lambda name: reduce(
                getattr, name.split(".")[1:], __import__(name.partition(".")[0])
            )

            if "." in function_name:
                return get_class(function_name[1:])
            else:
                return getattr(sys_modules, function_name[1:])
        return function_name
    