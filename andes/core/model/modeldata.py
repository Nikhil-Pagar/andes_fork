"""
Module for ModelData.
"""

import difflib
import logging
from collections import OrderedDict, defaultdict

import numpy as np
from andes.core.model.modelcache import ModelCache
from andes.core.param import (BaseParam, DataParam, IdxParam, NumParam,
                              TimerParam)
from andes.shared import pd
from andes.utils.func import validate_keys_values

logger = logging.getLogger(__name__)


class ModelData:
    r"""
    Class for holding parameter data for a model.

    This class is designed to hold the parameter data separately from model equations.
    Models should inherit this class to define the parameters from input files.

    Inherit this class to create the specific class for holding input parameters for a new model.
    The recommended name for the derived class is the model name with ``Data``. For example, data for `GENROU`
    should be named `GENROUData`.

    Parameters should be defined in the ``__init__`` function of the derived class.

    Refer to :py:mod:`andes.core.param` for available parameter types.

    Attributes
    ----------
    cache
        A cache instance for different views of the internal data.

    flags : dict
        Flags to control the routine and functions that get called. If the model is using user-defined
        numerical calls, set `f_num`, `g_num`, `j_num`, and `j_setup` properly.

    Notes
    -----
    Three default parameters are pre-defined in ``ModelData``
    and will be inherited by all models. They are

    - ``idx``, unique device idx of type :py:class:`andes.core.param.DataParam`
    - ``u``, connection status of type :py:class:`andes.core.param.NumParam`
    - ``name``, (device name of type :py:class:`andes.core.param.DataParam`

    In rare cases one does not want to define these three parameters,
    one can pass `three_params=True` to the constructor of ``ModelData``.

    Examples
    --------
    If we want to build a class ``PQData`` (for static PQ load) with three parameters, `Vn`, `p0`
    and `q0`, we can use the following ::

        from andes.core.model import ModelData, Model
        from andes.core.param import IdxParam, NumParam

        class PQData(ModelData):
            super().__init__()
            self.Vn = NumParam(default=110,
                               info="AC voltage rating",
                               unit='kV', non_zero=True,
                               tex_name=r'V_n')
            self.p0 = NumParam(default=0,
                               info='active power load in system base',
                               tex_name=r'p_0', unit='p.u.')
            self.q0 = NumParam(default=0,
                               info='reactive power load in system base',
                               tex_name=r'q_0', unit='p.u.')

    In this example, all the three parameters are defined as
    :py:class:`andes.core.param.NumParam`.
    In the full `PQData` class, other types of parameters also exist.
    For example, to store the idx of `owner`, `PQData` uses ::

        self.owner = IdxParam(model='Owner', info="owner idx")

    """

    def __init__(self, *args, three_params=True, **kwargs):
        self.params = OrderedDict()
        self.num_params = OrderedDict()
        self.idx_params = OrderedDict()
        self.timer_params = OrderedDict()
        self.n = 0
        self.uid = {}
        self._param_corrections = defaultdict(list)  # {(param_name, violation): [idx, ...]}

        # indexing bases. Most vectorized models only have one base: self.idx
        self.index_bases = []

        if not hasattr(self, 'cache'):
            self.cache = ModelCache()
        self.cache._owner = self

        if three_params is True:
            self.idx = DataParam(info='unique device idx')
            self.u = NumParam(default=1, info='connection status', unit='bool', tex_name='u')
            self.name = DataParam(info='device name')

            self.index_bases.append(self.idx)

    def __len__(self):
        return self.n

    def __setattr__(self, key, value):
        if isinstance(value, BaseParam):
            value.owner = self
            if not value.name:
                value.name = key

            if key in self.__dict__:
                logger.warning("%s: redefining <%s>. This is likely a modeling error.",
                               self.class_name, key)

            self.params[key] = value

        if isinstance(value, NumParam):
            self.num_params[key] = value
        elif isinstance(value, IdxParam):
            self.idx_params[key] = value

        # `TimerParam` is a subclass of `NumParam` and thus tested separately
        if isinstance(value, TimerParam):
            self.timer_params[key] = value

        super(ModelData, self).__setattr__(key, value)

    def add(self, **kwargs):
        """
        Add a device (an instance) to this model.

        Warnings
        --------
        This function is not intended to be used directly.
        Use the ``add`` method from System so that the index
        can be registered correctly.

        Parameters
        ----------
        kwargs
            model parameters are collected into the kwargs dictionary
        """
        idx = kwargs['idx']
        self.uid[idx] = self.n
        self.n += 1
        if "name" in self.params:
            name = kwargs.get("name")
            if (name is None) or (not isinstance(name, str) and np.isnan(name)):
                kwargs["name"] = idx

        if "idx" not in self.params:
            kwargs.pop("idx")

        for name, instance in self.params.items():
            value = kwargs.pop(name, None)
            violation = instance.add(value)
            if violation is not None:
                self._param_corrections[(name, violation)].append(idx)
        if len(kwargs) > 0:
            valid_names = list(self.params.keys())
            for bad_key in kwargs:
                close = difflib.get_close_matches(bad_key, valid_names, n=3, cutoff=0.6)
                if close:
                    logger.warning("%s: unrecognized field '%s'. Did you mean: %s?",
                                   self.class_name, bad_key, ', '.join(close))
                else:
                    logger.warning("%s: unrecognized field '%s'. Valid fields: %s",
                                   self.class_name, bad_key, ', '.join(valid_names))

    def report_corrections(self):
        """
        Emit grouped warnings for parameter corrections accumulated during ``add`` calls.

        Called once from ``System.setup`` after all devices have been added.
        Each unique (param_name, violation_type) combination produces one log line
        listing the affected device indices.
        """
        if not self._param_corrections:
            return

        for (param, violation), idxes in self._param_corrections.items():
            default = self.params[param].default
            logger.warning('%s: %d device(s) had %s param <%s> corrected to %s. idx=%s',
                           self.class_name, len(idxes), violation, param, default, idxes)

        self._param_corrections.clear()

    def as_dict(self, vin=False):
        """
        Export all parameters as a dict.

        Returns
        -------
        dict
            a dict with the keys being the `ModelData` parameter names
            and the values being an array-like of data in the order of adding.
            An additional `uid` key is added with the value default to range(n).
        """
        out = dict()

        # append 0-indexed `uid`
        out['uid'] = np.arange(self.n)

        for name, instance in self.params.items():
            # skip non-exported parameters
            if instance.export is False:
                continue

            out[name] = instance.v

            # use the original input if `vin` is True
            if (vin is True) and hasattr(instance, 'vin') and (instance.vin is not None):
                out[name] = instance.vin

            conv = instance.oconvert
            if conv is not None:
                out[name] = np.array([conv(item) for item in out[name]])

        return out

    def as_df(self, vin=False):
        """
        Export all parameters as a `pandas.DataFrame` object.
        This function utilizes `as_dict` for preparing data.

        Parameters
        ----------
        vin : bool
            If True, export all parameters from original input (``vin``).

        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        """
        if vin is False:
            out = pd.DataFrame(self.as_dict()).set_index('uid')
        else:
            out = pd.DataFrame(self.as_dict(vin=True)).set_index('uid')

        return out

    def as_df_local(self):
        """
        Export local variable values and services to a DataFrame.
        """

        out = dict()
        out['uid'] = np.arange(self.n)
        out['idx'] = self.idx.v

        for name, instance in self.cache.all_vars.items():
            out[name] = instance.v

        for name, instance in self.services.items():
            out[name] = instance.v

        return pd.DataFrame(out).set_index('uid')

    def update_from_df(self, df, vin=False):
        """
        Update parameter values from a DataFrame.

        Adding devices are not allowed.
        """
        if vin is False:
            for name, instance in self.params.items():
                if instance.export is False:
                    continue
                instance.set_all('v', df[name])
        else:
            for name, instance in self.params.items():
                if instance.export is False:
                    continue
                try:
                    instance.set_all('vin', df[name])
                    instance.v[:] = instance.vin * instance.pu_coeff
                except KeyError:
                    # fall back to `v`
                    instance.set_all('v', df[name])

        return True

    def find_param(self, prop):
        """
        Find params with the given property and return in an OrderedDict.

        Parameters
        ----------
        prop : str
            Property name

        Returns
        -------
        OrderedDict
        """
        out = OrderedDict()
        for name, instance in self.params.items():
            if instance.get_property(prop) is True:
                out[name] = instance

        return out

    def _get_busfreq_cached_matches(self, key, val):   # Extra function added by Nikhil
        """
        Retrieve cached `idx` matches for a single key/value search on the
        ``BusFreq`` model, used internally by `find_idx` as a fast path.

        Maintains a persistent cache at ``self.system._find_idx_cache``,
        mapping each searched attribute value to its corresponding list of
        device indices. The cache is rebuilt from scratch if the number of
        devices has decreased (e.g., devices removed), and extended
        incrementally (only new devices processed) if the number of devices
        has increased. This avoids rebuilding the lookup table on every call
        for a model that is queried very frequently with simple patterns.

        Parameters
        ----------
        key : str
            Name of the device attribute to search on (e.g., 'bus').
        val : Any
            The value to search for within the given attribute.

        Returns
        -------
        list
            List of matching device `idx` values. Empty list if no match found.
        """
        if not hasattr(self.system, "_find_idx_cache"):
            self.system._find_idx_cache = {}

        model_cache = self.system._find_idx_cache.setdefault(self.class_name, {})
        key_cache = model_cache.setdefault(key, {"size": 0, "map": {}})

        old_size = key_cache["size"]  # size of old cached dict
        new_size = len(self.idx.v)   # current number of devices

        arr = self.__dict__[key].v  # device values
        idx_arr = self.idx.v   # device names

        if new_size < old_size:  # rebuild the cache_map in case the size reduces
            cache_map = {}
            for pos, item in enumerate(arr):
                cache_map.setdefault(item, []).append(idx_arr[pos])
            key_cache["map"] = cache_map
            key_cache["size"] = new_size

        elif new_size > old_size:        # if size increases add the new entry into the dictionary
            cache_map = key_cache["map"]
            for pos in range(old_size, new_size):
                item = arr[pos]
                cache_map.setdefault(item, []).append(idx_arr[pos])
            key_cache["size"] = new_size

        return key_cache["map"].get(val, [])

    def find_idx(self, keys, values, allow_none=False, default=False, allow_all=False):
        """
        Find `idx` of devices whose values match the given pattern.

        Parameters
        ----------
        keys : str, array-like, Sized
            A string or an array-like of strings containing the names of parameters for the search criteria
        values : array, array of arrays, Sized
            Values for the corresponding key to search for. If keys is a str, values should be an array of
            elements. If keys is a list, values should be an array of arrays, each corresponds to the key.
        allow_none : bool, Sized, optional
            Allow key, value to be not found. Used by groups.
        default : bool, optional
            Default idx to return if not found (missing)
        allow_all : bool, optional
            If True, returns a list of lists where each nested list contains all the matches for the
            corresponding search criteria.

        Returns
        -------
        list
            indices of devices

        Notes
        -----
        - Only the first match is returned by default.
        - If all matches are needed, set `allow_all` to True.
        - Internally uses a hash-map lookup (O(n + m)) instead of a nested linear
        scan for improved performance on large models. A persistent, incrementally
        updated cache is additionally used for the ``BusFreq`` model via
        `_get_busfreq_cached_matches`, since it is queried very frequently with
        single key/value patterns during initialization.

        Examples
        --------
        >>> # Use example case of IEEE 14-bus system with PVD1
        >>> ss = andes.load(andes.get_case('ieee14/ieee14_pvd1.xlsx'))

        >>> # To find the idx of `PVD1` with `name` of 'PVD1_1' and 'PVD1_2'
        >>> ss.PVD1.find_idx(keys='name', values=['PVD1_1', 'PVD1_2'])
        [1, 2]

        >>> # To find the idx of `PVD1` connected to bus 4
        >>> ss.PVD1.find_idx(keys='bus', values=[4])
        [1]

        >>> # To find ALL the idx of `PVD1` with `gammap` equals to 0.1
        >>> ss.PVD1.find_idx(keys='gammap', values=[0.1], allow_all=True)
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        >>> # To find the idx of `PVD1` with `gammap` equals to 0.1 and `name` of 'PVD1_1'
        >>> ss.PVD1.find_idx(keys=['gammap', 'name'], values=[[0.1], ['PVD1_1']])
        [1]
        """
        keys, values = validate_keys_values(keys, values)

        # No devices in this model
        if len(self.idx.v) == 0:
            n_search = len(values[0])
            if not allow_none:
                raise IndexError(f'No {self.class_name} device found: model is empty')
            return [[default] for _ in range(n_search)] if allow_all else [default] * n_search

        # Special cached path for BusFreq: one key, one search value
        if self.class_name == "BusFreq" and len(keys) == 1 and len(values[0]) == 1:
            # breakpoint()
            key = keys[0]
            val = values[0][0]
            matches = self._get_busfreq_cached_matches(key, val)

            if not matches:
                if not allow_none:
                    raise IndexError(
                        f'No {self.class_name} device found matching {list(keys)}={(val,)}'
                    )
                return [[default]] if allow_all else [default]

            return [matches] if allow_all else [matches[0]]

        v_attrs = [self.__dict__[key].v for key in keys]

        lookup = {}
        for pos, attr_tuple in enumerate(zip(*v_attrs)):
            lookup.setdefault(attr_tuple, []).append(self.idx.v[pos])

        idxes = []
        for v_search in zip(*values):
            matches = lookup.get(v_search, [])

            if not matches:
                if not allow_none:
                    raise IndexError(
                        f'No {self.class_name} device found matching {list(keys)}={v_search}'
                    )
                matches = [default]

            idxes.append(matches if allow_all else matches[0])

        return idxes
    # def find_idx(self, keys, values, allow_none=False, default=False, allow_all=False):
    #     """
    #     Find `idx` of devices whose values match the given pattern.

    #     Parameters
    #     ----------
    #     keys : str, array-like, Sized
    #         A string or an array-like of strings containing the names of parameters for the search criteria
    #     values : array, array of arrays, Sized
    #         Values for the corresponding key to search for. If keys is a str, values should be an array of
    #         elements. If keys is a list, values should be an array of arrays, each corresponds to the key.
    #     allow_none : bool, Sized, optional
    #         Allow key, value to be not found. Used by groups.
    #     default : bool, optional
    #         Default idx to return if not found (missing)
    #     allow_all : bool, optional
    #         If True, returns a list of lists where each nested list contains all the matches for the
    #         corresponding search criteria.

    #     Returns
    #     -------
    #     list
    #         indices of devices

    #     Notes
    #     -----
    #     - Only the first match is returned by default.
    #     - If all matches are needed, set `allow_all` to True.

    #     Examples
    #     --------
    #     >>> # Use example case of IEEE 14-bus system with PVD1
    #     >>> ss = andes.load(andes.get_case('ieee14/ieee14_pvd1.xlsx'))

    #     >>> # To find the idx of `PVD1` with `name` of 'PVD1_1' and 'PVD1_2'
    #     >>> ss.PVD1.find_idx(keys='name', values=['PVD1_1', 'PVD1_2'])
    #     [1, 2]

    #     >>> # To find the idx of `PVD1` connected to bus 4
    #     >>> ss.PVD1.find_idx(keys='bus', values=[4])
    #     [1]

    #     >>> # To find ALL the idx of `PVD1` with `gammap` equals to 0.1
    #     >>> ss.PVD1.find_idx(keys='gammap', values=[0.1], allow_all=True)
    #     [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    #     >>> # To find the idx of `PVD1` with `gammap` equals to 0.1 and `name` of 'PVD1_1'
    #     >>> ss.PVD1.find_idx(keys=['gammap', 'name'], values=[[0.1], ['PVD1_1']])
    #     [1]
    #     """

    #     keys, values = validate_keys_values(keys, values)

    #     v_attrs = [self.__dict__[key].v for key in keys]

    #     idxes = []
    #     for v_search in zip(*values):
    #         v_idx = []
    #         for pos, v_attr in enumerate(zip(*v_attrs)):
    #             if all([i == j for i, j in zip(v_search, v_attr)]):
    #                 v_idx.append(self.idx.v[pos])
    #         if not v_idx:
    #             if allow_none is False:
    #                 raise IndexError(f'No {self.class_name} device found matching {list(keys)}={v_search}')
    #             else:
    #                 v_idx = [default]

    #         if allow_all:
    #             idxes.append(v_idx)
    #         else:
    #             idxes.append(v_idx[0])

    #     return idxes
