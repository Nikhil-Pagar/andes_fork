"""
PSS/E file parser.

Include a RAW parser and a DYR parser.
"""

import logging
import os
import pickle
import yaml

import andes.io

from andes.core.symprocessor import resolve_deps
from andes.models import file_classes
from andes.utils.func import list_flatten
from andes.shared import deg2rad, pd
from andes.utils.misc import to_number
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-version format descriptors.
# Each entry describes the block structure and column layout for that version.
# ---------------------------------------------------------------------------

_VERSION_CONFIGS = {
    33: {
        'blocks': [
            'bus', 'load', 'fshunt', 'gen', 'branch', 'transf', 'area',
            'twotermdc', 'vscdc', 'impedcorr', 'mtdc', 'msline', 'zone',
            'interarea', 'owner', 'facts', 'swshunt', 'gne', 'Q',
        ],
        'initial_block_idx': 0,
        'line_counts': [1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        'transf_block': 5,
        'branch_cols': {
            'bus1': 0, 'bus2': 1, 'ckt': 2,
            'r': 3, 'x': 4, 'b': 5,
            'rate_a': 6, 'rate_b': 7, 'rate_c': 8,
            'gi': 9, 'bi': 10, 'gj': 11, 'bj': 12,
            'u': 13, 'length': 14,
        },
    },
    34: {
        'blocks': [
            'bus', 'load', 'fshunt', 'gen', 'branch', 'swdev', 'transf',
            'area', 'twotermdc', 'vscdc', 'impedcorr', 'mtdc', 'msline',
            'zone', 'interarea', 'owner', 'facts', 'swshunt', 'gne',
            'indmach', 'substation', 'Q',
        ],
        'initial_block_idx': -1,  # extra "END OF SYSTEM-WIDE DATA" delimiter
        'line_counts': [1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        'transf_block': 6,
        'branch_cols': {
            'bus1': 0, 'bus2': 1, 'ckt': 2,
            'r': 3, 'x': 4, 'b': 5, 'name': 6,
            'rate_a': 7, 'rate_b': 8, 'rate_c': 9,
            'rate_d': 10, 'rate_e': 11, 'rate_f': 12,
            'rate_g': 13, 'rate_h': 14, 'rate_i': 15,
            'rate_j': 16, 'rate_k': 17, 'rate_l': 18,
            'gi': 19, 'bi': 20, 'gj': 21, 'bj': 22,
            'u': 23, 'met': 24, 'length': 25,
        },
    },
}

# v32 is identical to v33 for our purposes
_VERSION_CONFIGS[32] = _VERSION_CONFIGS[33]


def testlines(infile):
    """
    Check the raw file for frequency base.
    """
    lines_list = andes.io.read_file_like(infile)

    if len(lines_list) == 0:
        return False

    first = lines_list[0]
    first = first.strip().split('/')
    first = first[0].split(',')

    # get raw file version
    if len(first) >= 3:
        version = int(first[2])
        logger.debug(f'PSSE raw version {version} detected')

        if version not in _VERSION_CONFIGS:
            logger.warning(f'RAW file version {version} is not explicitly supported. '
                           'Falling back to v33 format. Errors may occur.')

        return True

    else:
        return False


def get_block_lines(b, mdata, cfg):
    """
    Return the number of lines based on the block index in the RAW file.
    """

    if b == cfg['transf_block']:
        if mdata[0][2] == 0:  # two-winding transformer
            return 4
        else:  # three-winding transformer
            return 5

    return cfg['line_counts'][b]

# added by Nikhil : the following helper function and changes


def _get_cache_path(file):
    """
    Return the cache file path for a given RAW file.


    The cache uses a `.pssecache` suffix appended to the original file
    path. The extension name is arbitrary and only serves to distinguish
    the cached binary from the original RAW input.

    Parameters
    ----------
    file : str
        Path to the source RAW file.

    Returns
    -------
    str
        Path to the corresponding cache file.
    """
    return file + '.pssecache'


def _is_cache_valid(source_file, cache_path, addfiles=None):
    """
    Check whether an existing cache file is valid and up to date.

    A cache is considered valid if it exists and its modification time is
    not older than the source RAW file (and, if provided, not older than
    any of the additional dependent files). This prevents stale caches
    from being used after the source file has been edited.

    Parameters
    ----------
    source_file : str
        Path to the source RAW file.
    cache_path : str
        Path to the cache file to validate.
    addfiles : list of str, optional
        Additional file paths whose modification times should also be
        checked against the cache.

    Returns
    -------
    bool
        True if the cache file exists and is at least as recent as the
        source file (and any `addfiles`), False otherwise.
    """
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    if os.path.getmtime(source_file) > cache_mtime:
        return False
    if addfiles:
        for f in addfiles:
            if os.path.exists(f) and os.path.getmtime(f) > cache_mtime:
                return False
    return True


def _save_cache(cache_path, data):
    """
    Serialize parsed RAW data to a binary cache file using pickle.

    Parameters
    ----------
    cache_path : str
        Destination path for the cache file.
    data : dict
        Data to serialize (expected keys: 'raw', 'mva', 'freq', 'version').
    """
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(cache_path):
    """
    Load previously cached parsed RAW data from a binary cache file.


    Parameters
    ----------
    cache_path : str
        Path to the cache file to load.

    Returns
    -------
    dict
        Deserialized data dictionary (expected keys: 'raw', 'mva',
        'freq', 'version').
    """
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def _parse_csv_with_quotes(line):
    """
    Parse a line of PSS/E data that may contain single-quoted strings with commas or slashes.

    Parameters
    ----------
    line : str
        A line from a PSS/E file that needs parsing

    Returns
    -------
    list
        List of values with quoted strings preserved as single elements
    """
    # Initialize result list and variables
    result = []
    current = ""
    in_quotes = False

    # Handle empty input
    if not line:
        return [""]

    # Process each character
    for char in line:
        if char == "'" and not in_quotes:
            # Starting quotes
            in_quotes = True
            current += char
        elif char == "'" and in_quotes:
            # Ending quotes
            in_quotes = False
            current += char
        elif char == ',' and not in_quotes:
            # Field separator outside quotes
            result.append(current)
            current = ""
        else:
            # Add character to current field
            current += char

    # Add the last field
    result.append(current)

    # Process each field to remove quotes and strip whitespace
    for i in range(len(result)):
        field = result[i]
        # Remove quotes and strip
        if field and len(field) >= 2 and field[0] == "'" and field[-1] == "'":
            field = field[1:-1]
        result[i] = field.strip()

    return result


def _split_line_with_quoted_parts(line, separator='/'):
    """
    Split a line by a separator character, but preserve the separator inside quoted strings.

    Parameters
    ----------
    line : str
        Line to split
    separator : str
        Character to split by, defaults to '/'

    Returns
    -------
    list
        List of parts
    """
    result = []
    current = ""
    in_quotes = False

    for char in line:
        if char == "'" and not in_quotes:
            in_quotes = True
            current += char
        elif char == "'" and in_quotes:
            in_quotes = False
            current += char
        elif char == separator and not in_quotes:
            result.append(current)
            current = ""
        else:
            current += char

    if current:
        result.append(current)

    return result


def _parse_raw_file(file):
    """
    Parse a PSS/E RAW file's text content into an intermediate dictionary.

    This function performs the expensive line-by-line text parsing stage
    (block detection, quoted-field splitting, numeric conversion) that was
    previously inline within `read()`. Separating it out allows the parsed
    result to be cached and reused across runs via `read()`.

    Parameters
    ----------
    file : str or file-like
        Path to the RAW file, or a file-like object.

    Returns
    -------
    tuple
        (raw, mva, freq, version) where `raw` is a dict mapping block
        names (e.g., 'bus', 'load', 'branch') to lists of parsed device
        data, `mva` is the system base MVA, `freq` is the nominal system
        frequency, and `version` is the detected RAW file version.
    """
    mva = 100
    freq = 60.0
    version = 0
    data = []
    mdata = []
    dev_line = 0

    line_list = andes.io.read_file_like(file)

    if line_list:
        first_parts = _split_line_with_quoted_parts(line_list[0].strip())
        first_data = _parse_csv_with_quotes(first_parts[0])
        if len(first_data) >= 3:
            version = int(first_data[2])

    cfg = _VERSION_CONFIGS.get(version, _VERSION_CONFIGS[33])
    blocks = cfg['blocks']
    block_idx = cfg['initial_block_idx']
    raw = {item: [] for item in blocks}

    for num, line in enumerate(line_list):
        line = line.strip()

        if num == 0:
            parts = _split_line_with_quoted_parts(line)
            data = _parse_csv_with_quotes(parts[0])
            mva = float(data[1])
            try:
                freq = float(data[5])
            except (IndexError, ValueError):
                logger.debug('System frequency not specified in RAW header. '
                             'Defaulting to 60 Hz.')
                freq = 60.0
            continue

        elif num == 1 or num == 2:
            if len(line) > 0:
                logger.info("  " + line)
            continue

        elif num >= 3:
            if line[0:2] == '0 ' or line[0:3] == ' 0 ':
                block_idx += 1
                continue
            elif line[0] == 'Q':
                break
            elif line.startswith('@!'):
                continue
            parts = _split_line_with_quoted_parts(line)
            data = _parse_csv_with_quotes(parts[0])

        data = [to_number(item) for item in data]
        mdata.append(data)
        dev_line += 1

        block_lines = get_block_lines(block_idx, mdata, cfg)
        if dev_line >= block_lines:
            if block_lines == 1:
                mdata = mdata[0]
            raw[blocks[block_idx]].append(mdata)
            mdata = []
            dev_line = 0

    return raw, mva, freq, version


def read(system, file):
    """
    Read PSS/E RAW file v32/v33/v34 formats.

    Parameters
    ----------
    system : System
        System instance to which parsed data will be loaded.
    file : str or os.PathLike or file-like
        Path to the RAW file. Caching is only enabled when `file` is an
        actual path (str or os.PathLike); file-like objects bypass caching.

    Returns
    -------
    bool
        True if parsing succeeded.

    Notes
    -----
    - If a valid `.pssecache` file exists for `file` (see `_is_cache_valid`),
    the expensive text-parsing stage is skipped and the cached parsed
    data is loaded directly via `_load_cache`.
    - Otherwise, the file is parsed via `_parse_raw_file`, and the result
    is saved to a `.pssecache` file for reuse on subsequent runs.
    - Cache load/save failures are logged as warnings and do not interrupt
    execution; parsing falls back to the standard (non-cached) path.
    - System object construction (bus, load, generator, line, transformer,
    shunt, area parsing) is unchanged from the original implementation.
    """
    ret = True

    # Caching only works when `file` is an actual path, not an open file object
    use_cache = isinstance(file, (str, os.PathLike))
    cache_path = _get_cache_path(file) if use_cache else None

    if use_cache and _is_cache_valid(file, cache_path):
        logger.info('Loading PSS/E cache from: %s', cache_path)
        try:
            cached = _load_cache(cache_path)
            raw = cached['raw']
            mva = cached['mva']
            freq = cached['freq']
            version = cached['version']
            logger.info('PSS/E cache loaded successfully.')
        except Exception as e:
            logger.warning('Cache load failed (%s). Re-parsing.', e)
            raw, mva, freq, version = _parse_raw_file(file)
            _save_cache(cache_path, {'raw': raw, 'mva': mva,
                                     'freq': freq, 'version': version})
    else:
        raw, mva, freq, version = _parse_raw_file(file)
        try:
            _save_cache(cache_path, {'raw': raw, 'mva': mva,
                                     'freq': freq, 'version': version})
            logger.info('PSS/E cache saved to: %s', cache_path)
        except Exception as e:
            logger.warning('Cache save failed (%s). Continuing.', e)

    system.config.mva = mva
    system.config.freq = freq

    cfg = _VERSION_CONFIGS.get(version, _VERSION_CONFIGS[33])

    bus_params, bus_idx_list, sw = _parse_bus_v33(raw, system)
    max_bus = max(bus_idx_list)

    _parse_load_v33(raw, system)
    _parse_fshunt_v33(raw, system)
    _parse_gen_v33(raw, system, sw)
    _parse_line(raw, system, cfg)
    _parse_transf_v33(raw, system, max_bus)
    _parse_swshunt_v33(raw, system)
    _parse_area_v33(raw, system)

    return ret


def _read_dyr_dict(file):
    """
    Parse dyr file into a dict where keys are model names and values are dataframes.
    """
    input_list = andes.io.read_file_like(file)

    # concatenate multi-line device data
    input_concat_dict = defaultdict(list)
    multi_line = list()
    for line in input_list:
        if line == '':
            continue
        if '/' not in line:
            multi_line.append(line)
        else:
            multi_line.append(line.split('/')[0])
            single_line = ' '.join(multi_line)

            if single_line.strip() == '':
                continue

            single_list = single_line.split("'")

            psse_model = single_list[1].strip()
            input_concat_dict[psse_model].append(single_list[0] + ' '.join(single_list[2:]))
            multi_line = list()

    # construct pandas dataframe for all models
    dyr_dict = dict()   # input data from dyr file

    for psse_model, all_rows in input_concat_dict.items():
        # DYR files are space-separated, not comma-separated
        dev_params_num = [([to_number(cell) for cell in row.replace(',', ' ').split()]) for row in all_rows]
        dyr_dict[psse_model] = pd.DataFrame(dev_params_num)

    return dyr_dict


# # addded by Nikhil:

# def _norm_id(x):
#     return str(x).strip().strip("'").strip('"')


def _filter_offline_dyr_models(system, dyr_dict):
    """
    Remove DYR rows corresponding to generators that are offline in RAW.
    Generators with STAT == 0 in the RAW file are skipped during static
    device construction (see `_parse_gen_v33`) and recorded in
    `system._offline_gens` as (bus, machine_id) tuples. This function uses
    that set to drop matching rows from each DYR model's DataFrame, so that
    dynamic model data is not instantiated for generators that do not
    exist as static devices.

    Matching key is (bus, machine_id), read from the first two columns of
    each DYR model's DataFrame.

    Parameters
    ----------
    system : System
        System instance containing `_offline_gens` (populated during RAW
        parsing).
    dyr_dict : dict
        Dict mapping PSS/E dynamic model names to their parsed DataFrame
        of device rows.

    Returns
    -------
    dict
        Filtered `dyr_dict` with rows referencing offline generators removed.
        Models with empty data or fewer than 2 columns are returned unchanged.
    """
    offline_gens = getattr(system, '_offline_gens', set())
    if not offline_gens:
        return dyr_dict

    filtered = {}

    for model_name, df in dyr_dict.items():
        if df.empty:
            filtered[model_name] = df
            continue

        # only try if at least two columns exist
        if df.shape[1] < 2:
            filtered[model_name] = df
            continue

        keep_rows = []
        for i in range(len(df)):
            try:
                bus = int(df.iloc[i, 0])
                # mach = str(df.iloc[i, 1]).strip()
                # mach = _norm_id(df.iloc[i, 1])
                mach = df.iloc[i, 1]
            except Exception:
                keep_rows.append(True)
                continue

            if (bus, mach) in offline_gens:
                logger.info("Skipping DYR model %s for offline generator (%s, %s)",
                            model_name, bus, mach)
                keep_rows.append(False)
            else:
                keep_rows.append(True)

        filtered[model_name] = df.loc[keep_rows].reset_index(drop=True)

    return filtered


def read_add(system, file):
    """
    Read an addition PSS/E dyr file.
    Parameters
    ----------
    system : System
        System instance to which data will be loaded
    file : str
        Path to the additional `dyr` file

    Returns
    -------
    bool
        data parsing status

    Notes
    -----
    - Rows referencing generators marked offline in the RAW file (STAT == 0)
    are filtered out via `_filter_offline_dyr_models` before dynamic
    model construction, to keep DYR data consistent with the static
    generator set.
    """
    # original code
    # dyr_dict = _read_dyr_dict(file)
    # system.dyr_dict = dyr_dict

    # added by Nikhil: the 3 lines below this
    dyr_dict = _read_dyr_dict(file)
    dyr_dict = _filter_offline_dyr_models(system, dyr_dict)
    system.dyr_dict = dyr_dict

    # read yaml and set header for each pss/e model
    dirname = os.path.dirname(__file__)
    with open(f'{dirname}/psse-dyr.yaml', 'r') as f:
        dyr_yaml = yaml.full_load(f)

    sorted_models = sort_psse_models(dyr_yaml, system)

    for psse_model in dyr_dict:
        if psse_model in dyr_yaml:
            if 'inputs' in dyr_yaml[psse_model]:
                dyr_dict[psse_model].columns = dyr_yaml[psse_model]['inputs']

    # collect not supported models
    not_supported = []
    for model in dyr_dict:
        if model not in sorted_models:
            not_supported.append(model)

    # print out debug messages
    if len(dyr_dict):
        logger.debug('dyr contains models %s', ", ".join(dyr_dict.keys()))

    if len(not_supported):
        logger.warning('Models not yet supported: %s', ", ".join(not_supported))
    else:
        logger.debug('All dyr models are supported.')

    # load data into models
    for psse_model in sorted_models:
        if psse_model not in dyr_dict:
            # device not exist
            continue

        if psse_model not in dyr_yaml:
            logger.error(f"PSS/E Model <{psse_model}> is not supported.")
            continue

        logger.debug(f'Parsing PSS/E model {psse_model}')

        dest = dyr_yaml[psse_model]['destination']
        find = {}

        if 'find' in dyr_yaml[psse_model]:
            for name, source in dyr_yaml[psse_model]['find'].items():

                for model, conditions in source.items():
                    allow_none = conditions.pop('allow_none', 0)
                    cond_names = conditions.keys()
                    cond_values = []

                    for col in conditions.values():
                        if col in find:
                            cond_values.append(find[col])
                        else:
                            cond_values.append(dyr_dict[psse_model][col])

                    try:
                        logger.debug("<%s> trying to find <%s> using cond_names=%s and cond_values=%s",
                                     psse_model, model, cond_names, cond_values)
                        logger.debug("<%s> contains %d devices", model, system.__dict__[model].n)

                        find[name] = system.__dict__[model].find_idx(cond_names, cond_values,
                                                                     allow_none=allow_none,
                                                                     default=None)
                    except IndexError as e:
                        logger.error("Data file likely contains references to unsupported models.")
                        logger.error(e)
                        return False

        if 'get' in dyr_yaml[psse_model]:
            for name, source in dyr_yaml[psse_model]['get'].items():
                for model, conditions in source.items():
                    idx_name = conditions['idx']
                    if idx_name in dyr_dict[psse_model]:
                        conditions['idx'] = dyr_dict[psse_model][idx_name]
                    else:
                        conditions['idx'] = find[idx_name]
                    find[name] = system.__dict__[model].get(**conditions)

        if 'outputs' in dyr_yaml[psse_model]:
            output_keys = list(dyr_yaml[psse_model]['outputs'].keys())
            output_exprs = list(dyr_yaml[psse_model]['outputs'].values())
            out_dict = {}

            for idx in range(len(output_exprs)):
                out_key = output_keys[idx]
                expr = output_exprs[idx]
                if expr in find:
                    out_dict[out_key] = find[expr]
                elif ';' in expr:
                    args, func = expr.split(';')
                    func = eval(func)
                    args = args.split(',')
                    # support local and external model parameters
                    argv = list()
                    for param in args:
                        if '.' in param:
                            argv.append(param.split('.'))
                        else:
                            argv.append((psse_model, param))
                    argv = [dyr_dict[model][param] for model, param in argv]
                    out_dict[output_keys[idx]] = func(*argv)
                else:
                    out_dict[output_keys[idx]] = dyr_dict[psse_model][expr]

            df = pd.DataFrame.from_dict(out_dict)
            for row in df.to_dict(orient='records'):
                system.add(dest, row)

        system.link_ext_param(system.__dict__[dest])

    return True


def _parse_bus_v33(raw, system):
    # version 32:
    #   0,   1,      2,     3,    4,   5,  6,   7,  8
    #   ID, NAME, BasekV, Type, Area Zone Owner Vm, Va
    #
    out = defaultdict(list)
    bus_idx_list = list()
    sw = dict()

    for data in raw['bus']:
        idx = data[0]
        bus_idx_list.append(idx)
        ty = data[3]
        a0 = data[8] * deg2rad

        if ty == 3:
            sw[idx] = a0
        param = {'idx': idx, 'name': data[1], 'Vn': data[2],
                 'v0': data[7], 'a0': a0,
                 'area': data[4], 'zone': data[5], 'owner': data[6]}
        out['Bus'].append(param)

    _add_devices_from_dict(out, system)

    return out, bus_idx_list, sw


def _parse_load_v33(raw, system):
    # version 32:
    #  0,  1,      2,    3,    4,    5,    6,      7,   8,  9, 10,   11
    # Bus, Id, Status, Area, Zone, PL(MW), QL (MW), IP, IQ, YP, YQ, OWNER

    mva = system.config.mva
    out = defaultdict(list)

    for data in raw['load']:
        bus = data[0]

        vn = system.Bus.get(src='Vn', idx=bus, attr='v')
        v0 = system.Bus.get(src='v0', idx=bus, attr='v')

        param = {'bus': bus, 'u': data[2], 'Vn': vn,
                 'p0': (data[5] + data[7] * v0 + data[9] * v0 ** 2) / mva,
                 'q0': (data[6] + data[8] * v0 - data[10] * v0 ** 2) / mva,
                 'owner': data[11]}

        out['PQ'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_fshunt_v33(raw, system):
    # 0,    1,      2,      3,      4
    # Bus, name, Status, g (MW), b (Mvar)

    mva = system.config.mva
    out = defaultdict(list)

    for data in raw['fshunt']:
        bus = data[0]
        vn = system.Bus.get(src='Vn', idx=bus, attr='v')

        param = {'bus': bus, 'Vn': vn, 'u': data[2],
                 'Sn': mva, 'g': data[3] / mva, 'b': data[4] / mva}

        out['Shunt'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_gen_v33(raw, system, sw):
    """
    Helper function for parsing static generator section.

    Notes
    -----
    Generators with STAT == 0 (offline) are excluded from device
    construction. Their (bus, subidx) pairs are recorded in
    `system._offline_gens` so that corresponding DYR dynamic model rows
    can be filtered out later via `_filter_offline_dyr_models`.
    """

    # 0,  1,  2,  3,  4,  5,  6,    7,     8,  9, 10, 11, 12,   13,   14,
    # I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT,
    #    15, 16, 17, 18, 19, ...,           26,  27
    # RMPCT, PT, PB, O1, F1, ..., O4, F4, WMOD, WPF
    # The columns above for v33 is different from the manual of v34.5, which includes two new columns:
    # `NREG`` at 8 and `BSLOD` before `O1`

    mva = system.config.mva
    out = defaultdict(list)
    gen_idx = 0

    for data in raw['gen']:

        bus = data[0]
        subidx = data[1]
        vn = system.Bus.get(src='Vn', idx=bus, attr='v')
        gen_mva = data[8]
        # the follwoing is original code
        # gen_idx += 1
        # status = data[14]

        # # modified by Nikhil
        ##############################
        status = data[14]
        # # subidx = str(data[1]).strip()
        # subidx = _norm_id(data[1])
        subidx = data[1]

        if not hasattr(system, '_offline_gens'):
            system._offline_gens = set()

        if status == 0:
            system._offline_gens.add((bus, subidx))
            continue

        gen_idx += 1
        ##############################
        wmod = data[26] if len(data) > 26 else 0

        param = {'Sn': gen_mva, 'Vn': vn, 'u': status,
                 'bus': bus, 'subidx': subidx,
                 'idx': gen_idx,
                 'p0': data[2] / mva,
                 'q0': data[3] / mva,
                 'pmax': data[16] / mva, 'pmin': data[17] / mva,
                 'qmax': data[4] / mva, 'qmin': data[5] / mva,
                 'v0': data[6],
                 'ra': data[9],   # ra - armature resistance
                 'xs': data[10],  # xs - synchronous reactance
                 'wmod': wmod,  # generator control mode
                 }

        if data[0] in sw.keys():
            param.update({'a0': sw[data[0]]})
            out['Slack'].append(param)
        else:
            out['PV'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_line(raw, system, cfg):
    # Column indices come from cfg['branch_cols'], which varies by version.
    # v33: I,J,CKT,R,X,B,RATEA,RATEB,RATEC,GI,BI,GJ,BJ,ST,LEN,O1,F1,...
    # v34: I,J,CKT,R,X,B,NAME,RATE1..RATE12,GI,BI,GJ,BJ,ST,MET,LEN,...

    cols = cfg['branch_cols']
    out = defaultdict(list)
    for data in raw['branch']:
        param = {
            'u': data[cols['u']],
            'bus1': data[cols['bus1']], 'bus2': data[cols['bus2']],
            'r': data[cols['r']], 'x': data[cols['x']], 'b': data[cols['b']],
            'rate_a': data[cols['rate_a']],
            'rate_b': data[cols['rate_b']],
            'rate_c': data[cols['rate_c']],
            'Vn1': system.Bus.get(src='Vn', idx=data[cols['bus1']], attr='v'),
            'Vn2': system.Bus.get(src='Vn', idx=data[cols['bus2']], attr='v'),
        }
        out['Line'].append(param)

    _add_devices_from_dict(out, system)

    return out

# This is modified version by Nikhil


def _parse_transf_v33(raw, system, max_bus):
    """
    Parse 2-winding and 3-winding transformer data into `Line` (and,
    for 3-winding transformers, auxiliary `Bus`) device parameters.

    Parameters
    ----------
    raw : dict
        Parsed RAW data blocks, as returned by `_parse_raw_file`.
    system : System
        System instance to which transformer-derived `Line`/`Bus` devices
        will be added.
    max_bus : int
        Maximum existing bus index, used to assign a non-conflicting
        auxiliary bus index for 3-winding transformer star points.

    Returns
    -------
    tuple
        (out, xf_3_count) where `out` is a dict of device parameters added
        to `system`, and `xf_3_count` is the count of 3-winding transformers
        processed (used for auxiliary bus index assignment).

    Notes
    -----
    Bus attribute lookups (Vn, area, zone, owner) are precomputed into
    local dictionaries (`bus_vn`, `bus_area`, `bus_zone`, `bus_owner`)
    before the main loop, instead of calling `system.Bus.get()` per
    transformer record, to reduce repeated lookup overhead for large
    systems. Numerical conversion logic (CW/CZ/CM codes, impedance and
    magnetizing admittance calculations) is unchanged from the original
    implementation.
    """
    out = defaultdict(list)
    xf_3_count = 1

    # local shortcuts for speed
    system_mva = system.config.mva

    bus_idx = list(system.Bus.idx.v)
    bus_idx_set = set(bus_idx)
    bus_vn = dict(zip(bus_idx, system.Bus.Vn.v))
    bus_area = dict(zip(bus_idx, system.Bus.area.v))
    bus_zone = dict(zip(bus_idx, system.Bus.zone.v))
    bus_owner = dict(zip(bus_idx, system.Bus.owner.v))

    for data in raw['transf']:
        hdr = data[0]

        if len(data) == 4:
            d1 = data[1]
            w1 = data[2]
            w2 = data[3]

            bus1 = hdr[0]
            bus2 = hdr[1]
            cw = hdr[4]
            cz = hdr[5]
            cm = hdr[6]
            u = hdr[11]

            bus_Vn1 = bus_vn[bus1]
            bus_Vn2 = bus_vn[bus2]

            Vn1 = w1[1] if w1[1] != 0.0 else bus_Vn1
            Vn2 = w2[1] if w2[1] != 0.0 else bus_Vn2

            tap = w1[0]
            phi = w1[2] * deg2rad
            rate_a = w1[3]
            rate_b = w1[4]
            rate_c = w1[5]

            # CW
            if cw == 2:
                tap = (w1[0] / bus_Vn1) / (w2[0] / bus_Vn2)
            elif cw == 3:
                tap = tap * (Vn1 / bus_Vn1) / (Vn2 / bus_Vn2)

            if tap == 0.0:
                tap = 1.0

            # CZ
            if cz == 1:
                Sn = system_mva
            elif cz == 2:
                Sn = d1[2]
            elif cz == 3:
                Sn = d1[2]
                r_pu_wb = d1[0] / (Sn * 1e6)
                z2 = d1[1] * d1[1]
                r2 = r_pu_wb * r_pu_wb

                if z2 > r2:
                    x_pu_wb = (z2 - r2) ** 0.5
                else:
                    logger.warning(f"Branch {bus1}-{bus2}:")
                    logger.warning("  CZ=3 conversion issue: |Z|^2 < R^2. Setting X to small value.")
                    x_pu_wb = 1e-6

                d1[0] = r_pu_wb
                d1[1] = x_pu_wb
            else:
                logger.warning('Unknown impedance code %s', cz)
                Sn = system_mva

            # CM
            if cm == 2:
                mag1 = hdr[7]
                mag2 = hdr[8]

                g_s = mag1 / (Vn1 * Vn1) / 1e6
                y_mag_s = abs(mag2) * system_mva / (Vn1 * Vn1)

                if y_mag_s * y_mag_s > g_s * g_s:
                    b_s = (y_mag_s * y_mag_s - g_s * g_s) ** 0.5
                else:
                    logger.warning("CM=2 conversion issue: |Y|^2 < G^2. Setting B to small value.")
                    b_s = 1e-6

                hdr[7] = g_s * (Vn1 * Vn1) / system_mva
                hdr[8] = b_s * (Vn1 * Vn1) / system_mva
            elif cm != 1:
                logger.warning('Unknown magnetizing admittance code %s', cm)

            out['Line'].append({
                'bus1': bus1,
                'bus2': bus2,
                'u': u,
                'b': hdr[8],
                'r': d1[0],
                'x': d1[1],
                'trans': True,
                'tap': tap,
                'phi': phi,
                'Sn': Sn,
                'Vn1': Vn1,
                'Vn2': Vn2,
                'rate_a': rate_a,
                'rate_b': rate_b,
                'rate_c': rate_c,
            })

        else:
            if hdr[5] == 3:
                data = _process_3wt_cz3(data, system)
                data[0][5] = 2

            if data[0][6] == 2:
                data = _process_3wt_cm2(data, system)
                data[0][6] = 1

            hdr = data[0]
            d1 = data[1]

            bus1 = hdr[0]
            bus2 = hdr[1]
            bus3 = hdr[2]

            new_bus = bus3 + 1
            if new_bus in bus_idx_set:
                new_bus = max_bus + xf_3_count
                logger.debug('Added bus <%s> for 3-winding transformer <%s-%s-%s>',
                             new_bus, bus1, bus2, bus3)

            out['Bus'].append({
                'idx': new_bus,
                'name': f'{bus1}_{bus2}_{bus3}',
                'Vn': 1.0,
                'v0': d1[-2],
                'a0': d1[-1] * deg2rad,
                'area': bus_area[bus1],
                'owner': bus_owner[bus1],
                'zone': bus_zone[bus1],
            })

            r_12 = d1[0]
            x_12 = d1[1]
            r_23 = d1[3]
            x_23 = d1[4]
            r_31 = d1[6]
            x_31 = d1[7]

            if hdr[5] == 2:
                r_12 = r_12 * system_mva / d1[2]
                x_12 = x_12 * system_mva / d1[2]
                r_23 = r_23 * system_mva / d1[5]
                x_23 = x_23 * system_mva / d1[5]
                r_31 = r_31 * system_mva / d1[8]
                x_31 = x_31 * system_mva / d1[8]

            rvals = [
                (r_12 + r_31 - r_23) / 2,
                (r_23 + r_12 - r_31) / 2,
                (r_31 + r_23 - r_12) / 2,
            ]
            xvals = [
                (x_12 + x_31 - x_23) / 2,
                (x_23 + x_12 - x_31) / 2,
                (x_31 + x_23 - x_12) / 2,
            ]

            for i, bus in enumerate((bus1, bus2, bus3)):
                w = data[2 + i]

                out['Line'].append({
                    'trans': True,
                    'bus1': bus,
                    'bus2': new_bus,
                    'g1': hdr[7] if i == 0 else 0.0,
                    'b1': hdr[8] if i == 0 else 0.0,
                    'r': rvals[i],
                    'x': xvals[i],
                    'tap': w[0] if w[0] != 0.0 else 1.0,
                    'phi': w[2] * deg2rad,
                    'Vn1': bus_vn[bus],
                    'Vn2': 1.0,
                    'Sn': system_mva,
                })

            xf_3_count += 1

    _add_devices_from_dict(out, system)

    return out, xf_3_count


def _parse_swshunt_v33(raw, system):

    # I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, RMIDNT,
    # BINIT, N1, B1, N2, B2, ... N8, B8

    out = defaultdict(list)
    mva = system.config.mva
    for data in raw['swshunt']:
        bus = data[0]
        vn = system.Bus.get(src='Vn', idx=bus, attr='v')
        param = {'bus': bus, 'Vn': vn, 'Sn': mva, 'u': data[3],
                 'b': data[9] / mva}

        out['Shunt'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_area_v33(raw, system):

    out = defaultdict(list)

    for data in raw['area']:

        # ID, ISW, PDES, PTOL, ARNAME

        param = {'idx': data[0], 'name': data[4],
                 # 'isw': data[1],
                 # 'pdes': data[2],
                 # 'ptol': data[3],
                 }

        out['Area'].append(param)

    for data in raw['zone']:
        # """ID, NAME"""
        param = {'idx': data[0], 'name': data[1]}

        # TODO: add back
        # system.add('Zone', param)

    _add_devices_from_dict(out, system)

    return out


def _add_devices_from_dict(params, system):
    """
    Add devices from a dict, where the key is the
    model name, and the value is a list of parameters.

    """
    for name, plist in params.items():
        for p in plist:
            system.add(name, p)


def sort_psse_models(dyr_yaml, system):
    """
    Sort supported models so that model names are ordered by dependency.

    Dependency is determined by checking the ``find`` key in
    ``psse-dyr.yaml`` for each model.

    Returns
    -------
    list
        The sequence of model names for loading parameters.
    """
    _, andes_models = map(list, zip(*file_classes))
    andes_models = list_flatten(andes_models)
    graph = defaultdict(list)

    for psse_model in dyr_yaml:

        # build model dependency graph
        if 'find' in dyr_yaml[psse_model]:
            # below is a list of `dict_keys`
            dep_models = [item.keys() for item in dyr_yaml[psse_model]['find'].values()]

            # extract strings from single-element `dict_keys`
            dep_models = [list(item)[0] for item in dep_models]
            graph[psse_model].extend(dep_models)

    # resolve PSS/E model dependency based on `find`
    # `find` requires the model indices to have existed
    for key in graph:
        value = list(graph[key])
        for item in graph[key]:
            if item in system.groups:
                value.remove(item)
                value.extend(list(system.groups[item].models.keys()))
        graph[key] = value

    sequence = resolve_deps(graph)
    # does not support ciruclar dependencies
    for item in sequence:
        if not isinstance(item, str):
            raise NotImplementedError("Circular depencency exists for models %s. %s".
                                      item, "Please report this bug.")

    # prepend no-dependency models to the front
    nodep_models = list()
    for item in andes_models:
        if item not in sequence:
            nodep_models.append(item)

    return nodep_models + sequence


def _process_3wt_cz3(data, system):
    """
    Process 3-winding transformer data with CZ=3 (Load loss & |Z|).

    Converts power loss (W) and impedance magnitude to R and X on winding base (CZ=2).

    Parameters
    ----------
    data : list
        Multi-line transformer data
    system : System
        The ANDES system object

    Returns
    -------
    list
        Updated data with converted R and X values
    """
    # For each winding pair, convert load loss and |Z| to R and X
    # Winding pairs are 1-2, 2-3, and 3-1
    sbase = [data[1][2], data[1][5], data[1][8]]  # SBASE1-2, SBASE2-3, SBASE3-1

    # Data indices for each winding pair
    indices = [
        (0, 1),  # R1-2, X1-2 indices
        (3, 4),  # R2-3, X2-3 indices
        (6, 7),  # R3-1, X3-1 indices
    ]

    for i, (loss_idx, z_idx) in enumerate(indices):
        # Load loss in watts
        loss = data[1][loss_idx]
        # |Z| in pu on winding base
        z_mag = data[1][z_idx]

        # Convert load loss to R in pu on winding base
        r_pu = loss / (sbase[i] * 1e6)

        # Calculate X from |Z| and R: X = sqrt(|Z|^2 - R^2)
        # Handle numeric issues - ensure we don't get imaginary numbers
        if z_mag**2 > r_pu**2:
            x_pu = (z_mag**2 - r_pu**2)**0.5
        else:
            # If |Z| is too small compared to R, assume X is very small
            logger.warning(
                f"CZ=3 conversion issue: |Z|^2 < R^2 for winding pair {
                    i + 1}. Setting X to small value.")
            x_pu = 1e-6

        # Replace the original values with calculated R and X
        data[1][loss_idx] = r_pu
        data[1][z_idx] = x_pu

    return data


def _process_3wt_cm2(data, system):
    """
    Process 3-winding transformer data with CM=2 (No load loss & exc. loss).

    Converts no-load loss and excitation current to G and B on system base (CM=1).

    Parameters
    ----------
    data : list
        Multi-line transformer data
    system : System
        The ANDES system object

    Returns
    -------
    list
        Updated data with converted G and B values
    """
    # Get no-load loss in watts and excitation current in pu
    mag1 = data[0][7]  # No-load loss in watts
    mag2 = data[0][8]  # Excitation current in pu

    # Get the rated voltage of winding 1 (kV)
    Vn1 = data[2][1] if data[2][1] != 0.0 else system.Bus.get(src='Vn', idx=data[0][0], attr='v')

    # Convert power loss to conductance (G) in Siemens
    # G [S] = MAG1 / VNOM1_V^2 / 1e6
    g_s = mag1 / (Vn1 ** 2) / 1e6

    # Convert |Y| pu to actual in Siemens
    # |Y| [S] = |MAG2| * SBASE_MVA / NOMV1_kV^2
    y_mag_s = abs(mag2) * system.config.mva / (Vn1**2)

    # Calculate B in Siemens
    # B [S] = sqrt(|Y|^2 - G^2)
    # Handle numeric issues
    if y_mag_s**2 > g_s**2:
        b_s = (y_mag_s**2 - g_s**2)**0.5
        logger.info(f"CM=2 Conversion ok: G={g_s}, B={b_s}")
    else:
        # If |Y| is too small compared to G, assume B is very small
        logger.warning(f"Branch {data[0][0]}-{data[0][1]}-{data[0][2]}:")
        logger.warning("  CM=2 conversion issue: |Y|^2 < G^2. Setting B to small value.")
        b_s = 1e-6

    # Convert G and B to per unit on system base
    g_pu = g_s * (Vn1**2) / system.config.mva
    b_pu = b_s * (Vn1**2) / system.config.mva

    # Replace MAG1 and MAG2 with calculated G and B in pu
    data[0][7] = g_pu
    data[0][8] = b_pu

    return data
