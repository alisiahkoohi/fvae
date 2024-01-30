import argparse
import json


def read_config(filename):
    """Read input variables and values from a json file."""
    with open(filename) as f:
        configs = json.load(f)
    return configs


def write_config(args, filename):
    "Write command line arguments into a json file."
    with open(filename, 'w') as f:
        json.dump(args, f)


def parse_input_args(args):
    "Use variables in args to create command line input parser."
    parser = argparse.ArgumentParser(description='')
    for key, value in args.items():
        parser.add_argument('--' + key, default=value, type=type(value))
    return parser.parse_args()


class Namespace:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def create_namespace_from_args(args):
    """
    Create a custom Namespace object from a dictionary.

    Parameters:
    - args: A dictionary where keys are argument names and values are their defaults.

    Returns:
    - namespace: A custom Namespace object.
    """
    return Namespace(**args)


def make_experiment_name(args):
    """Make experiment name based on input arguments"""
    experiment_name = args.experiment_name + '_'
    for key, value in vars(args).items():
        if key not in [
                'experiment_name',
                'cuda',
                'phase',
                'load_to_memory',
                'clip',
                'h5_filename',
                'w_cat',
                'w_gauss',
                'filter_key',
                'normalize',
                'temp_decay',
                'min_temp',
                'init_temp',
                'cluster_n',
                'scale_n',
                'cluster_g',
                'scale_g',
                'extension',
                'event_type',
                'event_quality',
                'umap_n_neighbors',
                'umap_min_dist',
                'umap_n_epochs',
        ]:
            experiment_name += key + '-{}_'.format(value)
    return experiment_name[:-1].replace(' ', '').replace(',', '-')


def make_h5_file_name(args):
    """Make HDF5 file name based on input arguments"""
    filename = args.filename + '_'
    for key, value in vars(args).items():
        if key not in ['cuda', 'filename', 'nchunks', 'filter_key']:
            filename += key + '-{}_'.format(value)
        elif key == 'filter_key':
            filename += key + '-true_'
    filename = filename[:-1].replace(' ', '') + '.h5'
    return filename.replace('[', '').replace(']', '').replace(',', '-')


def process_sequence_arguments(args):
    """Process sequence arguments to remove spaces and split by comma."""
    if hasattr(args, 'filter_key'):
        args.filter_key = args.filter_key.replace(' ', '').split(',')
    if hasattr(args, 'scales'):
        args.scales = args.scales.replace(' ', '').split(',')
    if hasattr(args, 'event_type'):
        args.event_type = args.event_type.replace(' ', '').split(',')
    if hasattr(args, 'event_quality'):
        args.event_quality = args.event_quality.replace(' ', '').split(',')
    if hasattr(args, 'cluster_n'):
        args.cluster_n = args.cluster_n.replace(' ', '').split(',')
        args.cluster_n = [int(_) for _ in args.cluster_n]
    if hasattr(args, 'scale_n'):
        args.scale_n = args.scale_n.replace(' ', '').split(',')
        args.scale_n = [int(_) for _ in args.scale_n]
    if hasattr(args, 'cluster_g'):
        args.cluster_g = args.cluster_g.replace(' ', '').split(',')
        args.cluster_g = [int(_) for _ in args.cluster_g]
    if hasattr(args, 'scale_g'):
        args.scale_g = args.scale_g.replace(' ', '').split(',')
        args.scale_g = [int(_) for _ in args.scale_g]
    if hasattr(args, 'q'):
        args.q = args.q.replace(' ', '').split(',')
        args.q = [int(_) for _ in args.q]
    if hasattr(args, 'j'):
        args.j = args.j.replace(' ', '').split(',')
        args.j = [int(_) for _ in args.j]
    return args
