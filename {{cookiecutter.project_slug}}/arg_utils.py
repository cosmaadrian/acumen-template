import pprint
import yaml
import os
import argparse
import sys
import easydict
import warnings
from collections import defaultdict


def extend_config(cfg_path, child = None):
    if not os.path.exists(cfg_path):
        warnings.warn(f'::: File {cfg_path} was not found!')
        return child

    with open(cfg_path, 'rt') as f:
        parent_cfg = yaml.load(f, Loader = yaml.FullLoader)

    if child is not None:
        parent_cfg.update(child)

    if '$extends$' in parent_cfg:
        path = parent_cfg['$extends$']
        del parent_cfg['$extends$']
        parent_cfg = extend_config(child = parent_cfg, cfg_path = path)

    return parent_cfg

def load_args(args):
    cfg = extend_config(cfg_path = f'{args.config_file}', child = None)

    if cfg is not None:
        for key, value in cfg.items():
            if key in args and args.__dict__[key] is not None:
                continue

            if type(args) != dict:
                args.__dict__[key] = value
            else:
                args[key] = value

    return args, cfg

def flatten_dict_keys(original_key, values):
	new_values = []
	for key, value in values.items():
		new_key = original_key + '.' + key
		if type(value) == list:
			continue
		if type(value) != dict:
			new_values.append((new_key, value, type(value)))
		elif type(value) == dict:
			new_values.extend(flatten_dict_keys(new_key, value))

	return new_values

def unflatten_dict_keys(dict_args, args):
	dict_values = {** {key: value for key, value in args.items() if '.' not in key}, **dict_args}

	unnested = defaultdict(dict)
	for key, value in args.items():
		if '.' not in key:
			continue

		root = '.'.join(key.split('.')[:-1])
		value = {key.split('.')[-1]: value}
		unnested[root].update(value)

	if len(unnested):
		dict_values = unflatten_dict_keys(dict_values.copy(), unnested)

	return dict_values

def update_parser(parser, args):
	for key, value in args.__dict__.items():
		if type(value) == dict:
			new_values = flatten_dict_keys(key, value)

			for arg_name, default_value, arg_type in new_values:
				parser.add_argument(f'--{arg_name}', type = arg_type, default = default_value, required = False)

			continue

		if key == 'config_file':
			continue

		parser.add_argument(f'--{key}', type = type(value), default = value)

	return parser

def define_args():
	config_path = None
	for i in range(len(sys.argv)):
		if sys.argv[i] == '--config_file':
			config_path = sys.argv[i + 1]
			break

	if config_path is None:
		raise Exception('::: Config path is required!')

	# Removing the config file from args
	sys.argv.pop(i)
	sys.argv.pop(i)


	args = argparse.Namespace()
	args.config_file = config_path

	args, cfg = load_args(args)
	parser = argparse.ArgumentParser(description='Do stuff.')
	parser.add_argument('--name', type = str, default = 'test')
	parser.add_argument('--group', type = str, default = 'default')
	parser.add_argument('--notes', type = str, default = '')
	parser.add_argument("--mode", type = str, default = 'dryrun')

	parser.add_argument('--use_amp', type = int, default = 1, required = False)

	parser.add_argument('--env', type = str, default = 'env1')

	# Needed to be able to update nested config keys
	parser = update_parser(parser = parser, args = args)
	args = parser.parse_args()

	# Make an EasyDict with all the args. This is used in all the main actors.
	nested_args = unflatten_dict_keys(dict(), args.__dict__)
	args = easydict.EasyDict(nested_args)

	if os.path.exists('configs/env_config.yaml'):
		with open('configs/env_config.yaml', 'rt') as f:
		    env_cfg = yaml.load(f, Loader = yaml.FullLoader)
		args.environment = env_cfg[args.env]

	os.environ['WANDB_MODE'] = args.mode
	os.environ['WANDB_NAME'] = args.name
	os.environ['WANDB_NOTES'] = args.notes

	return args
