#!/usr/bin/env python

"""Wrapper around train.py.

It runs training for multiple configuration consecutively.  The output
is captured in log files.  All produced files are copied to the
destinations specified in repsective configuration files (normally, a
GCP bucket).
"""

import argparse
from datetime import datetime
import os
import shutil
import subprocess

import yaml


def run_task(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    try:
        run_name = config['run']['name']
    except KeyError:
        run_name = os.path.splitext(os.path.basename(config_path))[0]
    dir_name = '{:%Y%m%dT%H%M%S}_{}'.format(datetime.now(), run_name)
    destination = os.path.join(config['run']['storage'], dir_name)

    os.makedirs(dir_name)
    shutil.copy(config_path, dir_name)

    stdout = open(os.path.join(dir_name, 'train.stdout'), 'w')
    stderr = open(os.path.join(dir_name, 'train.stderr'), 'w')
    subprocess.run(
        ['train.py', config_path, '-o', dir_name],
        stdout=stdout, stderr=stderr
    )
    stdout.close()
    stderr.close()

    subprocess.run([
        'gsutil', '-q', 'cp', '-r', dir_name, destination
    ])


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        'configs', nargs='+',
        help='Paths to configuration files of tasks to execute.'
    )
    args = arg_parser.parse_args()

    for path in args.configs:
        print(f'Running configuration "{path}"...')
        run_task(path)
