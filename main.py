import logging
import absl.logging
import os
import click
import importlib


@click.command()
@click.option('--runner', default="run-ppo", help='Choose runner to start')
@click.option('--working_dir', default="./__WORKING_DIRS__/__STANDARD__/", help='Path to working directory')
@click.option('--config', default="", help="Name of config to load. Leave empty for standard variables in cfg class")
def run(runner, working_dir, config):
    r = importlib.import_module('runners.' + runner)
    r.main(working_dir, config)


if __name__ == '__main__':
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run()
