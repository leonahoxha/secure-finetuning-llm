# -----------------------------------------------------------------------------------
# File: configurator.py
# Source: Official NanoGPT Repository by Andrej Karpathy
# URL: https://github.com/karpathy/nanoGPT/blob/master/configurator.py
# License: MIT License
#
# This file is used unmodified as part of Leona Hoxha’s master thesis project:
# "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Constructor University, 2025).
#
# Description from original author:
# > Poor Man’s Configurator. Loads a Python config file and allows command-line
# > overrides like --key=value. Used for minimal configuration in training.
# -----------------------------------------------------------------------------------

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
