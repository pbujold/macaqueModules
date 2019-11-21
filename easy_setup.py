# -*- coding: utf-8 -*-
"""
Setup script for the imacaque package
"""

import subprocess
import sys
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

print('------------------------------------------------------------------')
print('Checking packages required for the Macaque Data analysis package:')
print('------------------------------------------------------------------')
import conda.cli
required = ['scipy', 'matplotlib', 'seaborn', 'numpy', 'statsmodels',
            'pandas', 'tqdm', 'tabulate', 'r-base', 'r-essentials', 'rpy2', 
            'r-lsmeans', 'dill', 'tzlocal']
for module in required:
    if module in installed_packages:
        print(module + ' already installed')
    else:
        print('installing:' + module)
        conda.cli.main('conda', 'install',  '-y', module)
