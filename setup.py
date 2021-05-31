from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

# requiries = ['pyfesom2',
# 'xarray',
# 'numpy',
# 'scipy',
# 'matplotlib',
# # 'cartopy',
# 'os',
# 'shapely',
# 'great_cricle_calculator',
# 'tqdm',
# 'warnings',
# 'dask',
# 'glob'
# ]

setup(name='python_utils',
version='0.0.1',
description='Collection of various python functions',
long_description=readme(),
long_description_content_type='text/markdown',
author='FinnHeukamp',
author_email='Finn.Heukamp@awi.de',
license='AWI',
keywords='core package',
packages=['python_utils'],
# install_requires=requiries,
include_package_data=True,
zip_safe=False

)
