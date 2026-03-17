from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'trajectory_generator_new'

# Collect data files recursively

def _recursive_glob(path):
    return [p for p in glob(os.path.join(path, '**'), recursive=True) if os.path.isfile(p)]

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    data_files=[
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', _recursive_glob('config')),
        ('share/' + package_name + '/maps', _recursive_glob('maps')),
        ('share/' + package_name + '/inputs', _recursive_glob('inputs')),
        ('share/' + package_name + '/outputs', _recursive_glob('outputs')),
        ('share/' + package_name + '/params', _recursive_glob('params')),
        ('share/' + package_name + '/scripts', _recursive_glob('scripts')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='jin',
    maintainer_email='jin@example.com',
    description='Trajectory generator pipeline (refactored)',
    license='MIT',
)
