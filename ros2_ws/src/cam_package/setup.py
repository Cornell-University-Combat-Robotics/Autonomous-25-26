from setuptools import find_packages, setup

package_name = 'cam_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryans',
    maintainer_email='ryans@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'cam_node',
            'cam_listener',
        ],
    },
    entry_points={
        'console_scripts': [
            'cam_node = cam_package.cam_node:main',
            'cam_listener = cam_package.cam_listener:main',
        ],
    },
)
