from setuptools import find_packages, setup

package_name = 'kc_task2b_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu22_04',
    maintainer_email='ubuntu22_04@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'manipulation = kc_task2b_pkg.task2b_manipulation:main',
            'perception = kc_task2b_pkg.task2b_perception:main',
        ],
    },
)
