from setuptools import find_packages, setup

package_name = 'kc_task3b_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/task3b.launch.py']),
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
            'task3b_perception = kc_task3b_pkg.task3b_perception:main',
            'ebot_nav_task3b = kc_task3b_pkg.ebot_nav_task3b:main',
            'task3b_manipulation = kc_task3b_pkg.task3b_manipulation:main',
            'shape_detector_task3b = kc_task3b_pkg.shape_detector_task3b:main',
        ],
    },
)
