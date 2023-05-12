from setuptools import setup, find_packages

requirements = [
    'numpy',
    'matplotlib',
    'opencv-python'
]
setup(
    name='general_hough_transform',
    description="Parallel Object Detector implementation of General Hough Transform algorithm",
    version='0.1',
    packages=find_packages(),
    install_requires=requirements
)
