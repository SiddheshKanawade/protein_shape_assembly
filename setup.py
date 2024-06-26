from setuptools import find_packages, setup

requirements = [
    "numpy",
    "pyyaml",
    "trimesh",
    "torch",
    "pytorch_lightning",
    "pytorch3d",
    "tqdm",
    "yacs",
    "einops",
    "matplotlib",
    "isort==5.9.3",
    "black==22.3.0",
    "autoflake==1.4",
    "scipy==1.7.3",
    "pyuul"
]

__version__ = "0.0.1"

setup(
    name="multi_part_assembly",
    version=__version__,
    description="Code for Learning 3D Geometric Shape Assembly",
    long_description="Code for Learning 3D Geometric Shape Assembly",
    author="Ziyi Wu",
    author_email="ziyiwu@cs.toronto.edu",
    license="",
    url="",
    keywords="multi part shape assembly",
    packages=find_packages(),
    install_requires=requirements,
)
