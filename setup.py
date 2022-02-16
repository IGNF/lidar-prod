from setuptools import find_packages, setup

setup(
    name="lidar_prod",
    version="0.0.0",
    description="A 3D semantic segmentation production tool to augment rules-based Lidar classification with AI and databases.",
    author="Charles GAYDON",
    author_email="charles.gaydon@gmail.com",
    # replace with your own github project link
    url="https://github.com/IGNF/lidar-prod-quality-control",
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6"],
    packages=find_packages(),
)
