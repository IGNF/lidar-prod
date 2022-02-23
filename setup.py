from setuptools import find_packages, setup

setup(
    name="lidar_prod",
    version="0.1.0",
    description="A 3D semantic segmentation production tool to augment rules-based Lidar classification with AI and databases.",
    author="Charles GAYDON",
    author_email="charles.gaydon@gmail.com",
    # replace with your own github project link
    url="https://github.com/IGNF/lidar-prod-quality-control",
    install_requires=[],  # env should match the one in bash/setup_environment/setup_env.sh
    packages=find_packages(),
)
