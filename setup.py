"""
Package setup for 3D Reconstruction Pipeline.
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_path):
        return []
    with open(req_path) as f:
        lines = f.read().splitlines()
    return [
        l.strip() for l in lines
        if l.strip() and not l.startswith("#")
    ]


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="reconstruction_3d",
    version="1.0.0",
    author="3D Reconstruction Pipeline",
    description="End-to-end 3D reconstruction: SfM + MVS + NeRF",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "run-sfm=scripts.run_sfm:main",
            "run-dense=scripts.run_dense:main",
            "train-nerf=scripts.train_nerf:main",
            "render-views=scripts.render_views:main",
            "serve-api=api.app:app",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)