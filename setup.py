from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brain-tumor-segmentation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Brain Tumor Segmentation using MONAI and PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Brain-Tumor-Segmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "brain-seg-train=src.train:main",
            "brain-seg-predict=src.inference:main",
        ],
    },
)