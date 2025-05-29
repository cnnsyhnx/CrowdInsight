from setuptools import setup, find_packages
import os

# Read README.md with proper encoding
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="crowdinsight",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "torch>=1.7.0",
        "ultralytics>=8.0.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "pillow>=8.0.0"
    ],
    author="CrowdInsight Team",
    author_email="your.email@example.com",
    description="A Python AI library for crowd analysis from video streams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crowdinsight",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
