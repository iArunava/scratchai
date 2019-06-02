import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="scratchai-nightly",
    version="0.0.1a2",
    author="@iArunava",
    author_email="iarunavaofficial@gmail.com",
    description="Scratch AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iArunava/scratch.ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
