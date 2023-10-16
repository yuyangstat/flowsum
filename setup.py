from setuptools import setup, find_packages

setup(
    name="flowsum",
    version="1.0",
    packages=find_packages("."),
    package_dir={"": "."},
    install_requires=[
        "torch",
        "transformers == 4.25.1",
        "datasets == 2.9.0",
        "evaluate == 0.4.0",
        "tensorboardX == 2.5.1",
        "nltk == 3.8.1",
        "pyro-ppl == 1.8.4",
        "matplotlib == 3.5.3",
        "seaborn == 0.12.2",
        "numpy ~= 1.21.6",
        "pandas ~= 1.3.5",
        "aiohttp ~= 3.8.3",
        "fsspec == 2022.11.0",
        "absl-py",
        "rouge_score",
        "accelerate",
        "py7zr"
    ],
)