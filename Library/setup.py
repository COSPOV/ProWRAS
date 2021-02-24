import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prowras",
    version="0.0.1",
    author="Saptarshi Bej",
    author_email="saptarshi.bej@uni-rostock.de",
    description="A shadow sample generating algorithm for imbalanced datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/COSPOV/ProWRAS.git",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
