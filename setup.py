from setuptools import setup

setup(
    name="surrender",
    version="0.1.0",
    description="A basic software rasterizer",
    url="https://github.com/milmillin/surrender",
    author="Milin Kodnongbua",
    author_email="milink@cs.washington.edu",
    license="MIT",
    packages=["surrender"],
    install_requires=["numpy", "pyembree"],
)