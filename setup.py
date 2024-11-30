from setuptools import setup, find_packages

setup(
    name="DamidoumPyMLProto",
    version="0.1.0",
    author="Damidoum",
    author_email="damien.rouchouse@etu.minesparis.psl.eu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy>=2.1.3"],
)
