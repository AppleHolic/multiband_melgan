from setuptools import setup, find_packages


def get_requirements(filename='requirements.txt'):
    deps = []
    with open(filename, 'r') as f:
        for pkg in f.readlines():
            if pkg.strip():
                deps.append(pkg)
    return deps


setup(
    name="multiband_melgan",
    version="0.0.0",
    description="multi-band melgan",
    author="ILJI CHOI",
    author_email="choiilji@gmail.com",
    install_requires=get_requirements(),
    packages=find_packages(),
    include_package_data=True
)