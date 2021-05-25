from setuptools import setup

from python.nway import version

setup(
    name="nway",
    version=version.get_git_version(),
    author="",
    author_email="",
    url = "https://github.com/KIPAC/NWayMatch",
    package_dir={"":"python"},
    packages=["nway"],
    description="=Matching algorithm for multiple input source catalogs",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=[],
)
