from setuptools import setup, find_packages

setup(
    name="IBM_functions_package",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,  # This enables MANIFEST.in processing
)
