import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name='facvae',
    version='0.1',
    author='Rudy Morel and Ali Siahkoohi',
    author_email='alisk@rice.edu',
    description='Factorial VAEs for blind source separation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alisiahkoohi/factorialVAE',
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages()
)
