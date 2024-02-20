import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name='facvae',
    version='1.2',
    author='Ali Siahkoohi and Rudy Morel',
    author_email='alisk@rice.edu',
    description='fVAEs for unsupervised source separation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alisiahkoohi/facvae',
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages(),
)
