from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='macaqueModules',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Short description",
    license="MIT",
    author="PB",
    author_email='philipe.bujold@gmail.com',
    url='https://github.com/pbujold/macaqueModules',
    packages=['macaque'],
    entry_points={
        'console_scripts': [
            'macaque=macaque.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='macaqueModules',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
