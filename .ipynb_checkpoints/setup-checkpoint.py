from setuptools import setup
import versioneer

requirements = [ 
# package requirements go here
				'scipy', 
				'matplotlib', 
				'seaborn', 
				'numpy', 
				'statsmodels',
				'pandas', 
				'tqdm', 
				'tabulate', 
				'r-base', 
				'rpy2', 
				'r-emmeans', 
				'dill',
				'tzlocal',
                'r-afex'        
]

setup(
    name='macaque',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="This is the psychometric analysis code for binary choice data I gathered in the Schultz Lab at Cambridge. The code incorporates both the required analysis package (the jupyter notebooks containing draft figures and results are in separate repos).",
    license="MIT",
    author="Philipe Bujold",
    author_email='philipe.bujold@gmail.com',
    url='https://github.com/pbujold/macaqueModules',
    packages=['macaque'],
    entry_points={
        'console_scripts': [
            'macaque=macaque.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='macaque',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ]
)
