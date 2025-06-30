from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="skimgpt",
    version="0.1.8",
    description="Biomedical Knowledge Mining with co-occurrence modeling and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jack Freeman, Rob Millikin, Kevin George, Ron Stewart",
    author_email="jfreeman@morgridge.org",
    url="https://github.com/stewart-lab/skimgpt",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'skimgpt.src': ['*.py', '*.sh'],
        'skimgpt': ['*.json', '*.txt', '*.md'],
    },
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "biopython>=1.79",
        "requests>=2.25.0",
        "vllm>=0.6.0",
        "tiktoken>=0.7.0",
        "htcondor>=24.6.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "build>=0.7.0",
            "twine>=3.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="biomedical nlp co-occurrence knowledge-mining LLMs literature-analysis skim",
    entry_points={
        'console_scripts': [
            'skimgpt-relevance=skimgpt.src.relevance:main',
            'skimgpt-main=skimgpt.main:main',
        ],
    },
) 