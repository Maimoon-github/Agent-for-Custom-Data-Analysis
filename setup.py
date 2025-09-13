"""
Setup script for the RAG Agent package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="rag-agent",
    version="1.0.0",
    description="Privacy-focused local RAG agent for custom data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RAG Agent Development Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-agent=rag_agent.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="rag, retrieval-augmented-generation, llm, ai, machine-learning, nlp",
    project_urls={
        "Bug Reports": "https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/issues",
        "Source": "https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis",
    },
)