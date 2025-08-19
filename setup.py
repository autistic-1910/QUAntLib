from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantlib",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive quantitative analysis library for financial markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantlib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
        "ml": [
            "scikit-learn>=1.2",
            "xgboost>=1.7",
            "tensorflow>=2.12",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
        ],
    },
)