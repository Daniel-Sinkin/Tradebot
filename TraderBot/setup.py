from setuptools import find_packages, setup

setup(
    name="TraderBot",
    version="0.1.0",
    author="Daniel Sinkin",
    author_email="danielsinkin97@gmail.com",  # replace with your actual email
    description="A bot for trading with different currency pairs. Intended as a Portfolio Project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Daniel-Sinkin/Tradebot",  # replace with your actual repository URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.2.0",
        # add other dependencies your project requires
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=3.8.0",
            # add other development dependencies
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "traderbot=TraderBot.main:main",  # replace `main` with the actual callable
        ],
    },
)
