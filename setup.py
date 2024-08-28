from setuptools import find_packages, setup

setup(
    name="TraderBot",
    version="0.1.0",
    author="Daniel Sinkin",
    author_email="danielsinkin97@gmail.com",
    description="A bot for trading with different currency pairs. Intended as a Portfolio Project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Daniel-Sinkin/Tradebot",
    packages=find_packages(
        include=["TraderBot", "TraderBot.*"]
    ),  # Include only TraderBot package
    include_package_data=True,
    package_data={
        # Include any .so files in the TraderBot package
        "TraderBot": ["*.so"],
    },
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=3.8.0",
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
)
