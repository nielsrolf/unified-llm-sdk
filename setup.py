from setuptools import find_packages, setup

setup(
    name="any_llm",
    version=0.1,
    description="Description here",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        "openai",
        "anthropic",
        "google-generativeai",
        "backoff",
        "diskcache",
        "python-dotenv",
    ],
    extras_require={
        "test": ["pytest", "pylint!=2.5.0", "black", "mypy", "flake8", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [],
    },
    classifiers=[],
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    keywords="",
)
