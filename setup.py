from setuptools import setup, find_packages

setup(
    name="codagent",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai",
        "tqdm",
        "colorama",
        "prompt_toolkit",
        "openai",
    ],
    entry_points={
        "console_scripts": [
            "coda=codagent.cli:main",
        ],
    },
    author="JoseIcaro",
    description="A CLI tool for code generation and editing using Google's Gemini models",
    keywords="ai, code, generation, gemini, google",
    python_requires=">=3.7",
)
