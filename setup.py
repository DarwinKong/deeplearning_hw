"""
RL Solitaire - 强化学习孔明棋求解器
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-solitaire",
    version="1.0.0",
    author="Karl Hajjar and Team",
    author_email="your-email@example.com",
    description="Solving Peg Solitaire with Reinforcement Learning (A2C, PPO)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/rl-solitaire",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Puzzle Games",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rl-train=run:main",
            "rl-play=play:main",
        ],
    },
)
