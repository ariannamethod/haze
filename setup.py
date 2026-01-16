from setuptools import setup, find_packages

setup(
    name="haze-ai",
    version="0.1.0",
    description="Hybrid attention entropy system — post-transformer LM",
    author="Arianna Method",
    url="https://github.com/ariannamethod/haze",
    packages=find_packages(),
    install_requires=["numpy>=1.20.0", "sentencepiece>=0.1.96"],
    extras_require={
        "gradio": ["gradio>=4.0.0"],
        "dev": ["pytest>=7.0.0"],
    },
    python_requires=">=3.8",
    license="GPL-3.0",
    keywords="language-models transformer attention nlp resonance",
)
