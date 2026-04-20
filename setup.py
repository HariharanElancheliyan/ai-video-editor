from setuptools import setup, find_packages

setup(
    name="ai-video-editor",
    version="0.1.0",
    description="AI-powered video editing tool using Ollama and FFMPEG",
    packages=find_packages(),
    install_requires=[
        "ollama>=0.3.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "typer>=0.15.0",
        "rich>=13.0.0",
        "static-ffmpeg>=2.5",
    ],
    extras_require={
        "audio": ["openai-whisper>=20231117"],
        "all": ["openai-whisper>=20231117"],
        "dev": ["pytest", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "ai-video-editor=ai_video_editor.__main__:app"
        ]
    },
    python_requires=">=3.10"
)