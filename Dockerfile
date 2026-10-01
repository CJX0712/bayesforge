# BayesForge — author: 晨星
FROM python:3.12-slim

LABEL maintainer="晨星"
LABEL description="Deterministic Bayesian optimization with an exact NumPy GP"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY bayesforge ./bayesforge

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && python -c "import bayesforge; print('bayesforge ok', bayesforge.__version__)"

COPY tests ./tests
COPY pytest.ini ./
COPY examples ./examples
COPY docs ./docs

RUN python -m pip install pytest \
    && python -m pytest -q \
    && python -m pip uninstall -y pytest

ENTRYPOINT ["python", "-m", "bayesforge.cli"]
CMD ["check", "--n-evals", "16"]
