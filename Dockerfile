FROM python:3.11-bullseye as poetry-base

ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VERSION=1.4.2
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV USERNAME="python"
ENV APP_HOME="/home/$USERNAME"

ARG uid=1000
ARG gid=1000

ARG CI_JOB_TOKEN

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && python -m pip install --upgrade pip \
    && groupadd -g "$gid" "$USERNAME" \
    && useradd -u "$uid" -g "$gid" "$USERNAME" \
    && mkhomedir_helper "$USERNAME"

ENV PATH="$APP_HOME/.local/bin:$PATH"

FROM poetry-base

USER "$USERNAME"

WORKDIR "$APP_HOME"/code

RUN curl -sSL https://install.python-poetry.org | python3

COPY --chown="$USERNAME" pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project 1 \
    && poetry install --no-root --no-interaction --with api

COPY --chown="$USERNAME" . .

ENV PYTHONPATH .

CMD poetry run python -m apps.assistant.http.app:create_app
