# Installation

## Build from source

Dependencies needed to build and run Ragbits from the source code:

1. [**uv**](https://docs.astral.sh/uv/getting-started/installation/)
2. [**python**](https://docs.astral.sh/uv/guides/install-python/) 3.10


## Linting and formatting
We use `ruff` for linting and formatting our code. To format your code, run:

```bash
$ uv run ruff format
```

To lint the code, run:
```bash
$ uv run ruff check --fix
```

## Type checking
We use `mypy` for type checking. To perform type checking, simply run:

```bash
$ uv run mypy .
```

## Testing
We use `pytest` for testing. To run the tests, simply run:

```bash
$ uv run pytest
```
