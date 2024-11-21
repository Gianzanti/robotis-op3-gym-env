# Installation process

## Install uv package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Clone this hub repository

```bash
git clone https://github.com/Gianzanti/robotis-op3-gym-env
```

<!-- ## Exec uv sync to install all dependencies

```bash
uv sync
``` -->

## Create the package

```bash
uv build
```

## Install it as a package

```bash
pip install dist/robotis_op3-0.1.1-py3-none-any.whl
```

## Install it in editable mode

```bash
pip install -e .
```
