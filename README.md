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

## Installing environment

### As a package

```bash
pip install dist/robotis_op3-[version]-py3-none-any.whl
```

### In editable mode

```bash
pip install -e .
```
