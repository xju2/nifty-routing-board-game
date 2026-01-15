# Routing Board Game

## Installation
1. Install [uv](https://docs.astral.sh/uv/), an extremely fast Python package and project manager, written in Rust. See the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) details.
For a Linux or macOS system, you can run the following command in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

 2. Clone this repository and navigate to the `routing-board` directory, and then install the required dependencies using `uv`:
```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```

3. Launch the game locally by running:
```bash
python examples/pygame_gui.py
```
