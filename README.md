# Surrender

A basic software rasterizer. It just works when you have no GPU or are running on a headless server without display.

## Installation

```
pip install git+https://github.com/milmillin/surrender.git
```

## Usage

```python
from surrender import render

render(V, F, width, height)
```