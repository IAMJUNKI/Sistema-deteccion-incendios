"""Convierte un `.py` en formato *percent* a un cuaderno `.ipynb`.

El formato *percent* es el estándar de jupytext: `# %%` abre una celda de código y
`# %% [markdown]` una de texto, cuyo contenido va comentado con `#`. Escribir el cuaderno como
`.py` tiene dos ventajas prácticas: se revisa en un diff legible —un `.ipynb` es JSON con las
salidas incrustadas y los diffs son ilegibles— y se edita con cualquier herramienta.

Se implementa aquí en vez de depender de `jupytext` porque son cuarenta líneas y evita añadir
una dependencia al entorno del equipo por una conversión puntual.

Uso:
    python scripts/py_a_notebook.py notebooks/19_descubrimiento_unificado.py
    python scripts/py_a_notebook.py entrada.py --salida otro_nombre.ipynb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def separar_celdas(texto: str) -> list[tuple[str, str]]:
    """Parte el fichero en celdas. Devuelve pares (tipo, contenido)."""
    celdas: list[tuple[str, list[str]]] = []
    tipo = "code"
    actual: list[str] = []

    for linea in texto.splitlines():
        despojada = linea.strip()
        if despojada.startswith("# %%"):
            if actual:
                celdas.append((tipo, actual))
            tipo = "markdown" if "[markdown]" in despojada else "code"
            actual = []
            continue
        actual.append(linea)
    if actual:
        celdas.append((tipo, actual))

    salida: list[tuple[str, str]] = []
    for tipo_celda, lineas in celdas:
        if tipo_celda == "markdown":
            # El markdown viene comentado: se retira el `# ` inicial de cada línea.
            contenido = "\n".join(
                l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l)
                for l in lineas
            )
        else:
            contenido = "\n".join(lineas)
        contenido = contenido.strip("\n")
        if contenido.strip():
            salida.append((tipo_celda, contenido))
    return salida


def construir(celdas: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(c) if t == "markdown" else nbformat.v4.new_code_cell(c)
        for t, c in celdas
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrada", help="Fichero .py en formato percent")
    parser.add_argument("--salida", help="Destino .ipynb (por defecto, mismo nombre)")
    args = parser.parse_args()

    entrada = Path(args.entrada)
    if not entrada.exists():
        print(f"No existe: {entrada}", file=sys.stderr)
        return 1

    celdas = separar_celdas(entrada.read_text(encoding="utf-8"))
    salida = Path(args.salida) if args.salida else entrada.with_suffix(".ipynb")
    nbformat.write(construir(celdas), salida)

    codigo = sum(1 for t, _ in celdas if t == "code")
    print(f"escrito : {salida}")
    print(f"celdas  : {len(celdas)}  ({codigo} de código, {len(celdas) - codigo} de texto)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
