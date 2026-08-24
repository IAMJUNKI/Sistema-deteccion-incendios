"""Ejecuta secuencialmente la suite corregida sin acumular memoria."""

import subprocess
import sys


def main() -> None:
    """Lanza réplicas aisladas de selección de variables."""
    for remainder in (7, 13, 19):
        subprocess.run(
            [sys.executable, "-m", "src.modeling.run_ablation", "--remainder", str(remainder), "--output-dir", "outputs/modeling/corrected"],
            check=True,
        )


if __name__ == "__main__":
    main()
