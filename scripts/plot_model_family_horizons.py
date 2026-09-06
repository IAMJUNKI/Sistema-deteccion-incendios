#!/usr/bin/env python3
"""Alias con nombre descriptivo para el perfil de métricas por horizonte.

El módulo principal conserva el nombre histórico ``plot_model_family_boxplot``
para no romper comandos locales ya utilizados, aunque la figura actual es un
perfil de puntos y líneas, no un boxplot.
"""

from plot_model_family_boxplot import main


if __name__ == "__main__":
    main()
