from .linear_sist_methods import (
    eliminacion_gaussiana,
    descomposicion_LU,
    resolver_LU,
    matriz_aumentada,
    separar_m_aumentada,
    gauss_jordan,
)
from .min_cuadrados import ajustar_min_cuadrados  # type: ignore

from .iterative_methods import gauss_jacobi, gauss_seidel  # type: ignore
