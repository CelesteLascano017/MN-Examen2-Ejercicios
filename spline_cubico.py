import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def cubic_spline(xs: list[float], ys: list[float], m: float):
    """
    Cubic spline interpolation with imposed slope m at interior point x1.
    """

    # --- Ordenar puntos ---
    points = sorted(zip(xs, ys), key=lambda x: x[0])
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    n = len(xs) - 1
    h = [xs[i + 1] - xs[i] for i in range(n)]

    # --------------------------------------------------
    # PASO 1: Construcción del vector alpha
    # --------------------------------------------------
    alpha = [0] * (n + 1)

    for i in range(1, n):
        if i == 1:
            # ==================================================
            # 🔴 ALTERACIÓN CLAVE
            # En lugar de continuidad de derivadas:
            # S0'(x1) = S1'(x1)
            # imponemos:
            # S0'(x1) = m
            #
            # Esto modifica la ecuación del sistema
            # ==================================================
            alpha[i] = 3 * ((ys[i + 1] - ys[i]) / h[i] - m)
        else:
            alpha[i] = (
                3 / h[i] * (ys[i + 1] - ys[i])
                - 3 / h[i - 1] * (ys[i] - ys[i - 1])
            )

    # --------------------------------------------------
    # PASO 2: Algoritmo de Thomas (sistema tridiagonal)
    # --------------------------------------------------
    l = [1]
    u = [0]
    z = [0]

    for i in range(1, n):
        l_i = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * u[i - 1]
        l.append(l_i)
        u.append(h[i] / l_i)
        z.append((alpha[i] - h[i - 1] * z[i - 1]) / l_i)

    l.append(1)
    z.append(0)

    # --------------------------------------------------
    # PASO 3: Cálculo de coeficientes
    # --------------------------------------------------
    c = [0] * (n + 1)
    b = [0] * n
    d = [0] * n
    a = ys[:-1]

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - u[j] * c[j + 1]
        b[j] = (
            (ys[j + 1] - ys[j]) / h[j]
            - h[j] * (c[j + 1] + 2 * c[j]) / 3
        )
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # --------------------------------------------------
    # PASO 4: Construcción simbólica de los splines
    # --------------------------------------------------
    x = sym.Symbol("x")
    splines = []

    for j in range(n):
        S = (
            a[j]
            + b[j] * (x - xs[j])
            + c[j] * (x - xs[j]) ** 2
            + d[j] * (x - xs[j]) ** 3
        )
        splines.append(S)

    return splines

xs = [-1, 0, 1]
ys = [1, 5, 3]
m = -3

splines = cubic_spline(xs, ys, m)

print(splines)



x = sym.Symbol("x")
S0 = 3.75*x + 0.25*(x+1)**3 + 4.75
S1 = -0.25*x**3 + 0.75*x**2 - 2.5*x + 5

print("S0'(0) =", sym.diff(S0, x).subs(x, 0))
print("S1'(0) =", sym.diff(S1, x).subs(x, 0))

# Definimos los splines como funciones numéricas
def S0(x):
    return 3.75*x + 0.25*(x + 1)**3 + 4.75

def S1(x):
    return -0.25*x**3 + 0.75*x**2 - 2.5*x + 5

# Intervalos de cada spline
x0 = np.linspace(-1, 0, 200)
x1 = np.linspace(0, 1, 200)

y0 = S0(x0)
y1 = S1(x1)

# Puntos originales
xs = np.array([-1, 0, 1])
ys = np.array([1, 5, 3])

# Recta con pendiente deseada m = -3 en x1 = 0
m = -3
x_line = np.linspace(-0.5, 0.5, 100)
y_line = m * (x_line - 0) + 5   # pasa por (0,5)

# Gráfica
plt.figure(figsize=(8, 5))

plt.plot(x0, y0, 'r', label=r"$S_0(x)$")
plt.plot(x1, y1, 'b', label=r"$S_1(x)$")

plt.scatter(xs, ys, color='black', zorder=5, label="Puntos dados")
plt.plot(x_line, y_line, 'k--', label="Pendiente deseada m = -3")

plt.axvline(0, color='gray', linestyle=':', alpha=0.6)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.show()