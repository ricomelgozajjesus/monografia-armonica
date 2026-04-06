
# %%
# # Transformadas rápidas de Walsh y Hartley
# ## Un primer laboratorio comparativo frente a Fourier
#
# Este script está organizado con celdas estilo Jupyter/Colab.
# Puede ejecutarse como archivo `.py` en VS Code, JupyterLab o convertirse
# después a notebook si así se desea.
#
# Objetivos:
# 1. Implementar una versión rápida de la transformada de Walsh–Hadamard.
# 2. Implementar la transformada discreta de Hartley a partir de la FFT.
# 3. Comparar estas representaciones con la FFT sobre señales sencillas.
# 4. Mostrar que Fourier forma parte de una familia más amplia de
#    reorganizaciones ortogonales.

# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import hadamard
from numpy.fft import fft

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11


# %%
# ## Utilidades básicas

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def truncate_largest(coeffs: np.ndarray, m: int) -> np.ndarray:
    """
    Conserva solo los m coeficientes de mayor magnitud.
    """
    coeffs = np.array(coeffs, copy=True)
    idx = np.argsort(np.abs(coeffs))[::-1]
    keep = idx[:m]
    out = np.zeros_like(coeffs)
    out[keep] = coeffs[keep]
    return out


def stem_plot(ax, y, title, xlabel="Índice", ylabel="Valor"):
    ax.stem(np.arange(len(y)), y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# %%
# ## 1. Transformada de Walsh–Hadamard
#
# La transformada de Walsh–Hadamard reemplaza las oscilaciones sinusoidales
# por patrones ortogonales rectangulares, formados por valores +1 y -1.
#
# Su interés aparece cuando la señal tiene estructura por bloques o cambios
# abruptos. En tales casos, una base rectangular puede resultar más natural
# que una base sinusoidal.


def walsh_hadamard_matrix(n: int, normalize: bool = True) -> np.ndarray:
    """
    Devuelve la matriz de Hadamard de orden n.
    n debe ser potencia de 2.
    """
    if not is_power_of_two(n):
        raise ValueError("n debe ser potencia de 2.")
    H = hadamard(n).astype(float)
    if normalize:
        H = H / np.sqrt(n)
    return H


def wht_matrix(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Transformada de Walsh-Hadamard por multiplicación matricial.
    """
    x = np.asarray(x, dtype=float)
    H = walsh_hadamard_matrix(len(x), normalize=normalize)
    return H @ x


def fwht(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform.
    La longitud de x debe ser potencia de 2.
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)

    if not is_power_of_two(n):
        raise ValueError("La longitud de la señal debe ser potencia de 2.")

    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2

    if normalize:
        x = x / np.sqrt(n)

    return x


def ifwht(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    La inversa coincide con la directa si usamos normalización ortonormal.
    """
    return fwht(X, normalize=normalize)


# %%
# ### Verificación matricial vs rápida

x = np.array([1, 2, 3, 4, 0, 1, 0, -1], dtype=float)

X_mat = wht_matrix(x, normalize=True)
X_fast = fwht(x, normalize=True)
x_rec = ifwht(X_fast, normalize=True)

print("Error entre forma matricial y rápida:", np.linalg.norm(X_mat - X_fast))
print("Error de reconstrucción:", np.linalg.norm(x - x_rec))


# %%
# ### Visualización básica

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

stem_plot(ax[0], x, "Señal original", ylabel="x[n]")
stem_plot(ax[1], X_fast, "Coeficientes Walsh–Hadamard", ylabel="W[k]")

plt.tight_layout()
plt.show()


# %%
# ## 2. Transformada de Hartley
#
# La transformada de Hartley conserva una cercanía conceptual con Fourier,
# pero evita números complejos en su forma discreta.
#
# Mientras Fourier usa el núcleo exponencial complejo, Hartley combina
# seno y coseno en una sola función real:
#
#     cas(x) = cos(x) + sin(x)
#
# En la práctica, una forma muy cómoda de implementarla consiste en
# aprovechar su relación con la FFT.


def dht(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Discrete Hartley Transform.

    Implementación a partir de la FFT:
        H[k] = Re(FFT(x))[k] - Im(FFT(x))[k]
    """
    x = np.asarray(x, dtype=float)
    X = fft(x)
    H = X.real - X.imag

    if normalize:
        H = H / np.sqrt(len(x))

    return H


def idht(H: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    La DHT es involutiva salvo factor de escala.
    """
    H = np.asarray(H, dtype=float)
    n = len(H)
    x = dht(H, normalize=normalize)

    if normalize:
        return x
    return x / n


# %%
# ### Verificación básica

x = np.array([1, 2, 3, 4, 0, 1, 0, -1], dtype=float)

H = dht(x, normalize=False)
x_rec = idht(H, normalize=False)

print("Error de reconstrucción DHT:", np.linalg.norm(x - x_rec))


# %%
# ## 3. Comparación sobre señal oscilatoria suave

n = 64
t = np.linspace(0, 1, n, endpoint=False)

x_smooth = (
    1.2 * np.cos(2 * np.pi * 5 * t)
    + 0.5 * np.sin(2 * np.pi * 11 * t)
    + 0.2 * np.cos(2 * np.pi * 17 * t)
)

X_fft = fft(x_smooth)
X_walsh = fwht(x_smooth, normalize=True)
H_dht = dht(x_smooth, normalize=True)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))

ax[0].plot(t, x_smooth)
ax[0].set_title("Señal suave")
ax[0].set_xlabel("t")

stem_plot(ax[1], np.abs(X_fft), "Magnitud FFT", ylabel="|F[k]|")
stem_plot(ax[2], X_walsh, "Walsh–Hadamard", ylabel="W[k]")
stem_plot(ax[3], H_dht, "Hartley", ylabel="H[k]")

plt.tight_layout()
plt.show()


# %%
# ## 4. Comparación sobre señal por bloques

n = 64
x_block = np.zeros(n)
x_block[:16] = 1.0
x_block[16:32] = -1.0
x_block[32:48] = 2.0
x_block[48:] = 0.5

X_fft = fft(x_block)
X_walsh = fwht(x_block, normalize=True)
H_dht = dht(x_block, normalize=True)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))

ax[0].plot(x_block)
ax[0].set_title("Señal por bloques")
ax[0].set_xlabel("n")

stem_plot(ax[1], np.abs(X_fft), "Magnitud FFT", ylabel="|F[k]|")
stem_plot(ax[2], X_walsh, "Walsh–Hadamard", ylabel="W[k]")
stem_plot(ax[3], H_dht, "Hartley", ylabel="H[k]")

plt.tight_layout()
plt.show()


# %%
# ## 5. Reconstrucción truncada

n = 64
t = np.linspace(0, 1, n, endpoint=False)

x = (
    np.cos(2 * np.pi * 5 * t)
    + 0.5 * np.sin(2 * np.pi * 9 * t)
    + 0.2 * np.cos(2 * np.pi * 13 * t)
)

# Walsh
W = fwht(x, normalize=True)
W_trunc = truncate_largest(W, 8)
x_w_rec = ifwht(W_trunc, normalize=True)

# Hartley
H = dht(x, normalize=True)
H_trunc = truncate_largest(H, 8)
x_h_rec = idht(H_trunc, normalize=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(t, x, label="Original")
ax[0].plot(t, x_w_rec, "--", label="Reconstrucción Walsh")
ax[0].set_title("Reconstrucción truncada en Walsh")
ax[0].legend()

ax[1].plot(t, x, label="Original")
ax[1].plot(t, x_h_rec, "--", label="Reconstrucción Hartley")
ax[1].set_title("Reconstrucción truncada en Hartley")
ax[1].legend()

plt.tight_layout()
plt.show()


# %%
# ## 6. Lectura comparativa
#
# Los experimentos anteriores sugieren una idea importante:
#
# - cuando la señal es suave y oscilatoria, Fourier y Hartley suelen ofrecer
#   representaciones más naturales;
# - cuando la señal tiene estructura rectangular o por bloques,
#   Walsh–Hadamard puede capturar mejor su organización;
# - la elección de una base no es neutral: depende de la geometría de la
#   señal y del tipo de estructura que se desea revelar.
#
# En este sentido, Walsh, Hartley y Fourier pueden verse como miembros
# de una familia más amplia de reorganizaciones ortogonales.


# %%
# ## Conclusión
#
# Este script no pretende reemplazar el papel central de Fourier, sino
# mostrar que existen otras bases ortogonales útiles para análisis y computación.
#
# - Walsh–Hadamard enfatiza patrones rectangulares y estructura discreta.
# - Hartley ofrece una lectura espectral real, muy cercana a Fourier.
# - Fourier sigue siendo la referencia natural para oscilaciones suaves y periódicas.
#
# La lección más general es esta:
# la representación espectral depende de la base elegida, y la base elegida
# debe dialogar con la forma del fenómeno.
