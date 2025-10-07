import os
import math
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_constraints(constraints: List[Dict[str, float]], sense: str = "le") -> Tuple[List[List[float]], List[float], List[str]]:
    """
    Convierte una lista de restricciones en matrices A_ub y b_ub para linprog.
    - sense: "le" para <= (máximo) o "ge" para >= (mínimo, se multiplica por -1).
    Cada restricción es un dict con claves: {"a", "b", "c"}.
    """
    A_ub: List[List[float]] = []
    b_ub: List[float] = []
    textos: List[str] = []

    for con in constraints:
        a = float(con.get("a", 0))
        b = float(con.get("b", 0))
        c = float(con.get("c", 0))

        if sense == "ge":
            A_ub.append([-a, -b])
            b_ub.append(-c)
            textos.append(f"{a}*X₁ + {b}*X₂ ≥ {c}")
        else:
            A_ub.append([a, b])
            b_ub.append(c)
            textos.append(f"{a}*X₁ + {b}*X₂ ≤ {c}")

    return A_ub, b_ub, textos


def solve_lp(mode: str, c1: float, c2: float, A_ub: List[List[float]], b_ub: List[float]) -> Any:
    """
    Resuelve el PL con linprog (HiGHS). mode: "max" o "min".
    Devuelve el objeto resultado de scipy.optimize.linprog.
    """
    if mode == "max":
        c = [-c1, -c2]
    else:
        c = [c1, c2]

    x_bounds = [(0, None), (0, None)]
    return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method="highs")


def find_vertices(A_ub: List[List[float]], b_ub: List[float]) -> List[Tuple[float, float]]:
    """Encuentra vértices de la región factible considerando intersecciones y ejes."""
    vertices: List[Tuple[float, float]] = [(0.0, 0.0)]
    n = len(A_ub)

    # Intersecciones con ejes
    for i in range(n):
        a, b = A_ub[i]
        c = b_ub[i]
        if a != 0:
            x1 = c / a
            if x1 >= -1e-9:
                vertices.append((max(0.0, x1), 0.0))
        if b != 0:
            x2 = c / b
            if x2 >= -1e-9:
                vertices.append((0.0, max(0.0, x2)))

    # Intersecciones entre pares de restricciones
    for i in range(n):
        for j in range(i + 1, n):
            A_sys = np.array([A_ub[i], A_ub[j]], dtype=float)
            b_sys = np.array([b_ub[i], b_ub[j]], dtype=float)
            try:
                p = np.linalg.solve(A_sys, b_sys)
                x1, x2 = float(p[0]), float(p[1])
                if x1 >= -1e-9 and x2 >= -1e-9:
                    vertices.append((max(0.0, x1), max(0.0, x2)))
            except np.linalg.LinAlgError:
                continue

    # Eliminar duplicados con tolerancia
    unique: List[Tuple[float, float]] = []
    for v in vertices:
        if not any(abs(v[0] - u[0]) < 1e-6 and abs(v[1] - u[1]) < 1e-6 for u in unique):
            unique.append(v)
    return unique


def evaluate_vertices(mode: str, c1: float, c2: float, A_ub: List[List[float]], b_ub: List[float]):
    """
    Evalúa los vértices que cumplen todas las restricciones y localiza el óptimo.
    Devuelve: (best_point, best_value, rows)
    donde rows = [(letra, (x1,x2), valor)] para la tabla comparativa.
    """
    vertices = find_vertices(A_ub, b_ub)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rows = []

    if mode == "max":
        best_value = -float("inf")
    else:
        best_value = float("inf")
    best_point = None

    # Filtrar factibles y evaluar
    idx = 0
    for x1, x2 in vertices:
        factible = True
        for i in range(len(A_ub)):
            if A_ub[i][0] * x1 + A_ub[i][1] * x2 > b_ub[i] + 1e-9:
                factible = False
                break
        if not factible:
            continue
        val = c1 * x1 + c2 * x2
        rows.append((letters[idx] if idx < len(letters) else f"V{idx+1}", (x1, x2), val))
        idx += 1
        if (mode == "max" and val > best_value) or (mode == "min" and val < best_value):
            best_value = val
            best_point = (x1, x2)

    return best_point, best_value, rows

def _feasible_points(A_ub: List[List[float]], b_ub: List[float]) -> List[Tuple[float, float]]:
    """Devuelve los puntos factibles (vértices) de la región, incluyendo (0,0) si corresponde."""
    candidates = find_vertices(A_ub, b_ub)
    points: List[Tuple[float, float]] = []
    for x1, x2 in candidates:
        factible = True
        for i in range(len(A_ub)):
            if A_ub[i][0] * x1 + A_ub[i][1] * x2 > b_ub[i] + 1e-9:
                factible = False
                break
        if factible:
            points.append((max(0.0, x1), max(0.0, x2)))

    # Eliminar duplicados con tolerancia
    unique: List[Tuple[float, float]] = []
    for v in points:
        if not any(abs(v[0] - u[0]) < 1e-6 and abs(v[1] - u[1]) < 1e-6 for u in unique):
            unique.append(v)
    return unique

def _sort_polygon(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ordena puntos factibles alrededor del centroide para formar un polígono cerrado."""
    if not points:
        return []
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def plot_feasible(best_point: Tuple[float, float], c1: float, c2: float,
                  A_ub: List[List[float]], b_ub: List[float], restricciones_texto: List[str],
                  title: str, out_path: str) -> str:
    """Genera un gráfico exacto de la región factible y marca el punto óptimo."""
    # Puntos factibles y polígono
    feas_pts = _feasible_points(A_ub, b_ub)
    if best_point is not None and not any(abs(best_point[0]-p[0])<1e-6 and abs(best_point[1]-p[1])<1e-6 for p in feas_pts):
        feas_pts.append(best_point)
    poly = _sort_polygon(feas_pts)

    # Rango dinámico robusto (considera interceptos en ejes y puntos factibles)
    x1_cands = [p[0] for p in feas_pts]
    x2_cands = [p[1] for p in feas_pts]
    for i in range(len(A_ub)):
        a, b = A_ub[i]
        c = b_ub[i]
        if abs(a) > 1e-12:
            v = c / a
            if v > 0:
                x1_cands.append(v)
        if abs(b) > 1e-12:
            v = c / b
            if v > 0:
                x2_cands.append(v)
    max_x1 = max([1.0] + x1_cands) * 1.25
    max_x2 = max([1.0] + x2_cands) * 1.25

    # Dibujo de restricciones
    x1 = np.linspace(0, max_x1, 600)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for i, (a, b) in enumerate(A_ub):
        color = colors[i % len(colors)]
        if abs(b) > 1e-12:
            x2_r = (b_ub[i] - a * x1) / b
            plt.plot(x1, x2_r, color=color, linewidth=2.0, label=restricciones_texto[i])
        else:
            # Línea vertical exacta, independientemente del signo de 'a'
            x1_lim = (b_ub[i] / a) if abs(a) > 1e-12 else max_x1
            plt.axvline(x=x1_lim, color=color, linewidth=2.0, label=restricciones_texto[i])

    # Región factible como polígono
    if len(poly) >= 3:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        plt.fill(xs, ys, color="#7ac77a", alpha=0.35, label="Región factible")

    # Isocosto/isoganancia pasando por el punto óptimo
    z = c1 * best_point[0] + c2 * best_point[1]
    for factor in [0.6, 1.0, 1.4]:
        zf = z * factor
        if abs(c2) > 1e-12:
            x2_iso = (zf - c1 * x1) / c2
            plt.plot(x1, x2_iso, "--", color="#888", linewidth=1.0)

    # Punto óptimo
    plt.plot(best_point[0], best_point[1], "o", color="#2ca02c", markersize=10,
             label=f"Punto óptimo ({best_point[0]:.2f}, {best_point[1]:.2f})")

    # Estética
    plt.xlim(0, max_x1)
    plt.ylim(0, max_x2)
    plt.xlabel("X₁")
    plt.ylabel("X₂")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    return out_path


def format_rows_for_table(rows: List[Tuple[str, Tuple[float, float], float]], label: str) -> List[Dict[str, Any]]:
    """Convierte filas de resultados en estructuras amigables para UI."""
    formatted = []
    # calcular óptimo
    opt_val = None
    if rows:
        values = [r[2] for r in rows]
        opt_val = max(values) if label == "MÁXIMO" else min(values)
    for letter, (x1, x2), val in rows:
        mark = " ← " + label if (opt_val is not None and abs(val - opt_val) < 1e-6) else ""
        formatted.append({
            "vertex": letter,
            "coords": f"({x1:.2f}, {x2:.2f})",
            "value": f"{val:.2f}{mark}",
        })
    return formatted