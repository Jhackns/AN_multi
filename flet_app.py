import os
import time
from typing import List, Dict

import flet as ft

from lp_core import build_constraints, solve_lp, evaluate_vertices, plot_feasible, format_rows_for_table


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def default_example(mode: str):
    if mode == "max":
        return {
            "c1": 40.0,
            "c2": 60.0,
            "constraints": [
                {"a": 1.0, "b": 1.0, "c": 60.0},
                {"a": 1.0, "b": 3.0, "c": 120.0},
            ],
            "sense": "le",
        }
    else:
        return {
            "c1": 12.0,
            "c2": 8.0,
            "constraints": [
                {"a": 2.0, "b": 2.0, "c": 16.0},
                {"a": 4.0, "b": 1.0, "c": 20.0},
            ],
            "sense": "ge",
        }


def _constraint_row(index: int) -> ft.Row:
    return ft.Row(
        controls=[
            ft.Text(f"R{index+1}"),
            ft.TextField(label="a (X₁)", width=110, keyboard_type=ft.KeyboardType.NUMBER),
            ft.TextField(label="b (X₂)", width=110, keyboard_type=ft.KeyboardType.NUMBER),
            ft.TextField(label="c (lado derecho)", width=140, keyboard_type=ft.KeyboardType.NUMBER),
        ],
        alignment=ft.MainAxisAlignment.START,
        spacing=10,
    )


def build_tab(mode: str) -> ft.Tab:
    title = "Máximo" if mode == "max" else "Mínimo"
    sense = "le" if mode == "max" else "ge"
    subtitle = "Localización del punto óptimo"

    # Entradas principales
    c1_tf = ft.TextField(label="c1 (coeficiente de X₁)", hint_text="Ej.: 40" if mode == "max" else "Ej.: 12", width=220, keyboard_type=ft.KeyboardType.NUMBER)
    c2_tf = ft.TextField(label="c2 (coeficiente de X₂)", hint_text="Ej.: 60" if mode == "max" else "Ej.: 8", width=220, keyboard_type=ft.KeyboardType.NUMBER)

    num_dd = ft.Dropdown(
        label="Número de restricciones",
        width=220,
        options=[ft.dropdown.Option(str(n)) for n in [2, 3, 4, 5, 6]],
        value="2",
        on_change=lambda e: refresh_rest_rows(),
    )

    rest_container = ft.Column(spacing=6)
    # Inicializa las filas sin forzar update antes de que el control esté montado
    rest_container.controls = [_constraint_row(i) for i in range(int(num_dd.value))]

    def refresh_rest_rows():
        rest_container.controls = [_constraint_row(i) for i in range(int(num_dd.value))]
        # Solo actualiza si el control ya está añadido a la página
        if getattr(rest_container, "page", None) is not None:
            rest_container.update()

    # Área de resultados
    result_msg = ft.Text(value="", color=ft.Colors.ON_SURFACE)
    table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Vértice")),
            ft.DataColumn(ft.Text("Coordenadas")),
            ft.DataColumn(ft.Text("Valor")),
        ],
        rows=[],
        heading_row_color=ft.Colors.BLUE_GREY_50,
    )
    chart_image = ft.Image(src=f"plot_{mode}.png", width=600, height=420, fit=ft.ImageFit.CONTAIN)

    def fill_example(e=None):
        data = default_example(mode)
        c1_tf.value = str(data["c1"]) 
        c2_tf.value = str(data["c2"]) 
        num_dd.value = str(len(data["constraints"]))
        refresh_rest_rows()
        for i, con in enumerate(data["constraints"]):
            row: ft.Row = rest_container.controls[i]
            row.controls[1].value = str(con["a"]).rstrip(".0")
            row.controls[2].value = str(con["b"]).rstrip(".0")
            row.controls[3].value = str(con["c"]).rstrip(".0")
        c1_tf.update()
        c2_tf.update()
        num_dd.update()
        rest_container.update()
        # Ejecuta automáticamente el ejemplo para mostrar resultados y gráfico
        calculate()

    def calculate(e=None):
        try:
            # Sugerencias automáticas si falta info
            if not c1_tf.value or not c2_tf.value:
                fill_example()

            def _num(v):
                return float(str(v).replace(",", ".").strip())

            c1 = _num(c1_tf.value)
            c2 = _num(c2_tf.value)

            constraints: List[Dict[str, float]] = []
            if len(rest_container.controls) == 0:
                fill_example()
            invalid_rows = []
            for idx, row in enumerate(rest_container.controls, start=1):
                try:
                    a = _num(row.controls[1].value)
                    b = _num(row.controls[2].value)
                    c = _num(row.controls[3].value)
                    constraints.append({"a": a, "b": b, "c": c})
                except Exception:
                    invalid_rows.append(idx)

            if invalid_rows:
                result_msg.value = f"Revisa los campos numéricos en R{', R'.join(map(str, invalid_rows))}."
                result_msg.update()
                return

            A_ub, b_ub, textos = build_constraints(constraints, sense=sense)

            res = solve_lp(mode, c1, c2, A_ub, b_ub)
            if not res.success:
                result_msg.value = f"No se encontró solución: {res.message}"
                result_msg.update()
                return

            x1_opt, x2_opt = res.x
            opt_val = (-res.fun) if mode == "max" else res.fun

            best_point, best_value, rows = evaluate_vertices(mode, c1, c2, A_ub, b_ub)
            label = "MÁXIMO" if mode == "max" else "MÍNIMO"
            formatted = format_rows_for_table(rows, label)
            table.rows = [
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(r["vertex"])),
                    ft.DataCell(ft.Text(r["coords"])),
                    ft.DataCell(ft.Text(r["value"])),
                ]) for r in formatted
            ]
            table.update()

            # Gráfico
            # Usar nombre único para evitar caché del navegador
            img_name = f"plot_{mode}_{int(time.time()*1000)}.png"
            img_path = os.path.join(ASSETS_DIR, img_name)
            title = f"Región Factible y Punto Óptimo — {label}"
            plot_feasible(best_point, c1, c2, A_ub, b_ub, textos, title, img_path)
            chart_image.src = os.path.basename(img_path)
            chart_image.update()

            result_msg.value = (
                f"✓ Punto óptimo: ({best_point[0]:.2f}, {best_point[1]:.2f}) | "
                f"Z {label}: {best_value:.2f} — (algoritmo: X₁={x1_opt:.2f}, X₂={x2_opt:.2f}, Z={opt_val:.2f})"
            )
            result_msg.update()
        except Exception as ex:
            result_msg.value = f"Error en cálculo: {ex}"
            result_msg.update()

    # Ya se inicializan las filas en refresh_rest_rows() al construir la pestaña

    header = ft.Text(f"{title}", style=ft.TextThemeStyle.HEADLINE_MEDIUM)
    subtitle_text = ft.Text(subtitle, color=ft.Colors.BLUE_GREY)

    content = ft.Column(
        controls=[
            header,
            subtitle_text,
            ft.Divider(),
    ft.Row(controls=[c1_tf, c2_tf, num_dd, ft.ElevatedButton("Usar ejemplo", icon=ft.Icons.LIGHTBULB, on_click=fill_example)], wrap=False, spacing=12),
            ft.Container(content=rest_container, padding=ft.padding.only(left=0, right=0)),
    ft.Row(controls=[ft.ElevatedButton("Calcular", icon=ft.Icons.CALCULATE, on_click=calculate)], spacing=12),
            ft.Divider(),
            ft.Text("Resultados", weight=ft.FontWeight.BOLD),
            ft.Container(content=table, bgcolor=ft.Colors.WHITE, border_radius=8, padding=10),
            ft.Text("Gráfico", weight=ft.FontWeight.BOLD),
            chart_image,
            result_msg,
        ],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.START,
    )

    tab = ft.Tab(text=title, content=content)
    return tab


def main(page: ft.Page):
    page.title = "Optimización LP — Flet"
    page.theme_mode = ft.ThemeMode.LIGHT
    # Tema con color secundario azul moderno
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE)
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[build_tab("max"), build_tab("min")],
        expand=1,
    )
    # Diseño simple: sin degradado, con esquema de color azul
    page.add(tabs)


if __name__ == "__main__":
    # Ejecutar como web (navegador) para facilitar vista previa
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, host="127.0.0.1", port=8550, assets_dir=ASSETS_DIR)