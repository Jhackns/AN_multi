import os
from lp_core import build_constraints, solve_lp, evaluate_vertices, plot_feasible, format_rows_for_table


def ejemplo_maximo():
    return 40.0, 60.0, [
        {"a": 1.0, "b": 1.0, "c": 60.0},
        {"a": 1.0, "b": 3.0, "c": 120.0},
    ]


def main():
    print("=" * 50)
    print("OPTIMIZACIÓN DE PROGRAMACIÓN LINEAL — MÁXIMO")
    print("=" * 50)

    try:
        print("\n¿Desea usar el ejemplo predeterminado o ingresar sus propios datos?")
        print("1. Usar ejemplo (Z = 40*X₁ + 60*X₂)")
        print("2. Ingresar mis propios datos")
        opcion = input("Seleccione opción (1 o 2): ").strip()

        if opcion == "1":
            c1, c2, cons = ejemplo_maximo()
        else:
            # Ingreso manual con validación rápida
            c1 = float(input("c1: "))
            c2 = float(input("c2: "))
            n = int(input("Número de restricciones (2-4): "))
            cons = []
            for i in range(n):
                print(f"R{i+1} (formato a*X1 + b*X2 <= c)")
                a = float(input("  a: "))
                b = float(input("  b: "))
                c = float(input("  c: "))
                cons.append({"a": a, "b": b, "c": c})

        A_ub, b_ub, textos = build_constraints(cons, sense="le")
        res = solve_lp("max", c1, c2, A_ub, b_ub)
        if not res.success:
            print(f"\nNo se encontró solución: {res.message}")
            return

        x1_opt, x2_opt = res.x
        z_opt = -res.fun
        print(f"\n✓ Solución óptima (HiGHS): X₁={x1_opt:.2f}, X₂={x2_opt:.2f}, Z={z_opt:.2f}")

        best_point, best_value, rows = evaluate_vertices("max", c1, c2, A_ub, b_ub)
        if best_point:
            print("\n3. LOCALIZACIÓN DEL PUNTO ÓPTIMO (MÁXIMO)")
            formatted = format_rows_for_table(rows, "MÁXIMO")
            for r in formatted:
                print(f" {r['vertex']:>2}  {r['coords']:>10}  {r['value']}")

            # Gráfico
            assets = os.path.join(os.path.dirname(__file__), "assets")
            os.makedirs(assets, exist_ok=True)
            img_path = os.path.join(assets, "plot_cli_max.png")
            plot_feasible(best_point, c1, c2, A_ub, b_ub, textos, "Región Factible — MÁXIMO", img_path)
            print(f"\nImagen del gráfico guardada en: {img_path}")
        else:
            print("\nNo se encontraron puntos factibles")

    except Exception as ex:
        print(f"Error: {ex}")


if __name__ == "__main__":
    main()