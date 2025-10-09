import 'dart:math' as math;
import '../models/constraint.dart';
import '../models/lp_input.dart';

class LpSolver {
  static const double eps = 1e-9;

  static LpResult solve(LpInput input) {
    final constraints = input.constraints;
    // Añadimos x >= 0 y y >= 0 como restricciones ge (a,b,c)
    final all = [
      ...constraints,
      ConstraintModel(a: 1, b: 0, c: 0, sense: Sense.ge), // x >= 0
      ConstraintModel(a: 0, b: 1, c: 0, sense: Sense.ge), // y >= 0
    ];

    // Generar candidatos: intersecciones de pares de líneas ax+by=c
    final candidates = <List<double>>{}; // usar set por hash de string
    for (int i = 0; i < all.length; i++) {
      for (int j = i + 1; j < all.length; j++) {
        final p = _intersect(all[i], all[j]);
        if (p != null && p[0].isFinite && p[1].isFinite) {
          final key = _key(p);
          candidates.add([p[0], p[1]]);
        }
      }
    }

    // Intersecciones con ejes (x=0, y=0) ya están incluidas por ge, pero añadimos interceptos de cada restricción
    for (final c in all) {
      if (c.a.abs() > eps) {
        final xIntercept = c.c / c.a; // y=0
        if (xIntercept.isFinite) {
          candidates.add([xIntercept, 0]);
        }
      }
      if (c.b.abs() > eps) {
        final yIntercept = c.c / c.b; // x=0
        if (yIntercept.isFinite) {
          candidates.add([0, yIntercept]);
        }
      }
    }

    // Filtrar factibles
    final feasible = candidates.where((p) => _isFeasible(p, all)).toList();

    // Evaluar vértices
    final eval = <Map<String, dynamic>>[];
    for (final v in feasible) {
      final z = input.c1 * v[0] + input.c2 * v[1];
      eval.add({"x": v[0], "y": v[1], "z": z});
    }

    // Elegir mejor
    Map<String, dynamic>? best;
    for (final e in eval) {
      if (best == null) {
        best = e;
      } else {
        if (input.maximize) {
          if ((e["z"] as double) > (best["z"] as double) + eps) best = e;
        } else {
          if ((e["z"] as double) < (best["z"] as double) - eps) best = e;
        }
      }
    }

    // Ordenar polígono factible por ángulo para pintar
    final poly = _sortPolygon(feasible);

    return LpResult(
      feasibleVertices: poly,
      evaluatedVertices: eval,
      bestPoint: best == null ? null : [best["x"], best["y"]],
      bestValue: best == null ? null : (best["z"] as double),
    );
  }

  static List<double>? _intersect(ConstraintModel c1, ConstraintModel c2) {
    // Resolver:
    // a1 x + b1 y = c1
    // a2 x + b2 y = c2
    final a1 = c1.a, b1 = c1.b, c_1 = c1.c;
    final a2 = c2.a, b2 = c2.b, c_2 = c2.c;
    final det = a1 * b2 - a2 * b1;
    if (det.abs() < eps) return null; // paralelas o coincidentes
    final x = (c_1 * b2 - c_2 * b1) / det;
    final y = (a1 * c_2 - a2 * c_1) / det;
    return [x, y];
  }

  static bool _isFeasible(List<double> p, List<ConstraintModel> all) {
    final x = p[0], y = p[1];
    for (final c in all) {
      final lhs = c.a * x + c.b * y;
      if (c.sense == Sense.le) {
        if (lhs - c.c > eps) return false;
      } else {
        if (c.c - lhs > eps) return false;
      }
    }
    if (x < -eps || y < -eps) return false;
    if (!x.isFinite || !y.isFinite) return false;
    return true;
  }

  static List<List<double>> _sortPolygon(List<List<double>> pts) {
    if (pts.length <= 2) return pts;
    final cx = pts.map((p) => p[0]).reduce((a, b) => a + b) / pts.length;
    final cy = pts.map((p) => p[1]).reduce((a, b) => a + b) / pts.length;
    pts.sort((p1, p2) {
      final a1 = math.atan2(p1[1] - cy, p1[0] - cx);
      final a2 = math.atan2(p2[1] - cy, p2[0] - cx);
      return a1.compareTo(a2);
    });
    return pts;
  }

  static String _key(List<double> p) =>
      "${(p[0]).toStringAsFixed(6)}_${(p[1]).toStringAsFixed(6)}";
}