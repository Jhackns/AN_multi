import 'constraint.dart';

class LpInput {
  double c1; // coeficiente de x1 en la función objetivo
  double c2; // coeficiente de x2 en la función objetivo
  List<ConstraintModel> constraints;
  bool maximize; // true: Max; false: Min

  LpInput({
    required this.c1,
    required this.c2,
    required this.constraints,
    this.maximize = true,
  });
}

class LpResult {
  final List<List<double>> feasibleVertices; // puntos factibles (x, y)
  final List<Map<String, dynamic>> evaluatedVertices; // con z
  final List<double>? bestPoint; // (x*, y*)
  final double? bestValue; // z*

  LpResult({
    required this.feasibleVertices,
    required this.evaluatedVertices,
    this.bestPoint,
    this.bestValue,
  });
}