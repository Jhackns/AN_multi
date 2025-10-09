import 'package:flutter/material.dart';
import 'dart:math' as math;

class FeasiblePainter extends CustomPainter {
  final List<List<double>> polygon; // puntos del polígono factible
  final List<double>? best; // punto óptimo
  final Color strokeColor;

  FeasiblePainter({
    required this.polygon,
    required this.best,
    required this.strokeColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final margin = 24.0;
    final w = size.width - margin * 2;
    final h = size.height - margin * 2;

    // Calcular rango
    double maxX = 1, maxY = 1;
    for (final p in polygon) {
      maxX = math.max(maxX, p[0]);
      maxY = math.max(maxY, p[1]);
    }
    if (best != null) {
      maxX = math.max(maxX, best![0]);
      maxY = math.max(maxY, best![1]);
    }
    if (maxX <= 0) maxX = 1;
    if (maxY <= 0) maxY = 1;

    double sx(double x) => margin + (x / maxX) * w;
    double sy(double y) => size.height - margin - (y / maxY) * h;

    // Fondo
    final bg = Paint()..color = Colors.white..style = PaintingStyle.fill;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(0, 0, size.width, size.height),
        const Radius.circular(12),
      ),
      bg,
    );

    // Ejes
    final axis = Paint()
      ..color = const Color(0xFFB3E5FC)
      ..strokeWidth = 1.2;
    // X axis
    canvas.drawLine(Offset(margin, sy(0)), Offset(size.width - margin, sy(0)), axis);
    // Y axis
    canvas.drawLine(Offset(sx(0), margin), Offset(sx(0), size.height - margin), axis);

    // Grid y marcas
    final ticks = 4;
    final stepX = maxX / ticks;
    final stepY = maxY / ticks;
    final grid = Paint()
      ..color = const Color(0xFFE3F2FD)
      ..strokeWidth = 1.0;
    for (int i = 1; i <= ticks; i++) {
      final xv = i * stepX;
      final yv = i * stepY;
      // líneas verticales y etiquetas X
      canvas.drawLine(Offset(sx(xv), sy(0)), Offset(sx(xv), margin), grid);
      _label(canvas, '${xv.toStringAsFixed(2)}', Offset(sx(xv) - 12, sy(0) + 4));
      // líneas horizontales y etiquetas Y
      canvas.drawLine(Offset(sx(0), sy(yv)), Offset(size.width - margin, sy(yv)), grid);
      _label(canvas, '${yv.toStringAsFixed(2)}', Offset(sx(0) - 30, sy(yv) - 8));
    }

    // Etiquetas de ejes
    _label(canvas, 'X₁', Offset(size.width - margin + 4, sy(0) - 12), bold: true);
    _label(canvas, 'X₂', Offset(sx(0) - 16, margin - 8), bold: true);

    // Polígono factible
    if (polygon.length >= 2) {
      final path = Path();
      path.moveTo(sx(polygon[0][0]), sy(polygon[0][1]));
      for (int i = 1; i < polygon.length; i++) {
        path.lineTo(sx(polygon[i][0]), sy(polygon[i][1]));
      }
      path.close();
      final fill = Paint()
        ..color = strokeColor.withOpacity(0.15)
        ..style = PaintingStyle.fill;
      canvas.drawPath(path, fill);

      final stroke = Paint()
        ..color = strokeColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;
      canvas.drawPath(path, stroke);
    }

    // Punto óptimo
    if (best != null) {
      final p = Offset(sx(best![0]), sy(best![1]));
      final dot = Paint()..color = Colors.redAccent;
      canvas.drawCircle(p, 5.5, dot);
      _label(canvas, 'Óptimo (${best![0].toStringAsFixed(2)}, ${best![1].toStringAsFixed(2)})',
          Offset(p.dx + 6, p.dy - 18), bold: true, color: Colors.redAccent);
    }
  }

  @override
  bool shouldRepaint(covariant FeasiblePainter oldDelegate) {
    return oldDelegate.polygon != polygon || oldDelegate.best != best || oldDelegate.strokeColor != strokeColor;
  }

  void _label(Canvas canvas, String text, Offset offset, {bool bold = false, Color color = const Color(0xFF0D47A1)}) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: bold ? 12 : 11,
          fontWeight: bold ? FontWeight.w700 : FontWeight.w500,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    tp.layout();
    tp.paint(canvas, offset);
  }
}