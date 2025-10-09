import 'package:flutter/material.dart';
import 'feasible_painter.dart';

class GraphView extends StatelessWidget {
  final List<List<double>> polygon;
  final List<double>? bestPoint;
  final double? bestValue;

  const GraphView({
    super.key,
    required this.polygon,
    required this.bestPoint,
    this.bestValue,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: SizedBox(
          height: 300,
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: FeasiblePainter(
                    polygon: polygon,
                    best: bestPoint,
                    strokeColor: Theme.of(context).colorScheme.secondary,
                  ),
                ),
              ),
              if (bestPoint != null && bestValue != null)
                Positioned(
                  top: 8,
                  right: 8,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.95),
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: const [
                        BoxShadow(color: Color(0x22000000), blurRadius: 6, offset: Offset(0, 2)),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Óptimo', style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 6),
                        _row('x*', bestPoint![0]),
                        _row('y*', bestPoint![1]),
                        _row('z*', bestValue!),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _row(String label, double value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('$label:', style: const TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(width: 6),
        Text(value.toStringAsFixed(3)),
      ],
    );
  }
}