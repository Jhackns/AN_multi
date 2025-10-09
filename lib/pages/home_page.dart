import 'package:flutter/material.dart';
import '../models/constraint.dart';
import '../models/lp_input.dart';
import '../logic/lp_solver.dart';
import '../widgets/constraint_row.dart';
import '../widgets/graph_view.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  final TextEditingController _c1Ctrl = TextEditingController(text: '3');
  final TextEditingController _c2Ctrl = TextEditingController(text: '5');

  int _n = 3;
  final List<TextEditingController> _aCtrls = [];
  final List<TextEditingController> _bCtrls = [];
  final List<TextEditingController> _cCtrls = [];
  final List<Sense> _senses = [];

  LpResult? _result;
  bool _currentMaximize = true;

  @override
  void initState() {
    super.initState();
    _initRows(_n);
  }

  void _initRows(int count) {
    _aCtrls.clear();
    _bCtrls.clear();
    _cCtrls.clear();
    _senses.clear();
    for (int i = 0; i < count; i++) {
      _aCtrls.add(TextEditingController(text: i == 0 ? '1' : i == 1 ? '0' : '1'));
      _bCtrls.add(TextEditingController(text: i == 0 ? '2' : i == 1 ? '3' : '1'));
      _cCtrls.add(TextEditingController(text: i == 0 ? '8' : i == 1 ? '9' : '4'));
      _senses.add(Sense.le);
    }
    setState(() {});
  }

  void _setN(int n) {
    _n = n;
    _initRows(n);
  }

  void _useExample(bool maximize) {
    // Ejemplo típico:
    // Max z = 3x1 + 5x2
    // x1 + 2x2 ≤ 8
    // 3x2 ≤ 9
    // x1 + x2 ≤ 4
    _c1Ctrl.text = maximize ? '3' : '2';
    _c2Ctrl.text = maximize ? '5' : '3';
    _setN(3);
    // Autocompletar valores y ejecutar
    if (_aCtrls.length >= 3) {
      _aCtrls[0].text = '1';
      _bCtrls[0].text = '2';
      _cCtrls[0].text = '8';
      _senses[0] = Sense.le;

      _aCtrls[1].text = '0';
      _bCtrls[1].text = '3';
      _cCtrls[1].text = '9';
      _senses[1] = Sense.le;

      _aCtrls[2].text = '1';
      _bCtrls[2].text = '1';
      _cCtrls[2].text = '4';
      _senses[2] = Sense.le;
    }
    _calculate(maximize);
  }

  void _calculate(bool maximize) {
    _currentMaximize = maximize;
    final c1 = double.tryParse(_c1Ctrl.text) ?? 0;
    final c2 = double.tryParse(_c2Ctrl.text) ?? 0;
    final constraints = <ConstraintModel>[];
    for (int i = 0; i < _n; i++) {
      final a = double.tryParse(_aCtrls[i].text) ?? 0;
      final b = double.tryParse(_bCtrls[i].text) ?? 0;
      final c = double.tryParse(_cCtrls[i].text) ?? 0;
      constraints.add(ConstraintModel(a: a, b: b, c: c, sense: _senses[i]));
    }
    final input = LpInput(c1: c1, c2: c2, constraints: constraints, maximize: maximize);
    final res = LpSolver.solve(input);
    setState(() {
      _result = res;
    });
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Punto Óptimo — LP (Móvil)'),
          bottom: TabBar(
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white70,
            labelStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            indicatorColor: Colors.white,
            indicatorWeight: 3,
            tabs: const [
              Tab(text: 'Máximo'),
              Tab(text: 'Mínimo'),
            ],
          ),
        ),
        body: TabBarView(children: [
          _buildTab(true),
          _buildTab(false),
        ]),
      ),
    );
  }

  Widget _buildTab(bool maximize) {
    final isCurrent = _currentMaximize == maximize;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _objectiveCard(maximize),
          _configCard(),
          ...List.generate(_n, (i) => ConstraintRow(
                aController: _aCtrls[i],
                bController: _bCtrls[i],
                cController: _cCtrls[i],
                sense: _senses[i],
                onSenseChanged: (s) => setState(() => _senses[i] = s),
              )),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.auto_awesome),
                  label: const Text('Usar ejemplo'),
                  onPressed: () => _useExample(maximize),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  icon: const Icon(Icons.calculate),
                  label: Text(maximize ? 'Calcular Máximo' : 'Calcular Mínimo'),
                  onPressed: () => _calculate(maximize),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (_result != null && isCurrent) ...[
            _resultCard(_result!),
            GraphView(
              polygon: _result!.feasibleVertices,
              bestPoint: _result!.bestPoint,
              bestValue: _result!.bestValue,
            ),
          ],
        ],
      ),
    );
  }

  Widget _objectiveCard(bool maximize) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(maximize ? 'Función objetivo (Max)' : 'Función objetivo (Min)',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                SizedBox(
                  width: 140,
                  child: TextField(
                    controller: _c1Ctrl,
                    keyboardType: const TextInputType.numberWithOptions(decimal: true),
                    decoration: const InputDecoration(labelText: 'c₁ (X₁)'),
                  ),
                ),
                SizedBox(
                  width: 140,
                  child: TextField(
                    controller: _c2Ctrl,
                    keyboardType: const TextInputType.numberWithOptions(decimal: true),
                    decoration: const InputDecoration(labelText: 'c₂ (X₂)'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _configCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            const Text('Restricciones:'),
            const SizedBox(width: 12),
            DropdownButton<int>(
              value: _n,
              items: const [2, 3, 4, 5, 6]
                  .map((e) => DropdownMenuItem(value: e, child: Text('$e')))
                  .toList(),
              onChanged: (v) => v == null ? null : _setN(v),
            )
          ],
        ),
      ),
    );
  }

  Widget _resultCard(LpResult res) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Resultados', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (res.bestPoint != null)
              Wrap(
                spacing: 12,
                runSpacing: 8,
                children: [
                  Chip(label: Text('x* = ${res.bestPoint![0].toStringAsFixed(3)}')),
                  Chip(label: Text('y* = ${res.bestPoint![1].toStringAsFixed(3)}')),
                  Chip(label: Text('z* = ${res.bestValue!.toStringAsFixed(3)}')),
                ],
              )
            else
              const Text('No se encontró solución factible.'),
            const SizedBox(height: 8),
            _verticesTable(res),
          ],
        ),
      ),
    );
  }

  Widget _verticesTable(LpResult res) {
    if (res.evaluatedVertices.isEmpty) {
      return const Text('Sin vértices evaluados.');
    }
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columns: const [
          DataColumn(label: Text('x')),
          DataColumn(label: Text('y')),
          DataColumn(label: Text('z')),
        ],
        rows: res.evaluatedVertices
            .map((e) => DataRow(cells: [
                  DataCell(Text((e['x'] as double).toStringAsFixed(3))),
                  DataCell(Text((e['y'] as double).toStringAsFixed(3))),
                  DataCell(Text((e['z'] as double).toStringAsFixed(3))),
                ]))
            .toList(),
      ),
    );
  }
}