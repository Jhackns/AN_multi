import 'package:flutter/material.dart';
import '../models/constraint.dart';

class ConstraintRow extends StatefulWidget {
  final TextEditingController aController;
  final TextEditingController bController;
  final TextEditingController cController;
  final Sense sense;
  final ValueChanged<Sense> onSenseChanged;

  const ConstraintRow({
    super.key,
    required this.aController,
    required this.bController,
    required this.cController,
    required this.sense,
    required this.onSenseChanged,
  });

  @override
  State<ConstraintRow> createState() => _ConstraintRowState();
}

class _ConstraintRowState extends State<ConstraintRow> {
  late Sense _sense;

  @override
  void initState() {
    super.initState();
    _sense = widget.sense;
  }

  @override
  void didUpdateWidget(covariant ConstraintRow oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.sense != widget.sense) {
      _sense = widget.sense;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          crossAxisAlignment: WrapCrossAlignment.center,
          runSpacing: 8,
          spacing: 8,
          children: [
            _numField(widget.aController, label: 'a (X₁)', hint: 'coef. X₁'),
            _numField(widget.bController, label: 'b (X₂)', hint: 'coef. X₂'),
            _senseToggle(),
            _numField(widget.cController, label: 'c', hint: 'constante'),
          ],
        ),
      ),
    );
  }

  Widget _numField(TextEditingController controller,
      {required String label, String? hint}) {
    return SizedBox(
      width: 120,
      child: TextField(
        controller: controller,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(labelText: label, hintText: hint),
      ),
    );
  }

  Widget _senseToggle() {
    final isLe = _sense == Sense.le;
    return ToggleButtons(
      isSelected: [isLe, !isLe],
      onPressed: (index) {
        setState(() {
          _sense = index == 0 ? Sense.le : Sense.ge;
        });
        widget.onSenseChanged(_sense);
      },
      borderRadius: BorderRadius.circular(12),
      selectedColor: Colors.white,
      fillColor: Theme.of(context).colorScheme.secondary,
      color: Theme.of(context).colorScheme.primary,
      children: const [
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Text('≤'),
        ),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Text('≥'),
        ),
      ],
    );
  }
}