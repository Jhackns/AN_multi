enum Sense { le, ge }

class ConstraintModel {
  double a;
  double b;
  double c;
  Sense sense;

  ConstraintModel({
    required this.a,
    required this.b,
    required this.c,
    this.sense = Sense.le,
  });

  ConstraintModel copyWith({double? a, double? b, double? c, Sense? sense}) {
    return ConstraintModel(
      a: a ?? this.a,
      b: b ?? this.b,
      c: c ?? this.c,
      sense: sense ?? this.sense,
    );
  }
}