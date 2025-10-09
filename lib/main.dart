import 'package:flutter/material.dart';
import 'theme/app_theme.dart';
import 'pages/home_page.dart';

void main() {
  runApp(const PuntoOptimoApp());
}

class PuntoOptimoApp extends StatelessWidget {
  const PuntoOptimoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Optimización LP — Móvil',
      theme: AppTheme.lightBlue,
      debugShowCheckedModeBanner: false,
      home: const HomePage(),
    );
  }
}