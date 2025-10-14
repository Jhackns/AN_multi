# Punto Óptimo — Móvil (Flutter)

Aplicación Flutter para calcular y visualizar el punto óptimo de un problema de programación lineal en dos variables (X₁, X₂). Permite definir la función objetivo (Max/Min), configurar restricciones con sentidos ≤/≥, visualizar la región factible y obtener el óptimo con sus valores `x*`, `y*` y `z*`.

Este documento te guía paso a paso desde cero: instalación de herramientas (Android Studio y Flutter), ejecución en emulador o dispositivo físico, y despliegue en Android, Web y Windows.

## Contenido

- Qué es y cómo funciona
- Estructura del proyecto
- Requisitos previos
- Instalación rápida (Windows)
- Despliegue en Android (emulador y APK)
- Despliegue en Web y Windows
- Capturas de pantalla
- Uso de la aplicación (paso a paso)
- Ejemplo guiado
- Solución de problemas
- Comandos útiles
- Pruebas y calidad

---

## Qué es y cómo funciona

- La app resuelve problemas LP de dos variables mediante enumeración de vértices de la región factible y evaluación de la función objetivo.
- Añade automáticamente restricciones de no negatividad: `X₁ ≥ 0` y `X₂ ≥ 0`.
- Muestra:
  - El polígono factible en un plano (con ejes y cuadrícula).
  - El punto óptimo marcado y el valor óptimo `z*`.
  - Una tabla con los vértices evaluados.

## Estructura del proyecto

Ruta del proyecto: `d:\Python\amPuntoOptimo\mobil`

- `lib/`
  - `main.dart`: punto de entrada. Carga el tema y la `HomePage`.
  - `pages/home_page.dart`: pantalla principal con pestañas Máximo/Mínimo, inputs y acciones.
  - `logic/lp_solver.dart`: algoritmo que calcula vértices factibles y el óptimo.
  - `models/constraint.dart`: modelo de restricción (`a`, `b`, `c`, `sense` ≤/≥).
  - `models/lp_input.dart`: entrada del problema y estructura de resultados.
  - `widgets/constraint_row.dart`: fila de restricción con inputs y selector ≤/≥.
  - `widgets/graph_view.dart` y `widgets/feasible_painter.dart`: render del polígono y del óptimo.
  - `theme/app_theme.dart`: tema visual (azules, tarjetas, inputs y botones).
- `android/`, `ios/`, `web/`, `windows/`, `linux/`, `macos/`: soportes multiplataforma.
- `pubspec.yaml`: dependencias (Flutter, `cupertino_icons`), versión y configuración.
- `test/`: pruebas básicas de widget.

---

## Requisitos previos

1) Flutter SDK (estable) y Dart SDK incluidos en Flutter.
2) Android Studio (para compilar y ejecutar en Android) con:
   - Android SDK Platform y Platform Tools.
   - Emulador (AVD) opcional si no usarás dispositivo físico.
3) Windows 10/11 con virtualización habilitada si usarás emulador.
4) Git opcional (puedes descargar ZIP en su lugar).

Referencias oficiales:
- Flutter: https://docs.flutter.dev/get-started/install
- Android Studio: https://developer.android.com/studio

---

## Instalación rápida (Windows)

1. Instalar Android Studio
   - Descárgalo e instálalo.
   - Abre Android Studio y en el SDK Manager instala:
     - Al menos una plataforma Android reciente.
     - Android SDK Platform-Tools.
   - (Opcional) Instala un emulador creando un AVD en “Device Manager”.

2. Instalar Flutter SDK
   - Descarga Flutter para Windows (canal estable).
   - Descomprime en `C:\src\flutter` (recomendado).
   - Agrega `C:\src\flutter\bin` al `PATH` del sistema:
     - Panel de control → Sistema → Configuración avanzada → Variables de entorno → Path → Nuevo → `C:\src\flutter\bin`.

3. Verificar instalación
   - Abre PowerShell y ejecuta:
     - `flutter --version`
     - `flutter doctor`
   - Corrige lo que `flutter doctor` indique (plugins, licencias de Android, etc.).

4. Obtener el proyecto
   - Opción A (Git):
     - `git clone <tu-repositorio>`
     - En Windows, colócalo en `d:\Python\amPuntoOptimo\mobil` (o tu carpeta preferida).
   - Opción B (ZIP):
     - Descarga y descomprime el repositorio en la ruta que elijas.

5. Abrir en Android Studio
   - File → Open… → selecciona `d:\Python\amPuntoOptimo\mobil`.
   - Espera a que indexe el proyecto.
   - En la terminal integrada, ejecuta `flutter pub get`.

---

## Despliegue en Android

1) Ejecutar en emulador
- En Android Studio: crea un AVD (ej. Pixel) y ejecútalo.
- En la terminal del proyecto:
  - `flutter devices` (verás el emulador listado).
  - `flutter run` (lanza la app en el dispositivo conectado o emulador).

2) Ejecutar en dispositivo físico
- Activa “Opciones de desarrollador” y “Depuración USB” en tu móvil.
- Conecta por USB y acepta la huella de autorización.
- `flutter devices` debe listar tu dispositivo.
- `flutter run` para ejecutar.

3) Generar APK de producción
- `flutter build apk --release`
- Archivo resultante: `build\app\outputs\apk\release\app-release.apk`
- Pásalo al dispositivo para instalarlo manualmente.

4) Generar App Bundle (Play Store)
- `flutter build appbundle --release`
- Archivo: `build\app\outputs\bundle\release\app-release.aab`
- Nota: Para publicar, configura firma y datos de Play Console (fuera del alcance de este manual).

---

## Despliegue en Web y Windows

Web
- Habilitar (si no lo está): `flutter config --enable-web`
- Ejecutar en navegador: `flutter run -d chrome`
- Compilar sitio: `flutter build web` (salida en `build/web`)

Windows (Desktop)
- Habilitar: `flutter config --enable-windows-desktop`
- Ejecutar: `flutter run -d windows`
- Compilar: `flutter build windows` (generará binarios en `build/windows`)

Nota iOS (macOS requerido)
- Para iOS debes usar macOS con Xcode y `flutter build ios`.

---

## Uso de la aplicación (paso a paso)

1) Selecciona objetivo
- Pestañas en la barra superior: `Máximo` o `Mínimo`.

2) Define la función objetivo
- En la tarjeta “Función objetivo”, ingresa coeficientes:
  - `c₁ (X₁)` y `c₂ (X₂)` para `z = c₁·X₁ + c₂·X₂`.

3) Configura el número de restricciones
- En la tarjeta “Restricciones” usa el desplegable (2–6).

4) Ingresa cada restricción
- Cada fila representa: `a·X₁ + b·X₂ ≤ c` o `≥ c`.
- Rellena `a`, `b`, `c` y elige el sentido con el conmutador `≤ / ≥`.
- La app añade además `X₁ ≥ 0` y `X₂ ≥ 0` automáticamente.

5) Calcular
- Pulsa “Calcular Máximo” o “Calcular Mínimo” según la pestaña.
- Se mostrarán:
  - Tarjeta “Resultados” con `x*`, `y*`, `z*` (redondeados a 3 decimales).
  - Tabla con todos los vértices evaluados.
  - Gráfico con el polígono factible y el punto óptimo.

6) Cargar un ejemplo automático
- Botón “Usar ejemplo” completa un caso típico y ejecuta el cálculo.

Consejos de uso
- Usa valores razonables (evita magnitudes excesivamente grandes) para una visualización clara.
- Decimales son aceptados.
- Si ves “No se encontró solución factible”, revisa el sistema de restricciones.

---

## Capturas de pantalla (placeholders)

Cómo agregar imágenes
- La carpeta `screenshots/` ya está creada en la raíz del proyecto.
- Pega allí tus imágenes de la interfaz y de ejercicios.
- Usa los nombres sugeridos o cambia las rutas en este archivo si eliges otros nombres.

Galería (rutas de ejemplo)
- Inicio — pestaña Máximo:
  
  `![Inicio — pestaña Máximo](screenshots/home_max_tab.png)`

- Inicio — pestaña Mínimo:
  
  `![Inicio — pestaña Mínimo](screenshots/home_min_tab.png)`

- Resultados (chips x*, y*, z*):
  
  `![Resultados: óptimo](screenshots/results_card.png)`

- Gráfico del polígono factible y punto óptimo:
  
  `![Gráfico: región factible y óptimo](screenshots/graph_optimum.png)`


Nota
- Estas rutas son referencias; al agregar tus imágenes con esos nombres, se mostrarán directamente en el README.
- Si cambias nombres o agregas más capturas, ajusta las rutas en este documento.

---

## Ejemplo guiado

Objetivo (Máximo): `z = 3·X₁ + 5·X₂`

Restricciones:
- `X₁ + 2·X₂ ≤ 8`
- `3·X₂ ≤ 9`
- `X₁ + X₂ ≤ 4`

Procedimiento:
- Selecciona pestaña “Máximo”.
- En “Función objetivo”, escribe `c₁ = 3`, `c₂ = 5`.
- En “Restricciones”, pon 3.
- Completa las tres filas anteriores con `≤` y pulsa “Calcular Máximo”.
- Observa el gráfico y los valores `x*`, `y*`, `z*` en “Resultados”.

---

## Solución de problemas

- Emulador no arranca
  - Habilita virtualización (BIOS/UEFI). Cierra otras herramientas de virtualización.
  - Usa una imagen de sistema compatible en el AVD Manager.
- `flutter doctor` muestra errores
  - Abre Android Studio y acepta licencias del SDK.
  - Instala componentes faltantes desde el SDK Manager.
- Dispositivo físico no aparece
  - Activa “Depuración USB” y acepta la autorización.
  - Prueba `flutter devices`. Reinstala “Android USB driver” si es necesario.
- “No se encontró solución factible”
  - Verifica que las restricciones no sean mutuamente excluyentes.
  - Revisa signos y magnitudes de `a`, `b`, `c`.

---

## Comandos útiles

- Instalar dependencias: `flutter pub get`
- Limpiar artefactos: `flutter clean`
- Listar dispositivos: `flutter devices`
- Verificar instalación: `flutter doctor -v`
- Actualizar dependencias: `flutter pub upgrade --major-versions`

---

## Pruebas y calidad

- Ejecutar tests: `flutter test`
- Lints activados: `flutter_lints` (ver `analysis_options.yaml`).

---

## Notas técnicas (LP de 2 variables)

- La app calcula intersecciones de pares de restricciones y con ejes, filtra puntos factibles y evalúa `z` en cada vértice.
- Ordena los vértices por ángulo para dibujar el polígono factible.
- Solo está pensada para problemas en 2 variables (X₁, X₂).

---

## Créditos

Proyecto Flutter multi-plataforma. Si necesitas soporte adicional, consulta la documentación oficial de Flutter y Android Studio enlazada arriba.
