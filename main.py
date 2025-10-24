#!/usr/bin/env python3
"""
Sistema Unificado de Redes Neuronales
=====================================

Este script principal permite ejecutar dos sistemas de redes neuronales:
1. Diagnóstico de Diabetes (Clasificación)
2. Predicción de Precios de Viviendas (Regresión)

Autor: Sistema de IA
Fecha: 2024
"""

import os
import sys
from typing import Optional

def mostrar_banner():
    """Muestra el banner principal del sistema."""
    print("=" * 80)
    print("🧠 SISTEMA UNIFICADO DE REDES NEURONALES")
    print("=" * 80)
    print("Implementación de redes neuronales desde cero para:")
    print("  • Diagnóstico de Diabetes (Clasificación)")
    print("  • Predicción de Precios de Viviendas (Regresión)")
    print("=" * 80)

def mostrar_menu():
    """Muestra el menú principal de opciones."""
    print("\n📋 MENÚ PRINCIPAL")
    print("-" * 40)
    print("1. 🏥 Diagnóstico de Diabetes")
    print("   - Clasificación binaria")
    print("   - Dataset: Pima Indians Diabetes")
    print("   - Predicción de riesgo de diabetes")
    print()
    print("2. 🏠 Predicción de Precios de Viviendas")
    print("   - Regresión")
    print("   - Dataset: Boston Housing")
    print("   - Estimación de precios inmobiliarios")
    print()
    print("3. ℹ️  Información del Sistema")
    print("4. 🚪 Salir")
    print("-" * 40)

def mostrar_informacion():
    """Muestra información detallada del sistema."""
    print("\n" + "=" * 60)
    print("ℹ️  INFORMACIÓN DEL SISTEMA")
    print("=" * 60)
    
    print("\n🔬 CARACTERÍSTICAS TÉCNICAS:")
    print("  • Red neuronal implementada desde cero")
    print("  • Funciones de activación: ReLU, Sigmoid, Linear")
    print("  • Funciones de pérdida: MSE, Binary Crossentropy")
    print("  • Optimizador: Gradient Descent")
    print("  • Métricas: Accuracy, Precision, Recall, F1-Score, RMSE, R²")
    
    print("\n📊 DATASETS:")
    print("  • Diabetes: 768 muestras, 8 características")
    print("  • Boston Housing: 506 muestras, 13 características")
    
    print("\n🎨 VISUALIZACIONES GENERADAS:")
    print("  • Curvas de pérdida durante entrenamiento")
    print("  • Matrices de confusión (clasificación)")
    print("  • Gráficos de predicción vs valores reales")
    print("  • Análisis de residuos y errores")
    print("  • Comparación de métricas")
    
    print("\n💾 ARCHIVOS DE SALIDA:")
    print("  • Modelos entrenados (.pkl)")
    print("  • Estadísticas de normalización (.pkl)")
    print("  • Gráficos de alta resolución (.png)")
    
    print("\n🔗 REFERENCIA:")
    print("  • Basado en: https://cienciadedatos.net/documentos/py35-redes-neuronales-python")
    print("=" * 60)

def verificar_dependencias():
    """Verifica que todas las dependencias estén disponibles."""
    dependencias = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy'
    ]
    
    faltantes = []
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError:
            faltantes.append(dep)
    
    if faltantes:
        print(f"\n❌ ERROR: Faltan las siguientes dependencias:")
        for dep in faltantes:
            print(f"   • {dep}")
        print(f"\nInstale las dependencias con:")
        print(f"   pip install {' '.join(faltantes)}")
        return False
    
    return True

def verificar_archivos_datos():
    """Verifica que los archivos de datos estén disponibles."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    archivos_requeridos = [
        os.path.join(current_dir, "data", "diabetes.csv"),
        os.path.join(current_dir, "data", "Boston.csv"),
        os.path.join(current_dir, "neural_network.py"),
        os.path.join(current_dir, "diagnostico_diabetes.py"),
        os.path.join(current_dir, "prediccion_precio.py")
    ]
    
    faltantes = []
    for archivo in archivos_requeridos:
        if not os.path.exists(archivo):
            faltantes.append(os.path.basename(archivo))
    
    if faltantes:
        print(f"\n❌ ERROR: Faltan los siguientes archivos:")
        for archivo in faltantes:
            print(f"   • {archivo}")
        return False
    
    return True

def ejecutar_diabetes():
    """Ejecuta el sistema de diagnóstico de diabetes."""
    try:
        print("\n🚀 Iniciando Sistema de Diagnóstico de Diabetes...")
        print("=" * 60)
        
        # Importar y ejecutar el módulo de diabetes
        import diagnostico_diabetes
        diagnostico_diabetes.main()
        
    except Exception as e:
        print(f"\n❌ ERROR al ejecutar el sistema de diabetes:")
        print(f"   {str(e)}")
        print("\nVerifique que todos los archivos estén presentes y las dependencias instaladas.")

def ejecutar_boston():
    """Ejecuta el sistema de predicción de precios de viviendas."""
    try:
        print("\n🚀 Iniciando Sistema de Predicción de Precios de Viviendas...")
        print("=" * 70)
        
        # Importar y ejecutar el módulo de Boston
        import prediccion_precio
        prediccion_precio.main()
        
    except Exception as e:
        print(f"\n❌ ERROR al ejecutar el sistema de predicción de precios:")
        print(f"   {str(e)}")
        print("\nVerifique que todos los archivos estén presentes y las dependencias instaladas.")

def obtener_opcion_usuario() -> Optional[str]:
    """Obtiene y valida la opción del usuario."""
    try:
        opcion = input("\n👉 Seleccione una opción (1-4): ").strip()
        return opcion
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
        return "4"
    except EOFError:
        return "4"

def main():
    """Función principal del sistema unificado."""
    # Mostrar banner
    mostrar_banner()
    
    # Verificar dependencias
    print("🔍 Verificando dependencias...")
    if not verificar_dependencias():
        sys.exit(1)
    
    # Verificar archivos de datos
    print("📁 Verificando archivos de datos...")
    if not verificar_archivos_datos():
        sys.exit(1)
    
    print("✅ Todas las verificaciones pasaron correctamente!")
    
    # Bucle principal del menú
    while True:
        mostrar_menu()
        opcion = obtener_opcion_usuario()
        
        if opcion == "1":
            ejecutar_diabetes()
            
        elif opcion == "2":
            ejecutar_boston()
            
        elif opcion == "3":
            mostrar_informacion()
            
        elif opcion == "4":
            print("\n👋 ¡Gracias por usar el Sistema de Redes Neuronales!")
            print("=" * 60)
            break
            
        else:
            print("\n❌ Opción no válida. Por favor, seleccione 1, 2, 3 o 4.")
        
        # Pausa antes de mostrar el menú nuevamente
        if opcion in ["1", "2"]:
            input("\n📝 Presione Enter para volver al menú principal...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {str(e)}")
        print("Por favor, contacte al administrador del sistema.")
        sys.exit(1)