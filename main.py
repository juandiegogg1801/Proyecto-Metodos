import sys

import sympy
from PyQt5 import uic
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QTableWidgetItem, QGraphicsView, QGraphicsScene
import sympy as sp
import validators
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from tabulate import tabulate
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.datos = []
        self.f = None
        uic.loadUi("GUI_calculadora.ui", self)
        # Paneles Metodos
        self.puntofijoButton.clicked.connect(self.puntoFijo_Panel)
        self.biseccionButton.clicked.connect(self.biseccion_Panel)
        self.newtonButton.clicked.connect(self.newton_Panel)
        self.secanteButton.clicked.connect(self.secante_Panel)
        self.difeButton.clicked.connect(self.diferencias_Panel)
        self.jacobiButton.clicked.connect(self.jacobi_Panel)
        self.gaussButton.clicked.connect(self.gauss_Panel)
        self.trapecioButton.clicked.connect(self.trapecio_Panel)
        self.simpsonButton.clicked.connect(self.simpson_Panel)
        # Botones calculadora
        self.funcionButton.clicked.connect(self.calculator)
        self.funcionButton2.clicked.connect(self.calculator)
        self.funcionButton3.clicked.connect(self.calculator)
        #Bloquear botones ver grafica
        self.graf_button.setEnabled(False)
        self.bis_grafica.setEnabled(False)
        # Panel inicial
        self.puntoFijo_Panel()
    def validar_campos_no_vacios(self, *campos):
        for campo in campos:
            if campo.text().strip() == "":
                QMessageBox.critical(self, "Error", "campos vacios")
                return False
        return True

    def validar_funcion(self, entrada):
        try:
            x = sp.symbols('x')
            expr = sp.sympify(entrada)
            if isinstance(expr, sp.FunctionClass) or isinstance(expr, sp.Basic):
                return True
            else:
                return False
        except sp.SympifyError:
            return False

    def vaidar_numero(self, string):
        try:
            float(string)
            return True
        except (TypeError, ValueError):
            return False

    # -------------------------JUAN DIEGO-------------------------------

    def puntoFijo_Panel(self):  # Abrir panel punto fjo
        print('punto fijo')
        self.stackedWidget.setCurrentIndex(0)
        self.calcularButton.clicked.connect(self.validar_puntofijo)

    def validar_puntofijo(self):
        input_fx = self.fx_lineEdit.text()
        input_gx = self.gx_lineEdit.text()
        input_xo = self.xo_lineEdit.text()
        input_tol = self.tol_lineEdit.text()

        # Validar campos no vacíos
        if not self.validar_campos_no_vacios(self.fx_lineEdit, self.gx_lineEdit, self.xo_lineEdit, self.tol_lineEdit):
            print("Algunos campos están vacíos.")
            return

        # Validar funciones
        # fx
        if not self.validar_funcion(input_fx):
            print('fx no es funncion')
            self.fx_lineEdit.clear()
            return
        # gx
        if not self.validar_funcion(input_gx):
            print('gx no es funncion')
            self.gx_lineEdit.clear()
            return

        # validar numeros
        if not self.vaidar_numero(input_xo):
            print('Debe ser un numero decimal')
            self.xo_lineEdit.clear()
            return
        if not self.vaidar_numero(input_tol):
            print('Debe ser un numero decimal')
            self.tol_lineEdit.clear()
            return

        # Si todos los campos están completos y las funciones son válidas
        print("Todos los campos están completos y las funciones son válidas.")
        self.puntoFijo(input_fx, input_gx, input_xo, input_tol)

    def puntoFijo(self, fun_fx, fun_gx, Xo, tol):
        x = sp.symbols('x')
        f = fun_fx
        f = sp.lambdify(x, f)
        g = fun_gx
        g = sp.lambdify(x, g)
        x0 = float(Xo)
        tol = float(tol)
        n = 15  # numero de iteraciones
        for i in range(1, n + 1):
            x1 = g(x0)
            if i == 1:
                error = 1
            else:
                error = abs((x1 - x0) / x1)
            if error < tol:
                self.datos.append((round(x0, 5), round(g(x0), 5), round(f(x0), 5), round(error, 5)))
                result = round(x1, 5)
                # Mostrar resultado
                self.result_Label.setText(f'Solucion: {str(result)}')
                break
            self.datos.append((round(x0, 5), round(g(x0), 5), round(f(x0), 5), round(error, 5)))
            x0 = x1

        # Mostrar tabla
        self.mostrar_tabla()
        self.show()
        #self.tab_button.clicked.connect(self.mostrar_tabla)

        # Mostrar grafica
        #Desbloquear boton mostrar grafica hasta que se hayan mostrado los valores
        self.graf_button.setEnabled(True)
        self.calcularButton.setEnabled(False)
        self.f = f
        self.graf_button.clicked.connect(self.mostrar_grafica)

    def biseccion_Panel(self):
        print('biseccion')
        self.stackedWidget.setCurrentIndex(1)
        self.bis_calcular.clicked.connect(self.validar_biseccion)

    def validar_biseccion(self):
        input_fx = self.bis_fx.text()
        input_a = self.bis_a.text()
        input_b = self.bis_b.text()
        input_tol = self.bis_tol.text()

        # Validar campos no vacíos
        if not self.validar_campos_no_vacios(self.bis_fx, self.bis_a, self.bis_b, self.bis_tol):
            print("Algunos campos están vacíos.")
            return
        # Validar funciones
        # fx
        if not self.validar_funcion(input_fx):
            print('fx no es funncion')
            self.bis_fx.clear()
            return

        # validar numeros
        if not self.vaidar_numero(input_a):
            print('Debe ser un numero decimal')
            self.bis_a.clear()
            return
        if not self.vaidar_numero(input_b):
            print('Debe ser un numero decimal')
            self.bis_b.clear()
            return
        if not self.vaidar_numero(input_tol):
            print('Debe ser un numero decimal')
            self.bis_tol.clear()
            return
        self.biseccion(input_fx, input_a, input_b, input_tol)

    def biseccion(self, fun_fx, a, b, tol):
        print('biseccion')
        x = sp.symbols('x')
        f = fun_fx
        f = sp.lambdify(x, f)
        a = float(a)
        b = float(b)
        tol = float(tol)
        xi = a
        xr = b
        i = 0
        error = 1
        datos = []
        if f(a) * f(b) > 0:
            print('La funcion no cambia de signo')
        while error > tol:
            xi = xr
            xr = (a + b) / 2
            # Error
            if i == 0:
                error = 1
            else:
                error = abs((xr - xi) / xr)
            # Almacenar
            datos.append((i, round(xr, 5), round(a, 5), round(b, 5), round(f(a), 5), round(f(xr), 5), round(error, 5)))
            if f(a) * f(xr) < 0:
                b = xr
            if f(xr) * f(b) < 0:
                a = xr
            i += 1
        result = round(xr, 5)
        self.bis_solucion.setText(f'Solucion: {str(result)}')
        # Mostrar tabla
        # Mostrar grafica

    def newton_Panel(self):
        print('newton')
        self.stackedWidget.setCurrentIndex(2)
        self.new_cal.clicked.connect(self.validar_newton)

    def validar_newton(self):
        input_fx = self.new_fx.text()
        input_dgx = self.new_dgx.text()
        input_xo = self.new_xo.text()
        input_tol = self.new_tol.text()
        # Validar campos no vacíos
        if not self.validar_campos_no_vacios(self.new_fx, self.new_dgx, self.new_xo, self.new_tol):
            print("Algunos campos están vacíos.")
            return

        # Validar funciones
        # fx
        if not self.validar_funcion(input_fx):
            print('fx no es funncion')
            self.new_fx.clear()
            return
        # dgx
        if not self.validar_funcion(input_dgx):
            print('gx no es funncion')
            self.new_dgx.clear()
            return
        # validar numeros
        if not self.vaidar_numero(input_xo):
            print('Debe ser un numero decimal')
            self.new_xo.clear()
            return
        if not self.vaidar_numero(input_tol):
            print('Debe ser un numero decimal')
            self.new_tol.clear()
            return
        print("Todos los campos están completos y las funciones son válidas.")
        self.newton(input_fx, input_dgx, input_xo, input_tol)

    def newton(self, input_fx, input_dgx, input_xo, input_tol):
        print('bienvenido')
        x = sp.symbols('x')
        f = input_fx
        dg = input_dgx
        df = sp.diff(f, x)
        f = sp.lambdify(x, f, 'numpy')
        df = sp.lambdify(x, df, 'numpy')
        dg = sp.lambdify(x, dg, 'numpy')
        x0 = float(input_xo)
        tol = float(input_tol)
        error = 0
        n = 15
        datos = []
        for i in range(1, n + 1):
            x1 = x0 - f(x0) / df(x0)
            if i == 1:
                error = 1
            else:
                error = abs((x1 - x0) / x1)
            datos.append((i, round(x1, 5), round(abs(x1-x0), 5), round(dg(x0), 5), round(error, 5)))
            result = round(x1, 5)
            if abs(x1 - x0) < tol:
                self.new_result.setText(f'Solucion: {str(result)}')
                # Imprimir los datos como una tabla
                headers = ["Iteración", "X", "G(X)", "F(X)", "Error"]
                print(tabulate(datos, headers=headers, tablefmt="grid"))
                return
            x0 = x1
    # -------------------------LADY-------------------------------
    def secante_Panel(self):
        print('secante')
        self.stackedWidget.setCurrentIndex(3)

    def diferencias_Panel(self):
        print('diferencias')
        self.stackedWidget.setCurrentIndex(4)

    def jacobi_Panel(self):
        print('jacobi')
        self.stackedWidget.setCurrentIndex(5)

    # -------------------------JIMENA-------------------------------
    def gauss_Panel(self):
        print('gauss')
        self.stackedWidget.setCurrentIndex(6)

    def trapecio_Panel(self):
        print('trapecio')
        self.stackedWidget.setCurrentIndex(7)

    def simpson_Panel(self):
        print('simpson')
        self.stackedWidget.setCurrentIndex(8)

    # ---------------------------------------------------------------

    # Falta  metodo para ingresar la funcion por calculadora
    def calculator(self):
        print('calculator')
        self.stackedWidget.setCurrentIndex(9)
        # Boton confirmar
        self.next_button.clicked.connect(self.puntoFijo_Panel)

    # Falta para mostrar la grafica de la funcion
    def mostrar_grafica(self):
        print('mostrar grafica')
        self.stackedWidget.setCurrentIndex(10)
        min = -5
        max = 5
        eje_x = [x / 10 for x in range(10 * min, 10 * max + 1)]
        eje_y = [self.f(x) for x in eje_x]
        print(eje_x)
        print(eje_y)
        plt.plot(eje_x, eje_y)
        plt.show()

    def mostrar_tabla(self):
        #self.stackedWidget.setCurrentIndex(11)
        print('mostrar tabla')
        datos_str = [[str(elemento) for elemento in registro] for registro in self.datos]
        # Imprimir los datos como una tabla
        #headers = ["Iteración", "X", "G(X)", "F(X)", "Error"]
        #print(tabulate(datos_str, headers=headers, tablefmt="grid"))
        fila = 0
        for registro in datos_str:
            columna = 0
            self.table_puntoFijo.insertRow(fila)
            for elemento in registro:
                celda = QTableWidgetItem(elemento)
                self.table_puntoFijo.setItem(fila, columna, celda)
                columna += 1
            fila += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Calculadora = MainWindow()
    Calculadora.show()
    sys.exit(app.exec_())
