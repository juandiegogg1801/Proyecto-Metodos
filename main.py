import sys

import sympy
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QTableWidgetItem
import sympy as sp
import validators
from tabulate import tabulate

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.datos = []
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
        # Panel calculadora boton
        self.funcionButton.clicked.connect(self.calculator)
        self.funcionButton2.clicked.connect(self.calculator)
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
        self.graf_button.clicked.connect(self.mostrar_grafica)

    def biseccion_Panel(self):
        print('biseccion')
        self.stackedWidget.setCurrentIndex(1)

    def newton_Panel(self):
        print('newton')
        self.stackedWidget.setCurrentIndex(2)

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
        self.stackedWidget.setCurrentIndex(10)
        print('mostrar grafica')

    # Falta para mostrar la tabla con i,x,error
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
