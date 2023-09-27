import sys


from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QTableWidgetItem, QInputDialog
import sympy as sp
import validators
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi(r"GUI_calculadora.ui", self)
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
        #self.datos = []
        #self.f = None

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
            if isinstance(expr, sp.FunctionClass) or isinstance(expr,
                                                                sp.Basic):  # Verifica si es una función o una expresión básica
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
        self.puntoFijo(self, input_fx, input_xo, input_tol)

    def puntoFijo_Panel(self):  # Abrir panel punto fjo
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
        self.puntoFijo(input_fx, input_gx, input_xo, input_tol)

    def puntoFijo(self, fun_fx, fun_gx, Xo, tol):
        x = sp.symbols('x')
        f = fun_fx
        f = sp.lambdify(x, f)
        g = fun_gx
        g = sp.lambdify(x, g)
        x0 = float(Xo)
        tol = float(tol)
        datos = []
        n = 15  # numero de iteraciones
        for i in range(1, n + 1):
            x1 = g(x0)
            if i == 1:
                error = 1
            else:
                error = abs((x1 - x0) / x1)
            if error < tol:
                datos.append((round(x0, 5), round(g(x0), 5), round(f(x0), 5), round(error, 5)))
                result = round(x1, 5)
                # Mostrar resultado
                self.result_Label.setText(f'Solucion: {str(result)}')
                break
            datos.append((round(x0, 5), round(g(x0), 5), round(f(x0), 5), round(error, 5)))
            x0 = x1

        # Mostrar tabla
        self.mostrar_tabla(datos)
        self.show()
        #self.tab_button.clicked.connect(self.mostrar_tabla)

        # Mostrar grafica
        #Mostrar grafica
        print('mostrar grafica')
        min = -5
        max = 5
        eje_x = [x / 10 for x in range(10 * min, 10 * max + 1)]
        eje_y = [f(x) for x in eje_x]
        print(eje_x)
        print(eje_y)
        plt.plot(eje_x, eje_y)
        plt.show()
       


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
        self.mostrar_tabla_biseccion(datos)
        self.show()
        # Mostrar tabla
        
        #Mostrar grafica
        print('mostrar grafica')
        min = -5
        max = 5
        eje_x = [x / 10 for x in range(10 * min, 10 * max + 1)]
        eje_y = [f(x) for x in eje_x]
        print(eje_x)
        print(eje_y)
        plt.plot(eje_x, eje_y)
        plt.show()

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

            #Mostrar grafica
        print('mostrar grafica')
        min = -5
        max = 5
        eje_x = [x / 10 for x in range(10 * min, 10 * max + 1)]
        eje_y = [f(x) for x in eje_x]
        print(eje_x)
        print(eje_y)
        plt.plot(eje_x, eje_y)
        plt.show()

    # -------------------------LADY-----------------------------------------------------
    def secante_Panel(self):
        print('secante')
        self.stackedWidget.setCurrentIndex(3)
        self.calcularButton_2.clicked.connect(self.validar_secante)
        
        

    def validar_secante(self):
        input_fx = self.fx_lineEdit_2.text()  # Nueva entrada para la función
        input_x0s = self.xo_lineEdit_2.text()
        input_x1s = self.xo_lineEdit_3.text()
        input_tols = self.tol_lineEdit_2.text()

        if not self.validar_campos_no_vacios(self.fx_lineEdit_2, self.xo_lineEdit_2, self.xo_lineEdit_3, self.tol_lineEdit_2):
            print("Algunos campos están vacíos.")
            return

        if not (self.vaidar_numero(input_x0s) and self.vaidar_numero(input_x1s) and self.vaidar_numero(input_tols)):
            print('Los valores deben ser números decimales.')
            return

        x0 = float(input_x0s)
        x1 = float(input_x1s)
        tol = float(input_tols)

        self.secante(x0, x1, tol, input_fx)  # Pasa la función como argumento

    def secante(self, x0, x1, tol, input_fx):  # Añade input_fx como argumento
        Secante_table = []  
        raiz = []
        Iter = []
        raiz.insert(0, 0)
        Iter.insert(0, 0)

        i = 0
        error = 100

        # Define la función ingresada por el usuario
        x = sp.symbols('x')
        fx = sp.sympify(input_fx)
        fx = sp.lambdify(x, fx)

        while abs(error) > tol:
            Iter.append(i)
            x2 = x1 - (fx(x1) * (x1 - x0)) / (fx(x1) - fx(x0))
            raiz.append(x2)
            x0 = x1
            x1 = x2
            i = i + 1
            error = (raiz[i] - raiz[i - 1])
            Secante_table.append([i, x2, abs(error), fx(raiz[i])])

        column_names = ["Iteración", "xi", "(xi-1)-xi", "Error (%)"]
        # Tabla de datos
        print("Tabla de Datos ")
        print(tabulate(Secante_table, headers=column_names))
        print("La raíz es: ", x2)
        self.result_Label_2.setText(f'Solucion: {str(x2)}')


        self.tableWidget.setRowCount(len(Secante_table))
        self.tableWidget.setColumnCount(len(Secante_table[0]))

        for j, column_names in enumerate(column_names):
            item = QTableWidgetItem(column_names)
            self.tableWidget.setHorizontalHeaderItem(j, item)

        # Llenar la tabla con los datos
        for i, row in enumerate(Secante_table):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.tableWidget.setItem(i, j, item)

        # Mostrar la tabla en la pantalla
        self.stackedWidget.setCurrentIndex(12)

        # Gráfica
        plt.title("Metodo Secante")
        plt.ylabel("Eje X")
        plt.xlabel("Eje y")

        a = -3
        b = 20
        n = 100
        xn = np.linspace(a, b, n)
        yn = fx(xn)
        plt.plot(xn, yn)
        plt.grid(True)
        plt.axhline(0, color="#ff0000")
        plt.axvline(0, color="#ff0000")
        plt.plot(x2, 0, 'ko')
        plt.show()


        #------------DIFERENCIAS DIVIDIDAS---------------

    def diferencias_Panel(self):
        print('diferencias')
        self.stackedWidget.setCurrentIndex(4)
        self.calcularButton_4.clicked.connect(self.mostrar_resultados)
    
    def mostrar_resultados(self):
        num_points_text = self.fx_lineEdit_5.text()
        try:
            num_points = int(num_points_text)
            xi = []
            fi = []

            for i in range(num_points):
                xi_text, ok1 = QInputDialog.getDouble(self, f"Ingrese el valor de x{i + 1}", f"x{i + 1}:")
                fi_text, ok2 = QInputDialog.getDouble(self, f"Ingrese el valor de f(x{i + 1})", f"f(x{i + 1}):")

                if ok1 and ok2:
                    xi.append(xi_text)
                    fi.append(fi_text)
                else:
                    print("Entrada cancelada o no válida.")
                    return

            # Cálculos de diferencias divididas y polinomio aquí
            
            n = len(xi)
            titulo = ['i   ', 'xi  ', 'fi  ']

            tabla = np.column_stack((np.arange(n), xi, fi))
            dfinita = np.zeros(shape=(n, n), dtype=float)
            tabla = np.column_stack((tabla, dfinita))

            [n, m] = np.shape(tabla)
            diagonal = n - 1
            j = 3

            while j < m:
                titulo.append('F[' + str(j - 2) + ']')

                i = 0
                paso = j - 2
                while i < diagonal:
                    denominador = (xi[i + paso] - xi[i])
                    numerador = tabla[i + 1, j - 1] - tabla[i, j - 1]
                    tabla[i, j] = numerador / denominador
                    i = i + 1
                diagonal = diagonal - 1
                j = j + 1

            self.table.setRowCount(n)
            self.table.setColumnCount(m)
            self.table.setHorizontalHeaderLabels(titulo)

            for i in range(n):
                for j in range(m):
                    if j > 2:  # Formatea los valores de la columna 3 en adelante
                        item = QTableWidgetItem('{:.4f}'.format(tabla[i, j]))  # Convierte el valor a cadena con 4 decimales de precisión
                    else:
                        item = QTableWidgetItem(str(tabla[i, j]))  # Convierte el valor a cadena
                    self.table.setItem(i, j, item)

         # Cálculo de dDividida
            dDividida = tabla[0, 3:]

            # Cálculo del polinomio de interpolación
            x = sp.Symbol('x')
            polinomio = fi[0]
            for j in range(1, n, 1):
                factor = dDividida[j - 1]
                termino = 1
                for k in range(0, j, 1):
                    termino = termino * (x - xi[k])
                polinomio = polinomio + termino * factor

            polisimple = polinomio.expand()
            polinomio_str = str(polisimple)

            # Mostrar el polinomio en el QLabel
            self.polinomio_label.setText(f'Polinomio de Interpolación:\n{polinomio_str}')
            
            # Genera valores de x para el polinomio
            x_values = np.linspace(min(xi), max(xi), 100)
            y_values = [polisimple.subs(x, val) for val in x_values]

         # Gráfica de los puntos y el polinomio
            plt.plot(xi, fi, 'o', label='Puntos')
            plt.plot(x_values, y_values, label='Polinomio de Interpolación', color='red')
            plt.legend()
            plt.xlabel('xi')
            plt.ylabel('fi')
            plt.title('Diferencias Divididas - Newton')
            plt.grid(True)
            plt.show()
   
        except ValueError:
            print("Ingrese un número válido para el número de puntos.")
        
 

#--------------------JACOBI-----------------
    def jacobi_Panel(self):
      print('jacobi')
      self.stackedWidget.setCurrentIndex(5)
      self.calcularButton_3.clicked.connect(self.validar_jacobi)


    def validar_jacobi(self):
        # Obtener el número de filas y columnas desde los widgets de entrada de texto
        filas_text = self.fx_lineEdit_3.text()
        columnas_text = self.fx_lineEdit_4.text()

        tolerancia_text = self.tol_lineEdit_3.text()
        iteraciones_text = self.tol_lineEdit_4.text()


        try:
            # Convertir los valores a números enteros
            filas = int(filas_text)
            columnas = int(columnas_text)

            # Obtener tolerancia e iteraciones desde los widgets de entrada de texto
            

            # Convertir tolerancia e iteraciones a números reales
            tolerancia = float(tolerancia_text)
            num_iteraciones = int(iteraciones_text)
            
            
            if filas <= 0 or columnas <= 0 or tolerancia <= 0 or num_iteraciones <= 0:
                print("Los valores ingresados deben ser números positivos.")
                return
            
            

            # Solicitar la matriz a través de ventanas emergentes
            matriz = []
            for i in range(filas):
                fila = []
                for j in range(columnas):
                    valor, ok = QInputDialog.getDouble(self, f"Fila {i + 1}, Columna {j + 1}", f"Ingrese el valor para Fila {i + 1}, Columna {j + 1}:")
                    if ok:
                        fila.append(valor)
                    else:
                        # El usuario canceló la entrada, puedes manejarlo aquí
                        return
                matriz.append(fila)

            # Solicitar el vector independiente a través de una ventana emergente
            vector_independiente = []
            for i in range(filas):
                valor, ok = QInputDialog.getDouble(self, f"Vector independiente, Elemento {i + 1}", f"Ingrese el valor para el elemento {i + 1} del vector independiente:")
                if ok:
                    vector_independiente.append(valor)
                else:
                    # El usuario canceló la entrada, puedes manejarlo aquí
                    return

            # Solicitar el vector inicial a través de una ventana emergente
            vector_inicial = []
            for i in range(filas):
                valor, ok = QInputDialog.getDouble(self, f"Vector Inicial, Elemento {i + 1}", f"Ingrese el valor para el elemento {i + 1} del vector inicial:")
                if ok:
                    vector_inicial.append(valor)
                else:
                    # El usuario canceló la entrada, puedes manejarlo aquí
                    return

            # Llamada a la función jacobi
            resultados = self.jacobi(np.array(matriz), np.array(vector_independiente), np.array(vector_inicial), tolerancia, num_iteraciones)
            # Mostrar la solución final en el QLabel
            solucion_final = resultados[-1][0]  # Tomar la última solución de la lista de resultados
            self.result_Label_3.setText(f"Solución final: {solucion_final}")



            # Mostrar los resultados en la tabla
            self.tableWidget_2.clear()
            self.tableWidget_2.setRowCount(len(resultados))
            self.tableWidget_2.setColumnCount(3)  # Iteración, x, Error

            column_names = ["x", "iteración", "error"]
            self.tableWidget_2.setHorizontalHeaderLabels(column_names)

            for i, (x, err) in enumerate(resultados):
                self.tableWidget_2.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.tableWidget_2.setItem(i, 1, QTableWidgetItem(str(x)))
                self.tableWidget_2.setItem(i, 2, QTableWidgetItem(str(err)))


        except ValueError:
            # Manejar el caso en que la entrada no sea un número válido
            self.tableWidget_2.clear()
            print("Por favor, ingrese valores numéricos válidos.")

    def jacobi(self, A, b, x0, tol, max_iterations):
        B = np.diag(np.diag(A))
        Lu = A - B
        x = x0
        results = []  # Lista para almacenar resultados de todas las iteraciones

        for i in range(max_iterations):
            D_inv = np.linalg.inv(B)
            xtemp = x
            x = np.dot(D_inv, np.dot(-Lu, x)) + np.dot(D_inv, b)
            err = np.linalg.norm(x - xtemp)
            results.append((x, err))

            if err < tol:
                break

        return results




    # -------------------------JIMENA-------------------------------
    def gauss_Panel(self):
        print('gauss')
        self.stackedWidget.setCurrentIndex(6)
        self.ButtonGS.clicked.connect(self.calculate_gauss_seidel)

    def calculate_gauss_seidel(self):
        matrix_text = self.textEdit_GS.toPlainText()
        vector_text = self.lineEditGS.text()
        initial_values_text = self.lineEdit_2GS.text()
        print(matrix_text)
        matrix_rows = [row.split(',') for row in matrix_text.split('\n') if row.strip()]
        A = np.array([[float(val.strip()) for val in row] for row in matrix_rows])

        b = np.array([float(val.strip()) for val in vector_text.split(',')])

        initial_values = [float(val.strip()) for val in initial_values_text.split(',')]
        x0 = np.array(initial_values)

        tol = 1e-6  # Tolerancia de convergencia
        max_iter = 100  # Número máximo de iteraciones
        x = x0.copy()
        # ...

        x_history = [x.copy()]

        for k in range(max_iter):
            for i in range(len(initial_values)):
                sigma = sum(A[i, j] * x[j] for j in range(len(initial_values)) if j != i)
                x[i] = (b[i] - sigma) / A[i, i]

            x_history.append(x.copy())
            if np.linalg.norm(x - x0) < tol:
                break

            x0 = x.copy()

        num_iterations = len(x_history)
        self.tableWidget_GS.setRowCount(num_iterations)

        for i, x_value in enumerate(x_history):
            iteration_item = QTableWidgetItem(str(i))
            x_value_item = QTableWidgetItem(", ".join(map(str, x_value)))

            if self.tableWidget_GS.columnCount() < 2:
                self.tableWidget_GS.setColumnCount(2)

            self.tableWidget_GS.setItem(i, 0, iteration_item)  # Columna 0: Iteración
            self.tableWidget_GS.setItem(i, 1, x_value_item)  # Columna 1: Valor de x
#-----------------------TRAPECIO---------------------------------------
    def trapecio_Panel(self):
        print('trapecio')
        self.stackedWidget.setCurrentIndex(7)
        self.calcularButton_T.clicked.connect(self.calcular_integral)


    def calcular_integral(self):
        function_text = self.fxT_lineEdit_2.text()

        try:
            x = sp.symbols('x')
            f_x = sp.sympify(function_text)  # Convierte la entrada del usuario en una expresión simbólica

            a_str = self.aT_lineEdit.text()
            b_str = self.bT_lineEdit.text()
            n_str = self.nT_lineEdit.text()

            # Luego, puedes convertir estos valores a números si es necesario
            try:
                a = float(a_str)
                b = float(b_str)
                n = int(n_str)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Por favor, ingrese valores numéricos válidos para a, b y n.')
                return

            # Calcular "h" en función de "n"
            h = (b - a) / n

            # Definir una función Python a partir de la expresión simbólica
            f = sp.lambdify(x, f_x, 'numpy')

            results = []

            for i in range(n + 1):
                x_val = a + i * h
                f_val = f(x_val)
                results.append([i, h, x_val, f_val])

            integral = 0.5 * (results[0][3] + results[-1][3])
            for i in range(1, n):
                integral += results[i][3]
            integral *= h
            resultado = f'Integral de f(x) usando Trapecio: {integral}'

            # Configurar el texto del QLabel con el resultado
            self.label_5T.setText(resultado)

            # Graficar la función y los trapecios
            colors = plt.cm.viridis(np.linspace(0, 1, n + 1))  # Colores basados en cmap 'viridis'
            x_vals = [result[2] for result in results]
            y_vals = [result[3] for result in results]
            plt.plot(x_vals, y_vals, label='f(x)')
            for i in range(n):
                x_trap = [results[i][2], results[i + 1][2]]
                y_trap = [results[i][3], results[i + 1][3]]
                plt.fill_between(x_trap, y_trap, alpha=0.5, label='Trapecio', color=colors[i])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()

            # Supongamos que tienes una lista de nombres para tus columnas, por ejemplo:
            nombres_columnas = ['n', 'h', 'x', 'f(x)']

            # Luego, configura los títulos de las columnas en tu QTableWidget
            self.tableWidget_T.setRowCount(len(results))
            self.tableWidget_T.setColumnCount(
                len(results[0]))  # Asumiendo que todos los resultados tienen la misma cantidad de columnas

            # Configura los títulos de las columnas
            for j, nombre_columna in enumerate(nombres_columnas):
                item = QTableWidgetItem(nombre_columna)
                self.tableWidget_T.setHorizontalHeaderItem(j, item)

            # Llena la tabla con los datos de results
            for i, row in enumerate(results):
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    self.tableWidget_T.setItem(i, j, item)


        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error al calcular la integral: {str(e)}')

        #------------------------SIMPSON-------------------------------

    def simpson_Panel(self):
        print('simpson')
        self.stackedWidget.setCurrentIndex(8)
        self.calcularButtonS.clicked.connect(self.calculate_integralS)
    def calculate_integralS(self):
        function_text = self.fxS_lineEdit.text()

        try:
            x = sp.symbols('x')
            f_x = sp.sympify(function_text)  # Convierte la entrada del usuario en una expresión simbólica

            a_str = self.a2_lineEdit.text()
            b_str = self.b2_lineEdit.text()
            n_str = self.n2_lineEdit.text()

            # Luego, puedes convertir estos valores a números si es necesario
            try:
                a = float(a_str)
                b = float(b_str)
                n = int(n_str)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Por favor, ingrese valores numéricos válidos para a, b y n.')
                return

            # Calcular "h" en función de "n"
            h = (b - a) / n

            # Definir una función Python a partir de la expresión simbólica
            f = sp.lambdify(x, f_x, 'numpy')

            results = []

            if n == 2:
                m = (a + b) / 2
                results.append([0, h, a, f(a), f(m)])
                results.append([1, h, m, f(m), f(m)])  # Agregamos f(m) para n=2
                results.append([2, h, b, f(b), f(m)])
                integral = (h / 3) * (results[0][3] + 4 * results[1][3] + results[2][3])

                # Graficar la función y los trapecios
                colors = ['blue', 'green', 'red']  # Colores predefinidos para n=2
                x_vals = [result[2] for result in results]
                y_vals = [result[3] for result in results]
                plt.plot(x_vals, y_vals, label='f(x)')
                for i in range(n):
                    x_simpson = [results[i][2], results[i + 1][2]]
                    y_simpson = [results[i][3], results[i + 1][3]]
                    plt.fill_between(x_simpson, y_simpson, alpha=0.5, label='Simpson', color=colors[i])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.show()
                # Supongamos que ya tienes el resultado calculado almacenado en la variable 'integral'
                resultado = f'Integral de f(x) usando Simpson: {integral}'

                # Configurar el texto del QLabel con el resultado
                self.label_5S.setText(resultado)

            elif n % 2 == 0:
                for i in range(n + 1):
                    x_val = a + i * h
                    f_val = f(x_val)
                    results.append([i, h, x_val, f_val, f(x_val)])

                integral = (h / 3) * (results[0][3] + 4 * results[1][3] + 2 * sum(
                    result[3] for result in results[2:n - 1:2]) + 4 * results[n][3])
                # Supongamos que ya tienes el resultado calculado almacenado en la variable 'integral'
                resultado = f'Integral de f(x) usando Simpson: {integral}'

                # Configurar el texto del QLabel con el resultado
                self.label_5S.setText(resultado)
                # Graficar la función y los trapecios
                colors = plt.cm.viridis(np.linspace(0, 1, n + 1))  # Colores basados en cmap 'viridis'
                x_vals = [result[2] for result in results]
                y_vals = [result[3] for result in results]
                plt.plot(x_vals, y_vals, label='f(x)')
                for i in range(n):
                    x_simpson = [results[i][2], results[i + 1][2]]
                    y_simpson = [results[i][3], results[i + 1][3]]
                    plt.fill_between(x_simpson, y_simpson, alpha=0.5, label='Simpson', color=colors[i])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.show()
            else:
                QMessageBox.warning(self, 'Error',
                                    'El valor de n debe ser 2 o un número par para el método de Simpson.')

                # Supongamos que tienes una lista de nombres para tus columnas, por ejemplo:
            nombres_columnas = ['n', 'h', 'x', 'f(x)', 'f(xm)']

                # Luego, configura los títulos de las columnas en tu QTableWidget
            self.tableWidget_S.setRowCount(len(results))
            self.tableWidget_S.setColumnCount(
                    len(nombres_columnas))  # Usar la longitud de nombres_columnas en lugar de results[0]

                # Configura los títulos de las columnas
            for j, nombre_columna in enumerate(nombres_columnas):
                    item = QTableWidgetItem(nombre_columna)
                    self.tableWidget_S.setHorizontalHeaderItem(j, item)

                # Llena la tabla con los datos de results
            for i, row in enumerate(results):
                for j, val in enumerate(row):
                        item = QTableWidgetItem(str(val))
                        self.tableWidget_S.setItem(i, j, item)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error al calcular la integral: {str(e)}')

    # ---------------------------------------------------------------

    # Falta  metodo para ingresar la funcion por calculadora
    def calculator(self):
        print('calculator')
        self.stackedWidget.setCurrentIndex(9)
        # Boton confirmar
        self.next_button.clicked.connect(self.puntoFijo_Panel)
        self.next_button.clicked.connect(self.secante_Panel)

    # Falta para mostrar la grafica de la funcion
    def mostrar_grafica(self):
        self.stackedWidget.setCurrentIndex(10)
        print('mostrar grafica')

    # Falta para mostrar la tabla con i,x,error
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

    def mostrar_tabla(self, datos):
        #self.stackedWidget.setCurrentIndex(11)
        print('mostrar tabla')
        datos_str = [[str(elemento) for elemento in registro] for registro in datos]
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

    def mostrar_tabla_biseccion(self, datos):
        # Mostrar tabla
        datos_str = [[str(elemento) for elemento in registro] for registro in datos]
        fila = 0
        for registro in datos_str:
            columna = 0
            self.bis_tabla.insertRow(fila)
            for elemento in registro:
                celda = QTableWidgetItem(elemento)
                self.bis_tabla.setItem(fila, columna, celda)
                columna += 1
            fila += 1
if __name__ == '__main__':
    app = QApplication(sys.argv)
    Calculadora = MainWindow()
    Calculadora.show()
    sys.exit(app.exec_())
