from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QDialog, QGridLayout, QLabel, \
    QMessageBox, QRadioButton
import sys
import sympy as sp

class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Calculadora')
        self.setWindowIcon(QIcon('python2.ico'))
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()

        # Agregar imagen de la calculadora
        pixmap = QPixmap("calculator.png")
        calculator_image = QLabel()
        calculator_image.setPixmap(pixmap)
        self.layout.addWidget(calculator_image)

        # Agregar etiqueta
        label = QLabel('Seleccione un método: ')
        self.layout.addWidget(label)

        # Lista vertical de métodos
        self.method_radio_buttons = []
        methods = ['Punto fijo', 'Bisección', 'Newton-Raphson', 'Secante', 'Diferencias divididas', 'Jacobi',
                   'Gauss-Seidel', 'Trapecio', 'Simpson']

        for method in methods:
            radio_button = QRadioButton(method)
            self.method_radio_buttons.append(radio_button)
            self.layout.addWidget(radio_button)

        # Botón de cálculo
        self.calculate_button = QPushButton('Calcular')
        self.calculate_button.clicked.connect(self.calculate)
        self.layout.addWidget(self.calculate_button)

        self.setLayout(self.layout)

    def calculate(self):
        selected_method = None

        for i, radio_button in enumerate(self.method_radio_buttons):
            if radio_button.isChecked():
                selected_method = i
                break

        if selected_method is None:
            QMessageBox.warning(self, 'Error', 'Por favor, seleccione un método.')
            return

        # Mostrar un mensaje con el método seleccionado
        selected_method_name = self.method_radio_buttons[selected_method].text()
        QMessageBox.information(self, 'Método seleccionado', f'Se seleccionó el método: {selected_method_name}')

        # Método punto fijo
        if selected_method == 0:
            print('hola')
            self.point_fix_dialog = PointFixDialog()
            self.point_fix_dialog.exec_()
            user_function = self.point_fix_dialog.get_function()

class PointFixDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ingresar función f(x)')
        self.setGeometry(200, 200, 400, 400)

        self.layout = QVBoxLayout()

        # Etiqueta
        label = QLabel('Ingrese f(x): ')
        self.layout.addWidget(label)

        # Cuadro de texto
        self.function_input = QLineEdit()
        self.layout.addWidget(self.function_input)

        # Cuadro de botones principales de la calculadora
        button_grid = QGridLayout()
        buttons = [
            'x', 'x^2', 'x^n', 'sqrt(x)',  # Usamos 'sqrt(x)' para la raíz cuadrada
            'sin', 'cos', 'π', 'e',
            'ln', '(', ')', '+',
            '-', '*', 'Derivada'  # Botón para calcular la derivada
        ]

        row, col = 0, 0

        for button_text in buttons:
            button = QPushButton(button_text)
            if button_text == '=':
                button.clicked.connect(self.calculate_function)
            elif button_text == 'Derivada':
                button.clicked.connect(self.calculate_derivative)
            else:
                button.clicked.connect(lambda checked, text=button_text: self.insert_text(text))
            button_grid.addWidget(button, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Botón para mostrar la función ingresada
        self.show_function_button = QPushButton('Mostrar Función')
        self.show_function_button.clicked.connect(self.show_function)
        button_grid.addWidget(self.show_function_button, row, col, 1, 2)  # Coloca el botón en la última fila

        self.layout.addLayout(button_grid)

        self.setLayout(self.layout)

    def insert_text(self, text):
        current_text = self.function_input.text()
        self.function_input.setText(current_text + text)

    def calculate_function(self):
        function_text = self.function_input.text()

        try:
            x = sp.symbols('x')
            f_x = sp.sympify(function_text)

            # Calcula la integral de f(x)
            integral = sp.integrate(f_x, x)

            # Muestra el resultado en un cuadro de mensaje
            QMessageBox.information(self, 'Resultado', f'Integral de f(x): {integral}')

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error al calcular la integral: {str(e)}')

    def calculate_derivative(self):
        function_text = self.function_input.text()

        try:
            x = sp.symbols('x')
            f_x = sp.sympify(function_text)

            # Calcula la derivada de f(x)
            derivative = sp.diff(f_x, x)

            # Muestra el resultado en un cuadro de mensaje
            QMessageBox.information(self, 'Resultado', f'Derivada de f(x): {derivative}')

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error al calcular la derivada: {str(e)}')

    def get_function(self):
        return self.function_input.text()

    def show_function(self):
        user_function = self.function_input.text()
        QMessageBox.information(self, 'Función Ingresada', f'f(x) = {user_function}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    calc = Calculator()
    calc.show()
    sys.exit(app.exec_())
