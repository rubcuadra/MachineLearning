Ruben Cuadra A01019102
El proyecto consta de 11 funciones y 1 main, el codigo requiere tener instaladas las librerias numpy y matplot. El codigo esta diseñado para python2.7    

    getDataFromFile - Recibe el path a un archivo csv y nos regresa los datos como matrices numpy (Checar archivo ejemplo para ver formato)
    net - Funcion net usada para entrenar Adalines, requiere un vector X y un vector de pesos
    netE - Funcion escalon usada para entrenar Perceptrones, requiere un vector X y un vector de pesos
    entrenaPerceptron() - Recibe una matrix X y un vector Y, opcionalmente recibe un vector de pesos, un margen de error aceptable, iteraciones maximas, alpha - razon de aprendizaje, nos devuelve un vector W que representa los pesos calculados por un perceptron para esos valores X,Y
    entrenaPerceptron2() - Lo mismo que la funcion perceptron pero ademas de devolver W nos devuelve un arreglo con errores calculados por cada iteracion, utiles para graficar
    entrenaAdaline() - Recibe una matrix X y un vector Y, opcionalmente recibe un vector de pesos, un margen de error aceptable, alpha - razon de aprendizaje, nos devuelve un vector W que representa los pesos calculados por un Adaline para esos valores X,Y
    entrenaAdaline2() - Lo mismo que la funcion Adaline pero ademas de devolver W nos devuelve un arreglo con errores calculados por cada iteracion, utiles para graficar
    predicePerceptron() - Recibe un vector X y uno W de pesos, nos devuelve una Y estimada con esos vectores
    prediceAdaline() - Recibe un vector X y uno W de pesos, nos devuelve una Y estimada con esos vectores
    graficaErrores() - Recibe un vector de errores y lo grafica

