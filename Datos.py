# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np


class Datos:

    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):

        # Read file with pandas
        

        self.nominalAtributos = np.ones(len(file.dtypes), dtype=bool)
        # If the type of the column is a string / object the value in the array is True
        # If it is a float or integer it will be False
        # Otherwise we raise an exception
        for i in range(len(file.dtypes)):
            if file.dtypes[i] == object:
                pass
            elif file.dtypes[i] == np.int64 or file.dtypes[i] == np.float64:
                self.nominalAtributos[i] = False
            else:
                raise ValueError('Columns must be either nominal values, floats or integers')

        tmp_datos = file.to_numpy()
        # It will be a dict of dicts to store all the possible categorical variables from different fields
        # We implement it this so in case of collision of names, the weights will not be affected.
        self.diccionario = dict()

        # We iterate column per column
        contador = 0
        for j in range(tmp_datos.shape[1]):
            # If the column is nominal we get a set of its objects
            if self.nominalAtributos[j]:
                self.diccionario[file.columns[j]] = dict()
                unique = sorted(set(tmp_datos[:, j]))
                # Add the ordered set of words to the corresponding dictionary
                for u in unique:
                    if u not in self.diccionario:
                        self.diccionario[file.columns[j]][u] = contador
                        contador += 1

        # Replace dictionary values of nominal variables in datos
        self.datos = np.empty(tmp_datos.shape)
        for j in range(tmp_datos.shape[1]):
            # If not nominal the just copy the entire column
            if not self.nominalAtributos[j]:
                self.datos[:, j] = tmp_datos[:, j]
            # If nominal, go one by one extracting the value from the dictionary
            else:
                for i in range(tmp_datos.shape[0]):
                    self.datos[i, j] = self.diccionario[file.columns[j]][tmp_datos[i][j]]

    # This method will give back a matrix composed of the index passed by argument
    def extraeDatos(self, idx):
        matrix = np.empty((len(idx), self.datos.shape[1]))
        for id in range(len(idx)):
            matrix[id] = self.datos[idx[id]]
        return matrix

if __name__ == '__main__':
    datos = Datos('./ConjuntoDatos/tic-tac-toe.data')