#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Pablo Ramos
# Created Date: 04/07/2025
import re
from datetime import datetime

def raiz(a, b=2):
    '''
    Esta función devuelve la raiz b de un número entero a positivo.
    En caso de no definirse un valor para b, tomará por defecto 2.

    parámetros:
    a = número entero positivo
    b = número entero

    retorna:
    raiz b de a
    '''

    if isinstance(a, int) and isinstance(b, int) and a>=0:
        res = a**(1/b)
    elif isinstance(a, int) and a < 0:
        res = None
        print(colored(f'\n{a} debe ser un entero positivo.', 'red'))
    else:
        res = None
        print(colored(f'\n{a} y {b} deben ser números enteros.', 'red'))

    return res



def get_dates(path_ndvi):

    fechas = []    

    for nombre in path_ndvi:
        
        match = re.search(r"doy(\d{7})", nombre)
        
        if match:
            
            fecha = datetime.strptime(match.group(1), "%Y%j").date()
            
            fechas.append(fecha)

    return sorted(fechas)

    