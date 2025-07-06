#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Pablo Ramos
# Created Date: 04/07/2025
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from osgeo import gdal

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
    

# def scale(array, p=0):
    # '''
    # Escala un array 2D entre 0 y 1, usando como mínimo el percentil p y como máximo el percentil 100 - p.
    
    # Parámetros:
        # array : numpy array 2D
        # p     : percentil (entre 0 y 50). Si p = 0, se toma min y max reales del array.
    
    # Retorna:
        # scaled : array escalado entre 0 y 1

    # Se elige un maximo de p = 50 porque si
    
    # p = 60, por ejemplo, te daría:

    # vmin = percentil 60

    # vmax = percentil 40 → esto no puede ocurrir
    # '''

    # if not (0 <= p <= 50):
        
        # raise ValueError("El percentil p debe estar entre 0 y 50.")

    # vmin = np.percentile(array, p)
    # vmax = np.percentile(array, 100 - p)

    # if vmax == vmin:
        
        # scaled = np.zeros_like(array, dtype='float32')
        
    # else:
    
        # array_clipped = np.clip(array, vmin, vmax)
        
        # scaled = (array_clipped - vmin) / (vmax - vmin)

    # return scaled


def scale(array, p=0, nodata=None):
    '''
    Escala una imagen (array 2D o 1D) entre 0 y 1 usando percentiles.

    Parámetros:
        array : np.ndarray
            Imagen o banda que se desea escalar.
        p : float
            Porcentaje de recorte en los extremos (ej: p=2 usa percentiles 2 y 98).
        nodata : valor escalar (opcional)
            Valor que representa datos inválidos. Será excluido del cálculo.

    Retorna:
        np.ndarray escalado entre 0 y 1, con nodata preservado (si aplica).
    '''

    a = array.copy().astype('float32')
    mask_valid = np.ones_like(a, dtype=bool)

    if nodata is not None:
        mask_valid &= (a != nodata)
        a[~mask_valid] = np.nan

    a_valid = a[mask_valid & ~np.isnan(a)]
    if a_valid.size == 0:
        raise ValueError("No hay datos válidos para escalar.")

    vmin = np.percentile(a_valid, p)
    vmax = np.percentile(a_valid, 100 - p)

    if vmax == vmin:
        return np.zeros_like(a)

   
    a = np.clip(a, vmin, vmax)
    a = (a - vmin) / (vmax - vmin)

   
    if nodata is not None:
        a[~mask_valid] = np.nan

    return a


def scale_multiband(imagen, p=0, nodata=None):
    '''
    Escala cada banda del array multibanda [canales, alto, ancho]
    usando la función scale(). Devuelve el array escalado.
    
    Parámetros:
    - imagen: np.ndarray con forma (canales, alto, ancho)
    - p: percentil para escalar (ej: 2 para 2-98)
    - nodata: valor a excluir del cálculo

    Retorna:
    - imagen_escalada: array (canales, alto, ancho) escalado
    '''
    canales = imagen.shape[0]
    alto, ancho = imagen.shape[1], imagen.shape[2]

    imagen_escalada = np.zeros((canales, alto, ancho), dtype="float32")

    for i in range(canales):
    
        canal = imagen[i, :, :]
        
        imagen_escalada[i, :, :] = scale(canal, p=p, nodata=nodata)

    return imagen_escalada


def plot_rgb(imagen, band_list=[0,1,2], p=0, nodata=None):
    '''
    Plotea una imagen RGB a partir de un array multibanda y una lista de bandas.

    Parámetros:
    - imagen: array con forma (canales, alto, ancho)
    - band_list: lista de índices para R, G, B (ej: [3,2,1])
    - p: percentil para escalar
    - nodata: valor a excluir

    Acción:
    - Muestra la imagen RGB escalada usando matplotlib
    '''
    img_scaled = scale_multiband(imagen, p=p, nodata=nodata)
    rgb = np.array([img_scaled[band_list[0]],
                    img_scaled[band_list[1]],
                    img_scaled[band_list[2]]])
    rgb = rgb.transpose(1, 2, 0)

    plt.figure(figsize=(16, 8))
    plt.imshow(rgb)
    plt.title(f"Combinación RGB - bandas {band_list} - estiramiento {100 - 2*p}%")
    plt.axis('off')
    plt.show()


def subset_img(ds, ulx, uly, lrx, lry):
    
    gt = ds.GetGeoTransform()
    origin_x, pixel_width, _, origin_y, _, pixel_height = gt

    x_off = int((ulx - origin_x) / pixel_width)
    y_off = int((origin_y - uly) / abs(pixel_height))

    win_xsize = int((lrx - ulx) / pixel_width)
    win_ysize = int((uly - lry) / abs(pixel_height))

    if x_off < 0 or y_off < 0 or (x_off + win_xsize) > ds.RasterXSize or (y_off + win_ysize) > ds.RasterYSize:
        raise ValueError("El subset está fuera de los límites del raster.")

    subset = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_xsize, win_ysize)

    return subset
    

def write_raster_gdal(a, path, fname, fdriver, dtype, gt, src):
    '''
    Esta función está pensada para arreglos o matrices de 2 o 3 dimensiones, donde:
    Para 3 dimensiones:
    a.shape[0] = Numero de bandas
    a.shape[1] = Numero de filas
    a.shape[2] = Numero de columnas

    Para 2 dimensiones:
    a.shape[0] = Numero de filas
    a.shape[1] = Numero de columnas
    '''
    import os
    shape = a.shape
    driver  = gdal.GetDriverByName(fdriver)
    if len(a.shape) == 2:
        nband = 1
        out_image = driver.Create(os.path.join(path,fname ),shape[1],shape[0],nband,dtype)
        out_image.GetRasterBand(1).WriteArray(a)
    else:
        nband = shape[0]
        out_image = driver.Create(os.path.join(path,fname ),shape[2],shape[1],nband,dtype)
        for i in range(nband):
            out_image.GetRasterBand(i+1).WriteArray(a[i,:,:])
        
    out_image.SetGeoTransform(gt)
    out_image.SetProjection(src)
    
    del out_image