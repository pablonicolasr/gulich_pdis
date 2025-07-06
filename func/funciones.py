import os
from datetime import datetime
import numpy as np
from termcolor import colored
from osgeo import gdal
import matplotlib.pyplot as plt


def raiz(a, b=2):
    '''
    Esta función devuelve la raiz b de un número entero a positivo. 
    En caso de no definirse un valor para b, tomará por defecto 2.
    '''
    if isinstance(a, int) and isinstance(b, int) and a >= 0:
        res = a ** (1 / b)
    elif a < 0:
        res = None
        print(f'\n{a} debe ser un número positivo.')
    else:
        res = None
        print(f'\nLos parámetros {a} y {b} deben ser números enteros.')
    return res

def get_fechas(dir_in, ext='tif', verbose=True):
    '''
    Extrae la lista ordenada de fechas de los archivos MODIS 
    de una carpeta dada por dir_in.
    El resultado es una lista de objetos tipo datetime.
    '''
    files = os.listdir(dir_in)
    files_with_ext = [f for f in files if f.lower().endswith(ext.lower())]
    files_with_ext.sort()
    sfechas = [f.split('doy')[1].split('_')[0] for f in files_with_ext]
    fechas = [datetime.strptime(f, '%Y%j') for f in sfechas]
    
    if verbose:
        print(f'Recolecté {len(fechas)} fechas entre {fechas[0]} y {fechas[-1]}.')
    
    return fechas

def scale(array,p = 0, nodata = None):
    '''
    Esta función escala o estira la imagen a determinado % del histograma (trabaja con percentiles)
    Si p = 0 (valor por defecto) entonces toma el mínimo y máximo de la imagen.
    Devuelve un arreglo nuevo, escalado de 0 a 1
    '''
    a = array.copy()
    a_min, a_max = np.percentile(a[a!=nodata],p), np.percentile(a[a!=nodata],100-p)
    a[a<a_min]=a_min
    a[a>a_max]=a_max
    return ((a - a_min)/(a_max - a_min))

def scale_multiband(imagen, p = 0, nodata = None):  

    # Definir una variable con el número de canales
    canales = imagen.shape[0]
    imagen_escalada = np.empty_like(imagen, dtype = np.float32)

    for i in range(canales):
      canal = imagen[i]
      imagen_escalada[i, :, :] = scale(canal, p = p, nodata = nodata)

    return imagen_escalada
    
def plot_rgb(array,band_list, p = 0, nodata = None, figsize = (12,6)):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, una lista de índices correspondientes
    a las bandas que queremos usar, en el orden que deben estar (ej: [1,2,3]), y un parámetro
    p que es opcional, y por defecto es 0 (es el estiramiento a aplicar cuando llama a get_rgb(), que a su vez llama a scale()).
    
    Por defecto tambien asigna un tamaño de figura en (12,6), que también puede ser modificado.
    
    Devuelve solamente un ploteo, no modifica el arreglo original.
    Nota: Se espera una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    r = band_list[0]
    g = band_list[1]
    b = band_list[2]
           
    a = scale_multiband(array, p, nodata)
    a = a[[r,g,b]].transpose(1,2,0)
    
    plt.figure(figsize = figsize)
    plt.title(f'Combinación {r}, {g}, {b} \n (estirado al {p}%)' , size = 20)
    plt.imshow(a)
    plt.show()
    
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
