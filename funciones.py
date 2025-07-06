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
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon
from shapely.geometry import shape
import geopandas as gpd

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
    
   
def nequalize(array, p=2, nodata=None):
    """
    Esta función es similar a scale, solo que funciona tanto para matrices de 2 y 3 dimensiones.
    En el caso de 3 dimensiones, devuelve una matriz con la estructura (banda, fila, columna) para
    utilizarla en show() de Rasterio.
    """
    if len(array.shape)==2:
    
        vmin,vmax=np.percentile(array[array!=nodata],(p,100-p))
        
        eq_array = (array-vmin)/(vmax-vmin)
        
        eq_array = np.clip(eq_array,0,1)
        
    elif len(array.shape)==3:
    
        eq_array = np.empty_like(array, dtype=float)
        
        for i in range(array.shape[0]):
        
            eq_array[i]=nequalize(array[i], p=p, nodata=nodata)
            
    return eq_array
    

def guardar_GTiff(fn,crs,transform,a,nodata = 0, dtype = np.float32):

        if len(a.shape)==2:
        
            count=1
            
        else:
            count=a.shape[0]
            
        with rasterio.open(
        fn,
        'w',
        driver='GTiff',
        height=a.shape[-2],
        width=a.shape[-1],
        count=count,
        dtype=dtype,
        nodata = nodata,
        crs=crs,
        transform=transform) as dst:
            if len(a.shape)==2:
                dst.write(a.astype(dtype), 1)
            else:
                for b in range(0,count):
                    dst.write(a[b].astype(dtype), b+1)
                    

def raster_to_vector(img_name, value_field='value', output=None):
    """
    Convierte un raster a un GeoDataFrame vectorial.

    Parámetros:
    - img_name: ruta del raster de entrada.
    - value_field: nombre del campo para los valores del raster.
    - output: (opcional) ruta del archivo vectorial a guardar (.shp, .geojson, .gpkg...).

    Retorna:
    - gdf: GeoDataFrame con las geometrías vectorizadas.
    """
    
    with rasterio.open(img_name) as img:
    
        metadata = img.meta
        
        transform = metadata['transform']
        
        crs = metadata['crs']
        
        nodata = img.nodata
        
        array = img.read(1)

        mask = array != nodata if nodata is not None else None

        features = list(shapes(array, mask=mask, transform=transform))
        geoms = [shape(geom) for geom, val in features]
        values = [val for geom, val in features]

        gdf = gpd.GeoDataFrame({value_field: values}, geometry=geoms)
        gdf.set_crs(crs, inplace=True)

        if output:
        
            gdf.to_file(output, index=False)

        return gdf


def porcentaje_area_por_clase(clasif_vector, codigo_clase, poligonos, id_col='link', clase_col='value'):
    """
    Calcula el porcentaje de área que una clase ocupa en cada polígono de entrada.

    Parámetros:
    -----------
    clasif_vector : GeoDataFrame
        Capa vectorial clasificada con una columna que identifica clases (por ejemplo: 'value').

    codigo_clase : int
        Código de la clase a analizar (ej: 4 para construido, 3 para bosque, etc).

    poligonos : GeoDataFrame
        Capa vectorial sobre la cual calcular el porcentaje de cobertura.

    id_col : str, default='link'
        Columna de identificación única del polígono (por ejemplo: ID o código de radio censal).

    clase_col : str, default='value'
        Columna de clase dentro de la clasificación vectorial.

    Retorna:
    --------
    GeoDataFrame con columna nueva: 'area_m2' y 'pct_area_clase'.
    """

    if clasif_vector.crs != poligonos.crs:
        poligonos = poligonos.to_crs(clasif_vector.crs)

    clase = clasif_vector[clasif_vector[clase_col] == codigo_clase]

    inter = gpd.overlay(poligonos, clase, how='intersection')

    inter['area_interseccion'] = inter.geometry.area

    poligonos['area_total'] = poligonos.geometry.area
    
    area_por_id = inter.groupby(id_col)['area_interseccion'].sum().reset_index()

    result = poligonos.merge(area_por_id, on=id_col, how='left')

    result['area_interseccion'] = result['area_interseccion'].fillna(0)
    
    result['pct_area_clase'] = (result['area_interseccion'] / result['area_total']) * 100

    return result