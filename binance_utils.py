import numpy as np
import pandas as pd
import nolds

#####

def apply_outlier_capping(series, multiplier=1.5):
    """
    Aplica capping a los valores atípicos de una serie.

    Parámetros:
    - series (pd.Series): Serie de entrada.
    - multiplier (float): Multiplicador para el IQR para determinar los límites de capping. Default es 1.5.

    Devuelve:
    - pd.Series: Serie con capping aplicado.
    """
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # Definir límites
    upper_limit = Q3 + multiplier * IQR
    lower_limit = Q1 - multiplier * IQR

    # Aplicar capping
    series_capped = np.where(series > upper_limit, upper_limit, 
                             np.where(series < lower_limit, lower_limit, series))
    
    return pd.Series(series_capped)

#####

def cross_validated_lle(data, n=10000, cv=5, emb_dim=10, min_tsep=250, lag=1052, seed=None):
    """
    Realiza una validación cruzada del Mayor Exponente de Lyapunov (LLE) en una serie temporal.
    
    Parámetros:
    - data: Serie temporal para analizar.
    - n: Número de observaciones a seleccionar en cada iteración de validación cruzada.
    - cv: Número de segmentos en los que dividir los datos.
    - emb_dim: Dimensión del embedding para el cálculo del LLE.
    
    Retorna:
    - lles: Lista de LLE calculados en cada iteración de validación cruzada.
    """
    lles = []
    
    # Asegurarse de que los datos tienen al menos n puntos
    assert len(data) >= n, "Not enough data points for the specified n"
    
    # Establecer la semilla para la generación de números aleatorios
    np.random.seed(seed)
    
    for _ in range(cv):
        # Selecciona un índice de inicio aleatorio para el segmento de datos
        start_idx = np.random.randint(0, len(data) - n + 1)
        
        # Selecciona un segmento de datos de tamaño n
        segment = data[start_idx:start_idx+n]
        
        # Calcula el LLE para el segmento y lo añade a la lista de LLEs
        lle = nolds.lyap_r(segment, emb_dim=emb_dim, min_tsep=min_tsep, lag=lag)
        lles.append(lle)
        
        # print(f"LLE for segment starting at index {start_idx}: {lle}")
    
    return lles

#####

def cross_validated_dfa(data, n=10000, cv=5, order=1, seed=None):
    """
    Realiza una validación cruzada del Detrended Fluctuation Analysis (DFA) en una serie temporal.
    
    Parámetros:
    - data: Serie temporal para analizar.
    - n: Número de observaciones a seleccionar en cada iteración de validación cruzada.
    - cv: Número de segmentos en los que dividir los datos.
    - order: Orden del polinomio utilizado para detrendar los datos en el DFA.
    - seed: Semilla para la generación de números aleatorios.
    
    Retorna:
    - alphas: Lista de exponentes de escala (alfa) calculados en cada iteración de validación cruzada.
    """
    alphas = []
    
    # Asegurarse de que los datos tienen al menos n puntos
    assert len(data) >= n, "Not enough data points for the specified n"
    
    # Establecer la semilla para la generación de números aleatorios
    np.random.seed(seed)
    
    for _ in range(cv):
        # Selecciona un índice de inicio aleatorio para el segmento de datos
        start_idx = np.random.randint(0, len(data) - n + 1)
        
        # Selecciona un segmento de datos de tamaño n
        segment = data[start_idx:start_idx+n]
        
        # Calcula el alfa (exponente de escala) para el segmento y lo añade a la lista de alfas
        alfa = nolds.dfa(segment, order=order)
        alphas.append(alfa)
        
        # print(f"Alfa for segment starting at index {start_idx}: {alfa}")
    
    return alphas