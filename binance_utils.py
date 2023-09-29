import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import matplotlib.pyplot as plt

def feature_backward_elimination(data, target, significance_level=0.05):
    """
    Realiza la selección hacia atrás con una regresión OLS robusta en el dataframe `data` 
    con la columna de objetivo `target`. Devuelve una lista de características incluidas
     y también una lista de características excluidas. 
    
    Args:
    - data (pd.DataFrame): DataFrame que contiene las variables independientes.
    - target (pd.Series): Serie que contiene la variable dependiente.
    - significance_level (float): Nivel de significancia para mantener una variable en el modelo.
    
    Returns:
    - list: Lista de variables significativas.
    - list: Lista de variables excluidas.
    - model: Modelo OLS con las variables seleccionadas.
    """
    
    features = data.columns.tolist()
    excluded_features = []
    while len(features) > 0:
        features_with_constant = sm.add_constant(data[features])

        # Ajustamos una regresión OLS robusta con HC1 para muestras grandes
        model = sm.OLS(target, features_with_constant).fit(cov_type='HC1')
        p_values = model.pvalues[1:]

        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            excluded_features.append(excluded_feature)
        else:
            break
    
    return features, excluded_features, model

#####

def remove_outliers(df, multiplier=1.5):
    """
    Elimina valores atípicos de un DataFrame basado en el método IQR.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada del cual se deben eliminar los valores atípicos.
    - multiplier (float): Multiplicador para el IQR para determinar los límites de los valores atípicos. Default es 1.5.

    Devoluciones:
    - pd.DataFrame: Un nuevo DataFrame sin los valores atípicos.

    Ejemplo:
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200]})
    >>> remove_outliers(df, multiplier=2.0)
       A   B
    0  1   6
    1  2   7
    2  3   8
    3  4   9
    4  5  10
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - multiplier * IQR)) | (df > (Q3 + multiplier * IQR))).any(axis=1)]
    return df_out

#####

def drop_high_vif_vars(df, threshold=5):
    """
    Calculate VIF for each feature in a dataframe and iteratively remove features with high VIF.
    
    Parameters:
    - df: DataFrame with features to calculate VIF.
    - threshold: VIF threshold to decide if a feature should be removed. Default is 5.
    
    Returns:
    - vif_df: DataFrame with features and their VIF values.
    - removed_features: List of features removed due to high VIF.
    """
    # Initialize variables
    variables = df.columns
    vif_df = pd.DataFrame()
    removed_features = []
    
    # Calcula VIF de forma iterativa y remueve el VIF mayor si es mayor a 5:
    while True:
        vif_data = [variance_inflation_factor(df[variables].values, i) if np.linalg.det(df[variables].corr()) != 0 else float('inf') for i in range(df[variables].shape[1])]
        max_vif = max(vif_data)
        max_vif_feature = variables[vif_data.index(max_vif)]
        
        if max_vif > threshold:
            removed_features.append(max_vif_feature)
            variables = variables.drop(max_vif_feature)
        else:
            break
    
    # Crea DataFrame VIF:
    vif_df["low_vif_features"] = variables
    vif_df["VIF"] = [variance_inflation_factor(df[variables].values, i) if np.linalg.det(df[variables].corr()) != 0 else float('inf') for i in range(df[variables].shape[1])]
    
    return vif_df, removed_features

#####

def select_features_from_regularization(df, lasso_threshold=0.01, ridge_threshold=0.01):
    """
    Esta función selecciona las características basadas en los coeficientes de regularización de Lasso y Ridge.
    
    Parámetros:
    - df: DataFrame que contiene los coeficientes de Lasso y Ridge para cada característica.
    - lasso_threshold: Umbral para los coeficientes de Lasso.
    - ridge_threshold: Umbral para los coeficientes de Ridge.
    
    Devuelve:
    - Una lista de nombres de características seleccionadas.
    """
    
    # Seleccionar características basadas en Lasso
    lasso_selected = df[df['lasso_coefs'].abs() > lasso_threshold]['X'].tolist()
    
    # Seleccionar características basadas en Ridge
    ridge_selected = df[df['ridge_coefs'].abs() > ridge_threshold]['X'].tolist()
    
    # Intersectar las listas para obtener características que son importantes tanto para Lasso como para Ridge
    selected_features = list(set(lasso_selected) & set(ridge_selected))
    
    return selected_features

#####

def find_sorted_large_time_gaps(index, threshold_seconds):
    """
    Encuentra las brechas de tiempo que exceden un cierto umbral en segundos y las ordena de mayor a menor.
    
    Parameters:
    -----------
    index : pd.DatetimeIndex
        Índice de tipo datetime.
    threshold_seconds : int
        Umbral en segundos para considerar una brecha como "grande".
    
    Returns:
    --------
    list of tuples
        Lista ordenada de tuplas donde cada tupla representa el inicio y fin de una brecha que excede el umbral.
    """
    # Calcular las diferencias entre tiempos consecutivos
    time_diffs = pd.Series(index).diff().dropna()
    
    # Identificar las brechas que exceden el umbral
    large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=threshold_seconds)]
    
    # Crear una lista de tuplas con el inicio y fin de cada brecha
    gaps_list = [(index[i - 1], index[i]) for i in large_gaps.index]
    
    # Ordenar la lista de brechas de mayor a menor
    gaps_list.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    
    return gaps_list

#####

def add_nan_rows_for_large_gaps(df, threshold_seconds, nan_frequency_seconds=1):
    """
    Agrega filas NaN en un DataFrame para brechas temporales grandes.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con un índice de tipo datetime.
    threshold_seconds : int
        Umbral en segundos para identificar una brecha temporal grande.
    nan_frequency_seconds : int, opcional (por defecto es 1)
        Frecuencia en segundos con la que se agregarán filas NaN dentro de las brechas identificadas.

    Retorno:
    --------
    df_filled : pd.DataFrame
        DataFrame con filas NaN agregadas en brechas temporales grandes.

    Ejemplo:
    --------
    >>> df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01', '2023-01-01 00:01:00', '2023-01-01 00:01:02']))
    >>> df_filled = add_nan_rows_for_large_gaps(df, threshold_seconds=30, nan_frequency_seconds=5)
    """

    # Calcular las diferencias entre tiempos consecutivos
    time_diffs = df.index.to_series().diff()

    # Identificar las brechas que son mayores al umbral
    large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=threshold_seconds)]

    # Redondear nan_frequency_seconds al entero más cercano
    rounded_seconds = round(nan_frequency_seconds)
    freq_str = f"{rounded_seconds}S"

    # Para cada brecha identificada, agregar filas NaN con la frecuencia especificada
    for start, gap in large_gaps.items():
        gap_range = pd.date_range(start=start + pd.Timedelta(seconds=1), 
                                  end=start + gap - pd.Timedelta(seconds=1), 
                                  freq=freq_str)
        gap_df = pd.DataFrame(index=gap_range)
        df = pd.concat([df, gap_df])

    # Ordenar el DataFrame resultante por índice
    df_filled = df.sort_index()

    return df_filled

#####

def find_large_time_gaps(index, threshold_seconds):
    """
    Encuentra las brechas de tiempo que exceden un cierto umbral en segundos y las ordena de mayor a menor.
    
    Parameters:
    -----------
    index : pd.DatetimeIndex
        Índice de tipo datetime.
    threshold_seconds : int
        Umbral en segundos para considerar una brecha como "grande".
    
    Returns:
    --------
    list of tuples
        Lista ordenada de tuplas donde cada tupla representa el inicio y fin de una brecha que excede el umbral.
    """
    # Calcular las diferencias entre tiempos consecutivos
    time_diffs = pd.Series(index).diff().dropna()
    
    # Identificar las brechas que exceden el umbral
    large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=threshold_seconds)]
    
    # Crear una lista de tuplas con el inicio y fin de cada brecha
    gaps_list = [(index[i - 1], index[i]) for i in large_gaps.index]
    
    # Ordenar la lista de brechas de mayor a menor
    gaps_list.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    
    return gaps_list

#####

def add_nan_rows_for_large_gaps(df, threshold_seconds, nan_frequency_seconds=1):
    """
    Agrega filas NaN en un DataFrame para brechas temporales grandes.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con un índice de tipo datetime.
    threshold_seconds : int
        Umbral en segundos para identificar una brecha temporal grande.
    nan_frequency_seconds : int, opcional (por defecto es 1)
        Frecuencia en segundos con la que se agregarán filas NaN dentro de las brechas identificadas.

    Retorno:
    --------
    df_filled : pd.DataFrame
        DataFrame con filas NaN agregadas en brechas temporales grandes.

    Ejemplo:
    --------
    >>> df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01', '2023-01-01 00:01:00', '2023-01-01 00:01:02']))
    >>> df_filled = add_nan_rows_for_large_gaps(df, threshold_seconds=30, nan_frequency_seconds=5)
    """

    # Calcular las diferencias entre tiempos consecutivos
    time_diffs = df.index.to_series().diff()

    # Identificar las brechas que son mayores al umbral
    large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=threshold_seconds)]

    # Redondear nan_frequency_seconds al entero más cercano
    rounded_seconds = round(nan_frequency_seconds)
    freq_str = f"{rounded_seconds}S"

    # Para cada brecha identificada, agregar filas NaN con la frecuencia especificada
    for start, gap in large_gaps.items():
        gap_range = pd.date_range(start=start + pd.Timedelta(seconds=1), 
                                  end=start + gap - pd.Timedelta(seconds=1), 
                                  freq=freq_str)
        gap_df = pd.DataFrame(index=gap_range)
        df = pd.concat([df, gap_df])

    # Ordenar el DataFrame resultante por índice
    df_filled = df.sort_index()

    return df_filled

#####

def find_nearest_past_time_index(target, series, max_diff_seconds=5):
    """
    Encuentra el índice más cercano a 'target' en 'series' dentro de un rango de 'max_diff_seconds', 
    comenzando por el día inmediatamente anterior a 'target'.
    
    Parameters:
    -----------
    target : datetime
        Fecha y hora objetivo.
    series : pd.Series
        Serie temporal con índice de tipo datetime.
    max_diff_seconds : int, optional
        Diferencia máxima permitida en segundos.
    
    Returns:
    --------
    datetime or None
        Índice más cercano o None si no se encuentra.
    """
    # Convertir max_diff_seconds a Timedelta
    max_diff = pd.Timedelta(seconds=max_diff_seconds)
    
    # Calcular la diferencia en segundos
    target_seconds = target.hour * 3600 + target.minute * 60 + target.second
    
    # Comenzar la búsqueda en el día inmediatamente anterior
    days_back = 1
    while days_back <= (target - series.index.min()).days:
        filtered_series = series[series.index.date == (target - pd.Timedelta(days=days_back)).date()]
        
        if not filtered_series.empty:
            series_seconds = filtered_series.index.hour * 3600 + filtered_series.index.minute * 60 + filtered_series.index.second
            diff = np.abs(series_seconds - target_seconds)
            
            # Verificar si la diferencia mínima es menor o igual a la diferencia máxima permitida
            if diff.min() <= max_diff.total_seconds():
                return filtered_series.index[np.argmin(diff)]
        
        # Si no se encuentra un valor adecuado, buscar en el siguiente día anterior
        days_back += 1
    
    return None

#####



#####

def plot_different_color_within_range(series, start_date, end_date, main_color='blue', highlight_color='red'):
    """
    Grafica una serie de tiempo y colorea los datos dentro de un rango específico de fechas con un color diferente.
    
    Parámetros:
    - series: Serie de tiempo a graficar.
    - start_date: Fecha de inicio del rango a colorear.
    - end_date: Fecha de finalización del rango a colorear.
    - main_color: Color principal para la serie.
    - highlight_color: Color para los datos dentro del rango especificado.
    """
    # Graficar
    plt.figure(figsize=(15, 6))
    
    # Datos fuera del rango
    mask_outside = (series.index < start_date) | (series.index > end_date)
    plt.plot(series.index[mask_outside], series[mask_outside], label='Datos originales', color=main_color)
    
    # Datos dentro del rango
    mask_inside = (series.index >= start_date) & (series.index <= end_date)
    plt.plot(series.index[mask_inside], series[mask_inside], label='Datos en rango coloreado', color=highlight_color)

#####

def mark_rows_with_outliers(df, multiplier=1.5):
    """
    Marca las filas que contienen al menos un valor atípico en cualquiera de sus columnas.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada en el que se buscarán valores atípicos.
    - multiplier (float): Multiplicador para el IQR para determinar los límites de los valores atípicos. Default es 1.5.

    Devoluciones:
    - pd.Series: Una serie con valores 1 para filas con valores atípicos y 0 para filas sin valores atípicos.

    Ejemplo:
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200]})
    >>> df['is_outlier'] = mark_rows_with_outliers(df, multiplier=2.0)
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Marcar las filas que contienen al menos un valor atípico
    outlier_mask = ((df < (Q1 - multiplier * IQR)) | (df > (Q3 + multiplier * IQR))).any(axis=1)
    
    return outlier_mask.astype(int)

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

from scipy.stats import mstats

def apply_windsorizing(series, multiplier=1.5):
    """
    Aplica windsorizing a una serie para limitar los valores atípicos.

    Parámetros:
    - series (pd.Series): La serie a la que se le aplicará windsorizing.
    - multiplier (float): Multiplicador para el IQR para determinar los límites de los valores atípicos.

    Devoluciones:
    - pd.Series: Serie con valores atípicos limitados.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return mstats.winsorize(series, limits=[(series < lower_bound).mean(), (series > upper_bound).mean()])