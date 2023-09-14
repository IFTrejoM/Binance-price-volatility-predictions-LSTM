import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

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

def remove_outliers(df):
    """
    Elimina valores atípicos de un DataFrame basado en el método IQR.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada del cual se deben eliminar los valores atípicos.

    Devoluciones:
    - pd.DataFrame: Un nuevo DataFrame sin los valores atípicos.

    Ejemplo:
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200]})
    >>> remove_outliers(df)
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
    df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
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