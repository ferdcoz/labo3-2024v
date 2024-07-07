import os
import pandas as pd
import numpy as np
import calendar
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from datetime import datetime

def crear_directorios_xp(EXPERIMENTO, DIRECTORIO="../EXPERIMENTOS/"):
    MODELOS = os.path.join(DIRECTORIO, EXPERIMENTO, 'Saved Models')
    PLOTS = os.path.join(DIRECTORIO, EXPERIMENTO, 'Plots')
    PREDS = os.path.join(DIRECTORIO, EXPERIMENTO, 'Predictions')
    NB = os.path.join(DIRECTORIO, EXPERIMENTO, 'Notebook')

    # Crear los directorios si no existen
    for directory in [MODELOS, PLOTS, PREDS, NB]:
        os.makedirs(directory, exist_ok=True)

    return MODELOS, PLOTS, PREDS, NB


def cargar_datasets(filtrado=True):
    productos = "../Datasets/tb_productos_descripcion.txt"
    stocks = "../Datasets/tb_stocks.txt.gz"
    sells = "../Datasets/sell-in.txt.gz"
    prod_filtro = "../Datasets/productos_a_predecir.txt"

    df_productos = pd.read_csv(productos, delimiter='\t')
    df_stocks = pd.read_csv(stocks, compression='gzip', delimiter='\t')
    df_sells = pd.read_csv(sells, compression='gzip', delimiter='\t')
    prods_filtro = pd.read_csv(prod_filtro)
    
    if filtrado == True:
        df_sells_filtrado = df_sells[df_sells['product_id'].isin(prods_filtro['product_id'].to_list())]
        df_prods_filtrado = df_productos[df_productos['product_id'].isin(prods_filtro['product_id'].to_list())]
        df_stocks_filtrado = df_stocks[df_stocks['product_id'].isin(prods_filtro['product_id'].to_list())]
    else:
        return df_sells, df_productos, df_stocks
    
    return df_sells_filtrado, df_prods_filtrado, df_stocks_filtrado

def feature_engineering(df, l=12):
    df = df.sort_values(by=['product_id', 'periodo'])
    
    if l==36:
        lags = list(range(1, l))
        d_lags = list(range(1, l-1))
         # ----- Lags (meses) -----
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('product_id')['tn'].shift(lag)
            
        # ----- Delta lags (meses) ----- 
        for lag in d_lags:    
            df[f'delta_lag_{lag}'] = df[f'lag_{lag}'] - df[f'lag_{lag}'].shift(1)
        
        # ----- Medias moviles ----- 
        for window in lags:
            df[f'moving_avg_{window}'] = df.groupby(['product_id'])['tn'].transform(lambda x: x.shift().rolling(window=window, min_periods=window).mean())
            
    else:
        lags = list(range(1, l+1))
        
        # ----- Lags (meses) -----
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('product_id')['tn'].shift(lag)

        # ----- Delta lags (meses) ----- 
        for lag in lags:    
            df[f'delta_lag_{lag}'] = df[f'lag_{lag}'] - df[f'lag_{lag}'].shift(1)
            
        # ----- Medias moviles ----- 
        for window in lags:
            df[f'moving_avg_{window}'] = df.groupby(['product_id'])['tn'].transform(lambda x: x.shift().rolling(window=window, min_periods=window).mean())
        
    # ----- Ratios ----- 
    df['ratio_3'] = df['tn'] / df['moving_avg_3']
    df['ratio_6'] = df['tn'] / df['moving_avg_6']

    # ----- Max/Min in X months -----
    for i in range(2, l+1):
        df[f'max_{i}_months'] = (df['tn'] == df['tn'].rolling(window=i, min_periods=i).max()).astype(int)
        df[f'min_{i}_months'] = (df['tn'] == df['tn'].rolling(window=i, min_periods=i).min()).astype(int)
        
    # ----- Quarter ----- 
    def fiscal_quarter(month):
        if month in [7, 8, 9]:
            return 1
        elif month in [10, 11, 12]:
            return 2
        elif month in [1, 2, 3]:
            return 3
        else:
            return 4

    # ----- Temporales ----- 
    df['periodo'] = df['periodo'].astype(str)
    df['Year'] = df['periodo'].str[:4].astype(int)
    df['Month'] = df['periodo'].str[4:6].astype(int)
    df['Q'] = df['Month'].apply(fiscal_quarter)
    df['Days_Month'] = df.apply(lambda row: calendar.monthrange(row['Year'], row['Month'])[1], axis=1)

    df['periodo'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')

    # ----- Continuidad ----- 
    df['month_sin'] = np.sin(2 * np.pi * df['periodo'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['periodo'].dt.month / 12)
    
    return df

def asignar_familia(group):
    for i in range(len(group)):
        current_size = group.iloc[i]['sku_size']
        current_id = group.iloc[i]['product_id']
        
        mayor = group[group['sku_size'] > current_size]['product_id'].tolist()
        menor = group[group['sku_size'] < current_size]['product_id'].tolist()
        melli = group[(group['sku_size'] == current_size) & (group['product_id'] != current_id)]['product_id'].tolist()
        
        group.at[group.index[i], 'hermano_mayor'] = mayor
        group.at[group.index[i], 'hermano_menor'] = menor
        group.at[group.index[i], 'mellizo'] = melli if melli else None
    return group

def producto_canario(df):
    
    periodos = df['periodo'].unique()
    
    datos_ficticios = {
        'periodo': periodos,
        'product_id': 99999,
        'plan_precios_cuidados': 0.0,
        'cat1': 'HC',
        'cat2': 'Artificial',
        'cat3': 'Artificial',
        'brand': 'Tweety',
        'descripcion': 'Alpiste Canarito',
        'sku_size': 3000.0,
        'cust_request_tn': [0] * len(periodos),
        'tn': [0] * len(periodos),
        'stock_final': [0] * len(periodos)
    }

    df_ficticio = pd.DataFrame(datos_ficticios)

    for periodo in periodos:
        valores_producto = df[(df['periodo'] == periodo) & (df['product_id'] == 20001)]
        if not valores_producto.empty:
            df_ficticio.loc[df_ficticio['periodo'] == periodo, 'cust_request_tn'] = valores_producto['cust_request_tn'].values[0] * 10
            df_ficticio.loc[df_ficticio['periodo'] == periodo, 'cust_request_qty'] = valores_producto['cust_request_qty'].values[0] * 10
            df_ficticio.loc[df_ficticio['periodo'] == periodo, 'stock_final'] = valores_producto['stock_final'].values[0] * 10
            df_ficticio.loc[df_ficticio['periodo'] == periodo, 'tn'] = valores_producto['tn'].values[0] * 10

    df_con_alpiste = pd.concat([df, df_ficticio], ignore_index=True)
    
    return df_con_alpiste

def scaler_std(df, columns_to_scale):
    scaler_dict = {}
    data_transformed = []

    for product_id in df['product_id'].unique():
        
        product_data = df[df['product_id'] == product_id]
        
        y_series = product_data['tn'].values.reshape(-1, 1)
        
        # Crear el scaler y ajustarlo a los datos
        scaler_y = StandardScaler()
        scaler_X = StandardScaler()
        
        scaled_time_series = scaler_y.fit_transform(y_series)
        
        # Almacenar los valores de media y desviación estándar para este producto
        scaler_dict[product_id] = (scaler_y.mean_[0], scaler_y.scale_[0])
        
        # Crear un DataFrame para los datos transformados y agregar el product_id y demás columnas
        transformed_df = product_data.copy()
        transformed_df['tn'] = scaled_time_series
        
        # Opción 1: vector de pesos tal cual
        # transformed_df['weights'] = product_data['tn'] # Agrego el vector de pesos
        
        # Opción 2: vector de pesos promedio 2019
        transformed_df['weights'] = product_data[product_data['periodo'] > 201812]['tn'].mean()
        
        # Itero y transformo cada una de las columnas
        for col in columns_to_scale:
            ts = product_data[col].values.reshape(-1, 1)
            st = scaler_X.fit_transform(ts)
            transformed_df[col] = st
    
        # Agregar los datos transformados a la lista
        data_transformed.append(transformed_df)

    # Combinar todos los DataFrames transformados en uno solo
    df_transformed = pd.concat(data_transformed)
    
    df_transformed.sort_values(by=['product_id', 'periodo'], inplace=True)
    
    return df_transformed, scaler_dict

def pipeline_fit_with_eval_set(pipeline, X_train, y_train, X_test, y_test, X_predict, fit_params={}):
    """
    Fit a scikit-learn pipeline with eval_set support.

    Parameters:
    - pipeline: The scikit-learn pipeline.
    - X_train: Training data.
    - y_train: Training labels.
    - X_test: Test data.
    - y_test: Test labels.
    - fit_params: Additional fit parameters.
    - pipeline_model_step_name: Name of the model step in the pipeline.

    Usage:
    pipeline_fit_with_eval_set(my_pipeline, X_train, y_train, X_test, y_test, fit_params={'eval_metric': 'logloss'})
    """
    # Step 1: Extract Preprocessors
    pipeline_preprocessors = Pipeline(pipeline.steps[:-1])
    
    # Step 2: Fit preprocessors and Transform Training Data
    # Make sure not to use any test data for the fit step
    X_train_transformed = pipeline_preprocessors.fit_transform(X_train)

    # Step 3: Transform Test Data
    X_test_transformed = pipeline_preprocessors.transform(X_test)
    X_predict_transformed = pipeline_preprocessors.transform(X_predict)

    # Step 4: Prepare Eval Set
    fit_params["eval_set"] = [(X_train_transformed, y_train), (X_test_transformed, y_test)]

    # Step 5: Extract Model and Fit
    model = pipeline.steps[-1][1]
    model.fit(X_train_transformed, y_train, **fit_params, verbose=500)
    return X_predict_transformed, model

def pipeline_fit_with_eval_set_weights(pipeline, X_train, y_train, X_test, y_test, X_predict, w_train=None, fit_params={}):
    # Step 1: Extract Preprocessors
    pipeline_preprocessors = Pipeline(pipeline.steps[:-1])
    
    # Step 2: Fit preprocessors and Transform Training Data
    X_train_transformed = pipeline_preprocessors.fit_transform(X_train)

    # Step 3: Transform Test Data
    X_test_transformed = pipeline_preprocessors.transform(X_test)
    X_predict_transformed = pipeline_preprocessors.transform(X_predict)

    # Step 5: Prepare Eval Set
    #evals = [(dtrain, 'train'), (dtest, 'eval')]
    fit_params["eval_set"] = [(X_train_transformed, y_train), (X_test_transformed, y_test)]

    # Step 6: Extract Model and Fit
    model = pipeline.steps[-1][1]
    model.fit(X_train_transformed,
              y_train,
              sample_weight=w_train,
              #eval_set=evals,
              verbose=100,
              **fit_params)
       
    return X_predict_transformed, model

def invert_transformation(scaled_value, product_id, scaler_dict):
    mean_val, std_val = scaler_dict[product_id]
    original_value = scaled_value * std_val + mean_val
    return original_value


def plot_top_features_importance(importance_df, top_x, EXPERIMENTO, PLOTS):
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    top_importance_df = importance_df.head(top_x)
    
    plt.figure(figsize=(9, 10))
    plt.barh(top_importance_df['Feature'], top_importance_df['Importance'], color='purple')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'XGBoost - Top {top_x} predictores relevantes')
    plt.gca().invert_yaxis()
    
    #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #plt.text(0.5, -0.1, f'Timestamp: {timestamp}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(1, -0.1, f'Timestamp: {timestamp}', ha='right', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.text(0, -0.1, f'Experimento: {EXPERIMENTO}', ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
    
    plt.savefig(os.path.join(PLOTS, f'fetaure_importance_top_{top_x}_{EXPERIMENTO}_bis.png'), bbox_inches='tight')
    plt.show()

def plot_predictions(df, PLOTS, EXPERIMENTO, all):
    if all is not True:
        productos = [20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012]
    else:
        productos = df['product_id'].unique()

    # Convertir 'periodo' al último día del mes
    df['periodo'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')

    # Crear una carpeta para guardar los gráficos
    output_dir = PLOTS + "/preds/"
    os.makedirs(output_dir, exist_ok=True)

    # Crear los gráficos
    for product_id in productos:
        product_df = df[df['product_id'] == product_id]
        
        # Separar training y predicciones
        training_df = product_df[product_df['periodo'] < '2020-01-01']
        prediction_df = product_df[product_df['periodo'] >= '2020-01-01']
        
        plt.figure(figsize=(10, 6))
        
        # Plot de training
        plt.plot(training_df['periodo'], training_df['tn'], label='Entrenamiento', color='blue', marker='o')
        
        # Plot de predicciones
        plt.plot(prediction_df['periodo'], prediction_df['tn'], label='Predicción', color='orange', marker='o')
        
        # Conectar el último punto de training con el primer punto de predicción
        if not training_df.empty and not prediction_df.empty:
            plt.plot([training_df['periodo'].iloc[-1], prediction_df['periodo'].iloc[0]],
                    [training_df['tn'].iloc[-1], prediction_df['tn'].iloc[0]], color='blue')
        
        # Línea punteada para marcar el inicio de las predicciones
        plt.axvline(x=pd.to_datetime('2019-12-01') + pd.offsets.MonthEnd(0), color='grey', linestyle='--', linewidth=0.9)
        
        # Mostrar el valor de la predicción para febrero 2020 si existe
        feb_2020 = pd.to_datetime('2020-02-01')
        if feb_2020 in prediction_df['periodo'].values:
            feb_2020_value = prediction_df[prediction_df['periodo'] == feb_2020]['tn'].values[0]
            plt.text(feb_2020, feb_2020_value, f'{feb_2020_value:.2f}', color='black', fontsize=10)
        
        plt.xlabel('Periodo')
        plt.ylabel('Toneladas (tn)')
        plt.title(f'XGBoost - Predicción T+2 - Product ID: {product_id}')
        plt.legend()
        
        # Agregar ejes y cuadrícula
        plt.grid(True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(1, -0.1, f'Timestamp: {timestamp}', ha='right', va='center', transform=plt.gca().transAxes, fontsize=8)
        plt.text(0, -0.1, f'Experimento: {EXPERIMENTO}', ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
        
        # Guardar el gráfico
        plt.savefig(os.path.join(output_dir, f'product_{product_id}.png'))
        plt.close()