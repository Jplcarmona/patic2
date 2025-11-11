import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit, learning_curve

from sklearn.pipeline import Pipeline

from sklearn.metrics import ConfusionMatrixDisplay,PrecisionRecallDisplay,classification_report

from ft_engineering import crear_pipelines

class HeuristicModel(BaseEstimator, ClassifierMixin):
    """
    Modelo heurístico compatible con scikit-learn.
    Usa reglas simples basadas en variables financieras y demográficas.
    """

    def __init__(self, rdi_threshold=0.4, puntaje_min = 400, max_creditos=6, max_consultas = 10):
        self.rdi_threshold = rdi_threshold
        self.max_creditos = max_creditos
        self.puntaje_min = puntaje_min
        self.max_consultas = max_consultas

    def fit(self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            # --- Reglas heurísticas ---
            if row.get("relacion_deuda_ingreso", np.inf) > self.rdi_threshold:
                predictions.append(0)  # Riesgo alto
            elif row.get("puntaje_datacredito", 0) < self.puntaje_min:
                predictions.append(0)  # Puntaje bajo = riesgo
            elif row.get("cant_creditosvigentes", 0) > self.max_creditos:
                predictions.append(0)  # Demasiados créditos activos
            elif (row.get("huella_consulta", 0) > self.max_consultas):
                predictions.append(0)  # Demasiadas consultas de credito
            else:
                predictions.append(1)  # Caso favorable, paga a tiempo
        return np.array(predictions)

def evaluar_modelo ():
    
    # 1. Cargar datos procesados
    preprocessor, X_train, X_test, y_train, y_test, X_train_processed, X_test_processed = crear_pipelines()
        
    # 2. Definir modelo
    model = HeuristicModel()
    
    # 3. Configuracion de metricas
    scoring_metrics = ["accuracy", "f1", "precision", "recall"]
    kfold = KFold(n_splits=10)
    
    model_pipe = Pipeline(steps=[("model", model)])
    
    cv_results = {}
    train_results = {}
    
    # 4. Validacion cruzada
    for metric in scoring_metrics:
        cv_results[metric] = cross_val_score( model_pipe, X_train, y_train, cv=kfold, scoring=metric)
        
        model_pipe.fit(X_train, y_train)
        train_results[metric] = model_pipe.score(X_train, y_train)
        
    # Convertir resultados a DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    
    # 5. Imprimir resultados
    for metric_name in scoring_metrics:
        mean_score = cv_results_df[metric_name].mean()
        std_score = cv_results_df[metric_name].std()
        train_score = train_results[metric_name]
        print(f"{metric_name} - CV mean: {mean_score:.2f}, CV std: {std_score:.2f}")
        print(f"{metric_name} - Train score: {train_score:.2f}")
        
    # 6. Boxlplot de resultados
    ax = cv_results_df.boxplot()
    ax.set_title("Cross Validation Boxplot")
    ax.set_ylabel("Score")
    plt.show()
    
    # 7. Reporte de clasificacion
    y_pred = model_pipe.predict(X_test)
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    
    # 8. Consistencia del modelo
    common_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=10, test_size=0.2, random_state=123),
        "n_jobs": -1,
        "return_times": True,
    }
    scoring_metric = "recall"
    
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        model_pipe, **common_params, scoring=scoring_metric
    )
    
    train_mean, train_std = np.mean(train_scores, axis = 1), np.std(train_scores, axis = 1)
    test_mean, test_std = np.mean(test_scores, axis = 1), np.std(test_scores, axis = 1)
    
    fit_times_mean, fit_times_std = np.mean(fit_times, axis = 1), np.std(fit_times, axis = 1)
    score_times_mean, score_times_std = np.mean(score_times, axis = 1), np.std(score_times, axis = 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3, color="orange")
    ax.set_title(f"Learning Curve for {model.__class__.__name__}")
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring_metric)
    ax.legend(loc="best")
    plt.show()
    
    print("Training Sizes:", train_sizes)
    print("Training Scores Mean:", train_mean)
    print("Training Scores Std:", train_std)
    print("Test Scores Mean:", test_mean)
    print("Test Scores Std:", test_std)
    
    # 9. Escalabilidad
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex= True)
    
    # Fit time
    ax[0].plot(train_sizes, fit_times_mean, "o-")
    ax[0].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.3)
    ax[0].set_ylabel("Fit time (s)")
    ax[0].set_title(f"Scalability of the {model.__class__.__name__} classifier")
    
    # Score time
    ax[1].plot(train_sizes, score_times_mean, "o-")
    ax[1].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha=0.3)
    ax[1].set_ylabel("Score time (s)")
    ax[1].set_xlabel("Number of training samples")
    
    plt.show()
    
    # Imprimir valores
    print("Fit Times Mean:", fit_times_mean)
    print("Fit Times Std:", fit_times_std)
    print("Score Times Mean:", score_times_mean)
    print("Score Times Std:", score_times_std)
    
if __name__ == "__main__":

    print(" Iniciando prueba unitaria del HeuristicModel...\n")

    try:
        evaluar_modelo()
        print("\n Prueba unitaria completada: HeuristicModel evaluado correctamente.")
    except Exception as e:
        print("\n Error durante la prueba unitaria del HeuristicModel:")
        print(str(e))
        raise
