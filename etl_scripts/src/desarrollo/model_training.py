import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from ft_engineering import crear_pipelines


def summarize_classification(y_test, y_pred):
    
    acc = accuracy_score(y_test, y_pred, normalize=True)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    
    return {"Accuracy": acc,"Precision": prec, "Recall": recall,"F1": f1,"ROC_AUC": roc,} 
    
def build_model(classifier, X_train, X_test, y_train, y_test) -> dict:
     # Train the classifier pipeline
    model = classifier.fit(X_train, y_train)

    # Predict the test data
    y_pred = model.predict(X_test)

    # Predict the train data
    y_pred_train = model.predict(X_train)

    # Calculate the performance metrics
    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    return {"train": train_summary, "test": test_summary, "model": model}

def evaluar_consistencia(model, X, y, scoring="f1"):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    return scores.mean(), scores.std()

def evaluar_escalabilidad(model, X, y, scoring="f1"):
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=10, test_size=0.2, random_state=123),
        "n_jobs": -1,
        "return_times": True,
    }
    return learning_curve(model, **common_params, scoring=scoring)

def training():
    # 1. Cargar datos procesados del pipeline
    preprocessor, X_train, X_test, y_train, y_test, X_train_processed, X_test_processed = crear_pipelines()
    
    # 2. Modelos
    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear"),
        "Linear SVC": LinearSVC(C=1.0, max_iter= 1000, tol=1e-3, dual=False),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)
    }
    
    results = {}
    
    # 3. Entrenar y evaluar cada modelo
    for model_name, clf in models.items():
        print(f"\n Evaluando modelo: {model_name}")
        summary = build_model(clf, X_train_processed, X_test_processed, y_train, y_test)
        results[model_name] = summary
        
        # Consistencia
        mean_cv, std_cv = evaluar_consistencia(clf, X_train_processed, y_train)
        print(f" CV mean: {mean_cv:.3f}, CV Std: {std_cv:.3f}")
        
        # Escalabilidad
        train_sizes, train_scores, test_scores, fit_times, score_times = evaluar_escalabilidad(clf, X_train_processed, y_train)
        
        # Curva de aprendizaje
        train_mean, train_std = train_scores.mean(axis= 1), train_scores.std(axis=1)
        test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_mean, "o-", label="Train Score")
        plt.plot(train_sizes, test_mean, "o-", label="CV Score", color="orange")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3, color="orange")
        plt.title(f"Learning Curve: {model_name}")
        plt.xlabel("Training Samples")
        plt.ylabel("Score")
        plt.legend()
        plt.show()
        
    # 4. Tabla Comparativa
    summary_data = []
    for model_name, summary in results.items():
        test_metrics = summary["test"]
        summary_data.append([model_name] + list(test_metrics.values())) 
                 
    df_summary = pd.DataFrame(summary_data, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
    print("\n Tabla comparativa de modelo:")
    print(df_summary)
    
    # 5. Seleccion de mejor modelo 
    best_model_row = df_summary.sort_values(by="F1", ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    best_model = results[best_model_name]["model"]
    print(f"\n El mejor modelo es {best_model_name} con F1 = {best_model_row['F1']:.3f}")
    
    # 6. Matriz de confusion
    y_pred = best_model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión - {best_model_name}")
    plt.show()
    
    return best_model, best_model_name

if __name__ == "__main__":
    print(" Iniciando prueba unitaria de model_training...")

    try:
        best_model, best_model_name = training()

        # Validar tipo de salida: el modelo debe ser un estimador de sklearn
        from sklearn.base import BaseEstimator
        assert isinstance(best_model, BaseEstimator), "  Error: training() no devolvió un modelo válido de scikit-learn."

        # Validar que tiene el método predict
        assert hasattr(best_model, "predict"), " Error: el modelo no tiene método predict()."

        # Validar que ya está entrenado (coef_, feature_importances_ o classes_)
        trained_ok = any(
            hasattr(best_model, attr) 
            for attr in ["coef_", "feature_importances_", "classes_"]
        )
        assert trained_ok, " Error: el modelo parece no estar entrenado correctamente."

        # Validar que el nombre del modelo sea un string
        assert isinstance(best_model_name, str), " Error: no se devolvió el nombre del modelo."

        print(f" Prueba unitaria completada: {best_model_name} entrenado correctamente y listo para deploy.")

    except Exception as e:
        print(f" Prueba unitaria fallida: {e}")