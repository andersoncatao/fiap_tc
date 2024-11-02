import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from pandas.plotting import scatter_matrix

# Carregar o dataset
dataset = pd.read_csv('insurance.csv')

# Explorando o DataFrame
print(dataset.head())
print("Shape:", dataset.shape)
print(dataset.info())
print("Regiões:", set(dataset["região"]))
print("Contagem por região:", dataset["região"].value_counts())
print(dataset.describe())

# Pré-processamento: converter variáveis categóricas
dataset["gênero"] = dataset["gênero"].map({"masculino": 0, "feminino": 1})
dataset["fumante"] = dataset["fumante"].map({"não": 0, "sim": 1})
dataset = pd.get_dummies(dataset, columns=["região"], drop_first=False)

# Gerar histogramas das colunas numéricas antes da normalização
dataset.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histograma das Colunas Numéricas (Antes da Normalização)")
plt.show()

# Normalizar as variáveis 'idade', 'imc' e 'encargos' usando StandardScaler
scaler = StandardScaler()
dataset[["idade", "imc", "encargos"]] = scaler.fit_transform(dataset[["idade", "imc", "encargos"]])

# Analisar a proporção dos dados na categoria 'fumante'
print("Proporção de fumantes e não fumantes no dataset:")
print(dataset["fumante"].value_counts(normalize=True))

# Gerar a matriz de correlação
correlation_matrix = dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

# Exibir a correlação baseada na coluna 'encargos'
print("Correlação com encargos:")
print(correlation_matrix["encargos"].sort_values(ascending=False))

# Criar uma matriz de dispersão para as variáveis mais correlacionadas
attributes = ["encargos", "fumante", "idade", "imc", "filhos"]
scatter_matrix(dataset[attributes], figsize=(15, 10), alpha=0.8)
plt.suptitle("Matriz de Dispersão das Variáveis Mais Correlacionadas com Encargos")
plt.show()

# Aplicar a amostragem estratificada baseada na coluna 'fumante'
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["fumante"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]

# Dividir as variáveis independentes e dependentes
X_train = strat_train_set.drop("encargos", axis=1)
y_train = strat_train_set["encargos"]
X_test = strat_test_set.drop("encargos", axis=1)
y_test = strat_test_set["encargos"]

# Encontrar o melhor max_depth usando GridSearchCV
param_grid = {'max_depth': range(1, 21)}
tree_reg = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_max_depth = grid_search.best_params_['max_depth']
print("Melhor max_depth:", best_max_depth)

# Treinar o modelo DecisionTreeRegressor com o melhor max_depth
best_tree_reg = DecisionTreeRegressor(max_depth=best_max_depth, random_state=42)
best_tree_reg.fit(X_train, y_train)
y_pred_tree_best = best_tree_reg.predict(X_test)

# Função para calcular o intervalo de confiança
def calcular_intervalo_de_confianca(y_true, y_pred, confidence=0.95):
    squared_errors = (y_pred - y_true) ** 2
    mean_squared_error_value = np.mean(squared_errors)
    mse_std_error = np.std(squared_errors) / np.sqrt(len(y_true))
    intervalo = stats.t.interval(confidence, len(y_true) - 1, loc=mean_squared_error_value, scale=mse_std_error)
    return mean_squared_error_value, intervalo

# 1. Regressão Linear com Validação Cruzada e Intervalo de Confiança
lin_reg = LinearRegression()
scores_lin = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
rmse_scores_lin = np.sqrt(-scores_lin)
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin, intervalo_lin = calcular_intervalo_de_confianca(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print("\nRegressão Linear:")
print("RMSE (Validação Cruzada):", rmse_scores_lin)
print("RMSE Médio:", rmse_scores_lin.mean(), "Desvio Padrão:", rmse_scores_lin.std())
print("MSE no Conjunto de Teste:", mse_lin)
print("R² no Conjunto de Teste:", r2_lin)
print("Intervalo de Confiança (95%):", intervalo_lin)

# 2. Decision Tree Regressor com Validação Cruzada e Intervalo de Confiança
scores_tree = cross_val_score(best_tree_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
rmse_scores_tree = np.sqrt(-scores_tree)
mse_tree_best, intervalo_tree_best = calcular_intervalo_de_confianca(y_test, y_pred_tree_best)
r2_tree_best = r2_score(y_test, y_pred_tree_best)
print("\nDecision Tree Regressor:")
print("RMSE (Validação Cruzada):", rmse_scores_tree)
print("RMSE Médio:", rmse_scores_tree.mean(), "Desvio Padrão:", rmse_scores_tree.std())
print("MSE no Conjunto de Teste:", mse_tree_best)
print("R² no Conjunto de Teste:", r2_tree_best)
print("Intervalo de Confiança (95%):", intervalo_tree_best)

# 3. Random Forest Regressor com Validação Cruzada e Intervalo de Confiança
rf_reg = RandomForestRegressor(random_state=42)
scores_rf = cross_val_score(rf_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
rmse_scores_rf = np.sqrt(-scores_rf)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mse_rf, intervalo_rf = calcular_intervalo_de_confianca(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Regressor:")
print("RMSE (Validação Cruzada):", rmse_scores_rf)
print("RMSE Médio:", rmse_scores_rf.mean(), "Desvio Padrão:", rmse_scores_rf.std())
print("MSE no Conjunto de Teste:", mse_rf)
print("R² no Conjunto de Teste:", r2_rf)
print("Intervalo de Confiança (95%):", intervalo_rf)

# 4. Gradient Boosting Regressor com Validação Cruzada e Intervalo de Confiança
gb_reg = GradientBoostingRegressor(random_state=42)
scores_gb = cross_val_score(gb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
rmse_scores_gb = np.sqrt(-scores_gb)
gb_reg.fit(X_train, y_train)
y_pred_gb = gb_reg.predict(X_test)
mse_gb, intervalo_gb = calcular_intervalo_de_confianca(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print("\nGradient Boosting Regressor:")
print("RMSE (Validação Cruzada):", rmse_scores_gb)
print("RMSE Médio:", rmse_scores_gb.mean(), "Desvio Padrão:", rmse_scores_gb.std())
print("MSE no Conjunto de Teste:", mse_gb)
print("R² no Conjunto de Teste:", r2_gb)
print("Intervalo de Confiança (95%):", intervalo_gb)

# Base dos graficos
model_names = ["Regressão Linear", "Decision Tree", "Random Forest", "Gradient Boosting"]
predictions = [y_pred_lin, y_pred_tree_best, y_pred_rf, y_pred_gb]

# Gerar graficos comparando previsoes vs. valores reais para cada modelo com uma linha vermelha
plt.figure(figsize=(16, 10))
for i, (name, y_pred) in enumerate(zip(model_names, predictions)):
    plt.subplot(2, 2, i + 1)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='black', label="Previsões")  # Scatter plot das previsões
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label="Linha Ideal")  # Linha ideal
    plt.xlabel("Valores Reais")
    plt.ylabel("Previsões")
    plt.title(f"{name}: Previsões vs. Valores Reais")
    plt.legend()  # Adiciona a legenda
    plt.grid(visible=True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Gerar graficos comparando valores reais (azul) e valores previstos (verde) para cada modelo
plt.figure(figsize=(16, 10))
for i, (name, y_pred) in enumerate(zip(model_names, predictions)):
    plt.subplot(2, 2, i + 1)
    plt.plot(range(len(y_test)), y_test, 'o', color='blue', alpha=0.6, label="Valores Reais")  # Bolinhas azuis para valores reais
    plt.plot(range(len(y_test)), y_pred, 'o', color='green', alpha=0.6, label="Valores Previstos")  # Bolinhas verdes para valores previstos
    plt.xlabel("Índice")
    plt.ylabel("Encargos (Normalizados)")
    plt.title(f"{name}: Valores Reais vs. Previstos")
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


