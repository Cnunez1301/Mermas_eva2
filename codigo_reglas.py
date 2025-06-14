import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df_mermas = pd.read_csv(r"C:\Users\kanom\OneDrive\Desktop\Universidad\Programacion\mermas.csv")

df_mermas.columns = df_mermas.columns.str.strip().str.lower()

df_reglas = df_mermas[['nombre_producto', 'categoria', 'tienda', 'motivo', 'subtipo']]

transacciones = df_reglas.astype(str).values.tolist()

te = TransactionEncoder()
transacciones_codificadas = te.fit(transacciones).transform(transacciones)
df_binario = pd.DataFrame(transacciones_codificadas, columns=te.columns_)

frequent_itemsets = apriori(df_binario, min_support=0.05, use_colnames=True)

reglas = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

reglas = reglas.sort_values(by='lift', ascending=False)

print("\nReglas de Asociación Detectadas (Top 10):\n")
print(reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

reglas.to_csv("reglas_asociacion_mermas.csv", index=False)
print("\nReglas exportadas a 'reglas_asociacion_mermas.csv'")