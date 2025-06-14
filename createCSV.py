import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="datamart_mermas",
    user="postgres",
    password="Inacap2025"
)

tablas = ['mermas', 'tienda', 'tiempo', 'producto', 'motivo_merma']

for tabla in tablas:
    try:
        print(f"Exportando tabla: {tabla}")
        df = pd.read_sql_query(f"SELECT * FROM {tabla}", conn)
        df.to_csv(f"{tabla}.csv", index=False)
        print(f"Exportado como {tabla}.csv")
    except Exception as e:
        print(f"Error al exportar {tabla}: {e}")

conn.close()
print("Todos los CSV fueron generados correctamente.")