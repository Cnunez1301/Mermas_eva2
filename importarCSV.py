import pandas as pd
import psycopg2

# Cargar el Excel
df = pd.read_excel(
    r"C:\Users\kanom\OneDrive\Desktop\Universidad\Inteligencia de Negocio\mermas_actividad_unidad_2.xlsx",
    sheet_name="Hoja1"  # Ajustar si el nombre cambia
)

# Normalizar y preparar datos
df = df.rename(columns={
    'descripcion': 'nombre_producto',
    'merma_unidad_p': 'unidades_merma',
    'merma_monto_p': 'costo_unitario',
    'ubicación_motivo': 'area_responsable'
})

df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
df = df.dropna(subset=['fecha'])

# Calcular campos
df['semana'] = df['fecha'].dt.isocalendar().week
df['motivo'] = df['motivo'].str.strip().str.lower()

# Subtipo (según reglas anteriores)
df['subtipo'] = df['motivo'].apply(
    lambda m: 'Caducidad' if m == 'vencimiento' else
              'Manipulacion' if m in ['interno', 'clientes', 'cliente'] else
              'Mal Estado' if m == 'proveedor' else 'Otro'
)

# Restaurar capitalización
df['motivo'] = df['motivo'].str.capitalize()
df['subtipo'] = df['subtipo'].str.capitalize()

# Calcular valor total de merma
df['valor_total_merma'] = df['unidades_merma'] * df['costo_unitario']

# Seleccionar columnas finales
df_mermas = df[['nombre_producto', 'fecha', 'semana', 'categoria', 'tienda', 'comuna',
                'region', 'motivo', 'subtipo', 'area_responsable', 'unidades_merma',
                'costo_unitario', 'valor_total_merma']].dropna()

# Conexión a PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="datamart_mermas",
    user="postgres",
    password="Inacap2025"
)
cur = conn.cursor()

# Insertar en la tabla mermas
for _, row in df_mermas.iterrows():
    cur.execute("""
        INSERT INTO mermas (
            nombre_producto, fecha, semana, categoria, tienda, comuna, region,
            motivo, subtipo, area_responsable, unidades_merma, costo_unitario, valor_total_merma
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        row['nombre_producto'],
        row['fecha'],
        int(row['semana']),
        row['categoria'],
        row['tienda'],
        row['comuna'],
        row['region'],
        row['motivo'],
        row['subtipo'],
        row['area_responsable'],
        row['unidades_merma'],
        row['costo_unitario'],
        row['valor_total_merma']
    ))

# Finalizar
conn.commit()
cur.close()
conn.close()

print("✅ Datos insertados correctamente en la tabla 'mermas'")