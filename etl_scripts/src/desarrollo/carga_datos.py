from google.cloud import bigquery
 
# Set your GCP project ID
project_id = "proyecto-ciencia-produccion"
 
# Create client
client = bigquery.Client(project=project_id)
 
# SQL Query
query = "SELECT * FROM `proyecto-ciencia-produccion.proyecto_final_CDP.restaurantes`"
df = client.query(query).to_dataframe()
 
print(df.head())