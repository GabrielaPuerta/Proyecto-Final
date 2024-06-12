from flask import Flask, render_template, request
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import io
import base64
from sklearn.cluster import KMeans

app = Flask(__name__)

scaler_and_kmeans = joblib.load('ModeloEntrenadoPinguinsUltimo.pkl')
scaler = scaler_and_kmeans[0]
kmeans_loaded = scaler_and_kmeans[1]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file)

    df = df.dropna()
    df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1})

    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[features]

    df['Cluster'] = kmeans_loaded.predict(X)

    print(kmeans_loaded.n_clusters) 

    silhouette_coefficient = silhouette_score(X, df['Cluster'])
    
    ch_score = calinski_harabasz_score(X, kmeans_loaded.labels_)


    # Graficos
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='Cluster', palette='viridis')
    plt.title('Clusters de Pingüinos basados en Longitud y Profundidad del Pico')
    plt.xlabel('Longitud del Pico (mm)')
    plt.ylabel('Profundidad del Pico (mm)')

    # Guarda la imagen en un objeto BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Se convierte la imagen a una cadena codificada en base64
    imagen_codificada = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    confusion_matrix = pd.crosstab(df['Cluster'], df['sex'], rownames=['Cluster'], colnames=['Sex'])

    confusion_matrix_html = confusion_matrix.to_html()

    # Calcular la cantidad de pingüinos machos y hembras en cada cluster
    gender_counts = df.groupby(['Cluster', 'sex']).size().reset_index(name='count')
    gender_counts['sex'] = gender_counts['sex'].replace({0: 'Macho', 1: 'Hembra'})

    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=gender_counts, x='Cluster', y='count', hue='sex', palette={'Macho': 'blue', 'Hembra': 'red'})
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.xlabel('Cluster')
    plt.ylabel('Cantidad')
    plt.title('Cantidad de Pinguinos Hembras y Machos en cada Cluster')

    # Se guarda la imagen del gráfico de barras
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grafico_barras_codificado = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    # Gráficos de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bill_length_mm', y='body_mass_g', hue='sex', palette='viridis')
    plt.title('Longitud del pico vs. Masa corporal')
    plt.xlabel('Longitud del Pico (mm)')
    plt.ylabel('Masa Corporal (g)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grafico_bill_mass_codificado = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()


    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='sex', palette='viridis')
    plt.title('Longitud del pico vs. Profundidad del pico')
    plt.xlabel('Longitud del Pico (mm)')
    plt.ylabel('Profundidad del Pico (mm)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grafico_bill_depth_codificado = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    # Método del codo
    scc = []
    X_scaled = scaler.transform(X)
    for i in range(1, 11):
        modelo = KMeans(n_clusters=i, random_state=0, n_init="auto")
        modelo.fit(X_scaled)
        scc.append(modelo.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), scc, marker="o")
    plt.title('Elbow Method')
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grafico_elbow_codificado = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

        # Box plot de masa corporal por cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='body_mass_g', data=df)
    plt.title('Boxplot de masa corporal por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Masa Corporal (g)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grafico_masa_codificado = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()



    return render_template('result.html',grafico_masa_codificado = grafico_masa_codificado,ch_score= ch_score, silhouette_coefficient=silhouette_coefficient, confusion_matrix_html=confusion_matrix_html,  imagen_codificada=imagen_codificada, grafico_barras_codificado=grafico_barras_codificado, grafico_bill_mass_codificado=grafico_bill_mass_codificado, grafico_bill_depth_codificado=grafico_bill_depth_codificado, grafico_elbow_codificado=grafico_elbow_codificado)

if __name__ == '__main__':
    app.run(debug=True)