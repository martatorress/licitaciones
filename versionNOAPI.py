import numpy as np
import fitz
import os
import tiktoken
import json
from typing import List, Dict
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import tempfile
import shutil
from openai import OpenAI

client = OpenAI(
  api_key=""
)


# === Función para contar tokens ===
def contar_tokens(texto: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(texto))

# === Dividir texto en bloques de máximo 4000 tokens ===
def dividir_texto_en_chunks(texto: str, max_tokens: int = 4000) -> List[str]:
    chunks = []
    palabras = texto.split()
    chunk_actual = []
    tokens_actuales = 0

    for palabra in palabras:
        tokens_palabra = contar_tokens(palabra + " ")
        if tokens_actuales + tokens_palabra > max_tokens:
            chunks.append(" ".join(chunk_actual))
            chunk_actual = [palabra]
            tokens_actuales = tokens_palabra
        else:
            chunk_actual.append(palabra)
            tokens_actuales += tokens_palabra

    if chunk_actual:
        chunks.append(" ".join(chunk_actual))

    return chunks

# === Extraer texto del PDF ===
def pdf_to_text(ruta_pdf: str) -> str:
    texto = ""
    pdf = fitz.open(ruta_pdf)
    for pagina in pdf:
        texto += pagina.get_text("text") + "\n"
    pdf.close()
    return texto.strip()


def key_points_json(texto: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Usa un modelo potente si es posible
            messages=[{
                "role": "system",
                "content": """
Eres un asistente experto en contratación pública. Los documentos que te mando puede ser un pliego de una licitación española u ofertas al pliego y extrae un resumen objetivo en formato JSON con la siguiente información estructurada. Tómate tu tiempo para encontrar bien los requisitos. Por ejemplo: requisito en el pliego: requiere título habilitante; json de la oferta: tiene título habilitante.

✅ Criterios principales:
- Claridad y precisión en el objeto del contrato.
- Identificación del órgano contratante.
- Presupuesto base y valor estimado si aparecen.
- Fechas clave (presentación, apertura, ejecución).
- Criterios de adjudicación (con peso si lo hay).
- Requisitos relevantes o condiciones técnicas.
- Información sobre plazos y costes si figuran.
- Códigos CPV (si están disponibles, como lista de strings).
- Cualquier otro campo útil detectado.

Importante:
- No uses "desconocido", si no está presente, simplemente omítelo.
- No inventes información ni generalices.
- No des importancia excesiva a la "actividad principal" si no se menciona explícitamente.

Ejemplo de estructura esperada:

{
  "tipo": "licitacion",
  "objeto": "descripción clara del contrato",
  "organo_contratante": "...",
  "presupuesto_base": "...",
  "valor_estimado": "...",
  "procedimiento": "...",
  "lugar_ejecucion": "...",
  "actividad_principal": "..."
  "codigo_cpv": ["cpv1", "cpv2"],
  "criterios_adjudicacion": [
    {"criterio": "...", "ponderacion": ..., "evaluacion": "..."}
  ],
  "fecha_limite_presentacion": "...",
  "fecha_apertura": "...",
  "requisitos": ["...", "..."],
  "plazos": ["...", "..."],
  "costes": ["...", "..."]
}
"""
            }, {
                "role": "user",
                "content": texto[:4000]
            }]
        )
        print("✅ Entrando en el bloque de análisis de respuesta...")
        respuesta_texto = response.choices[0].message.content.strip()
        print("🔵 Texto recibido:\n", respuesta_texto) 
        try:
            result = json.loads(respuesta_texto)
            print("🟢 JSON extraído:", json.dumps(result, indent=2, ensure_ascii=False))
            return result
        except:
            inicio = respuesta_texto.find('{')
            fin = respuesta_texto.rfind('}') + 1
            if inicio >= 0 and fin > 0:
                json_texto = respuesta_texto[inicio:fin]
                print(json_texto)
                return json.loads(json_texto)
            raise
    except Exception as e:
        print(f"Error en extracción: {str(e)}")
        return {}


# === Procesar todo el documento en chunks y fusionar resultados ===
def chunks_key_points(texto: str) -> dict:
    chunks = dividir_texto_en_chunks(texto, max_tokens=4000)

    resultado_final = {
        "tipo": "licitacion",
        "objeto": None,
        "organo_contratante": None,
        "presupuesto_base": None,
        "valor_estimado": None,
        "procedimiento": None,
        "lugar_ejecucion": None,
        "codigo_cpv": [],
        "criterios_adjudicacion": [],
        "fecha_limite_presentacion": None,
        "fecha_apertura": None,
        "requisitos": [],
        "plazos": [],
        "costes": []
    }

    for i, chunk in enumerate(chunks):
        print(f"Procesando chunk {i+1}/{len(chunks)}...")
        resultado_parcial = key_points_json(chunk)

        for campo in resultado_final:
            if isinstance(resultado_final[campo], list):
                nuevos = resultado_parcial.get(campo, [])
                for item in nuevos:
                    if item not in resultado_final[campo]:
                        resultado_final[campo].append(item)
            elif resultado_final[campo] is None and resultado_parcial.get(campo):
                resultado_final[campo] = resultado_parcial.get(campo)

    return resultado_final


# === Embedding promedio (si lo usas más adelante) ===
def average_embedding(chunks: List[str]) -> np.ndarray:
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-3-large"
            )
            embeddings.append(np.array(response.data[0].embedding))
        except Exception as e:
            print(f"Error en embedding: {str(e)}")
            continue

    if not embeddings:
        raise ValueError("No se pudo obtener embedding válido")

    return np.mean(embeddings, axis=0)

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
        Calcula una puntuación escalada (0-100) basada en la similitud coseno entre dos vectores.

        - < 0.3   → 0 puntos (sin relación)
        - 0.3–0.6 → transición lineal de 15 a 60 puntos
        - > 0.6   → transición lineal de 60 a 100 puntos
    """
    similitud_base = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    threshold_no_relacionado = 0.3
    threshold_relacionado = 0.6

    if similitud_base < threshold_no_relacionado:
        return 0
    elif similitud_base < threshold_relacionado:
        return ((similitud_base - threshold_no_relacionado) / 
                     (threshold_relacionado - threshold_no_relacionado)) * 45
    else:
        factor = (similitud_base - threshold_relacionado) / (1 - threshold_relacionado)
        return 60 + (factor * 40)
    

    
def content_similarity(key_points1: dict, key_points2: dict) -> float:
    """
    Evalúa la similitud semántica entre una licitación y una oferta.
    Devuelve una puntuación entre 0 y 100 basada en múltiples criterios.
    """
    try:
        activity_1 = key_points1.get('actividad_principal', '').lower()
        activity_2 = key_points2.get('actividad_principal', '').lower()

        if activity_1 and activity_2 and activity_1 != activity_2:
            print(f"⚠️ Actividades principales diferentes: {activity_1} vs {activity_2}. Se penalizará, pero no se descarta.")

        prompt_sistema = """
        Eres un experto en contratación pública. Tu tarea es evaluar objetivamente la similitud entre una licitación y una oferta presentada, basándote en la alineación documental y técnica entre ambos textos. Tómate tu tiempo para leer y comparar.

🎯 TU OBJETIVO:
Calcular un único valor numérico (de 0 a 100) que represente cuán bien se ajusta la oferta a los requisitos, objetivos y condiciones expresadas en la licitación. Interpreta correctamente redacciones equivalentes: por ejemplo, si un texto dice "dos platos más postre y bebida" y el pliego pide “menú completo”, se considera que cumple.
Ten en cuenta los siguientes criterios de evaluación y aplica el peso indicado a cada uno de ellos:

CRITERIOS DE EVALUACIÓN:
1. Coincidencia de sector y objeto del contrato (20%)
   - ¿La oferta pertenece al mismo sector económico que el pliego?
   - ¿El objeto de la oferta se ajusta al objeto principal del contrato?
   - ¿Coincide algún código CPV si está presente?

2. Cumplimiento de requisitos (20%)
   - ¿La oferta responde adecuadamente a los requisitos exigidos en el pliego?
   - ¿Incluye especificaciones, capacidades o menciona documentación técnica relevante, títulos habilitantes o certificados?

3. Cumplimiento de condiciones administrativas (10%)
   - ¿Se presentan los documentos administrativos requeridos?
   - ¿Se mencionan procedimientos, certificaciones u obligaciones legales solicitadas?

4. Alineación de plazos (15%)
   - ¿La oferta contempla los plazos de entrega, ejecución u otros tiempos clave definidos en el pliego?

5. Correspondencia económica (25%)
   - ¿La oferta incluye detalles de costes o presupuesto alineados con los importes del pliego?
   - ¿Menciona el presupuesto base o el valor estimado?

6. Inclusión de mejoras u optimizaciones voluntarias (10%)
   - ¿Se proponen elementos adicionales, mejoras, innovaciones o compromisos no exigidos pero beneficiosos?

ESCALA:
- 0–20: Muy baja relación
- 21–50: Relación parcial
- 51–80: Buena relación
- 81–100: Excelente relación

🚨 RETORNA ÚNICAMENTE UN NÚMERO ENTRE 0 Y 100 (sin explicaciones, sin símbolos ni texto adicional).
"""

        user_content = f"Licitación:\n{json.dumps(key_points1, ensure_ascii=False)}\n\nOferta:\n{json.dumps(key_points2, ensure_ascii=False)}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": user_content}
            ]
        )

        result_text = response.choices[0].message.content.strip()
        result_clean = ''.join(filter(lambda c: c.isdigit() or c == '.', result_text))

        return float(result_clean)

    except Exception as e:
        print(f"❌ Error al calcular similitud semántica: {str(e)}")
        return 0.0

def listar_ofertas(ofertas):
    if not ofertas:
        return ""
    return "\n".join([f"Oferta_{i+1}: {oferta.name}" for i, oferta in enumerate(ofertas)])


def pros_and_cons(key_points_licitacion: dict, key_points_oferta: dict) -> tuple:
    ventajas = []
    desventajas = []

    # 1. Coincidencia de actividad o sector
    pliego_activity = key_points_licitacion.get('actividad_principal', '').lower()
    bidder_activity = key_points_oferta.get('actividad_principal', '').lower()

    pliego_cpv = key_points_licitacion.get('codigo_cpv', '')
    bidder_cpv = key_points_oferta.get('codigo_cpv', '')

    if pliego_activity and bidder_activity or pliego_cpv and bidder_cpv:
        if pliego_activity == bidder_activity or pliego_cpv == bidder_cpv:
            ventajas.append(f"✓ Actividad alineada: {pliego_activity}")
        else:
            desventajas.append(f"⚠️ Actividades distintas: {pliego_activity} vs. {bidder_activity}")
    else:
        desventajas.append("⚠️ No se puede verificar la actividad principal")

    # 2. Requisitos
    for req in key_points_licitacion.get('requisitos', []):
        if any(req.lower() in r.lower() for r in key_points_oferta.get('requisitos', [])):
            ventajas.append(f"✓ Cumple requisito: {req}")
        else:
            desventajas.append(f"✗ No cumple requisito: {req}")

    # 3. Plazos
    plazos_oferta = key_points_oferta.get('plazos', [])
    if plazos_oferta:
        ventajas.append("✓ Incluye información sobre plazos")
        for p in plazos_oferta[:3]:
            ventajas.append(f"  → {p}")
    else:
        desventajas.append("✗ No se especifican plazos")

    # 4. Costes
    costes_oferta = key_points_oferta.get('costes', [])
    if costes_oferta:
        ventajas.append("✓ Incluye desglose de costes")
        for c in costes_oferta[:3]:
            ventajas.append(f"  → {c}")
    else:
        desventajas.append("✗ No se incluyen detalles de costes")

    # 5. Evaluación frente a criterios de adjudicación
    criterios = key_points_licitacion.get('criterios_adjudicacion', [])
    if criterios:
        ventajas.append("✓ Contempla criterios de evaluación")
        for crit in criterios[:2]:
            ventajas.append(f"  → {crit.get('criterio', 'Criterio no especificado')}")

    # 6. Fechas clave
    fecha_presentacion = key_points_licitacion.get('fecha_limite_presentacion')
    fecha_apertura = key_points_licitacion.get('fecha_apertura')
    if fecha_presentacion:
        ventajas.append(f"✓ Fecha límite detectada: {fecha_presentacion}")
    else:
        desventajas.append("✗ No se detecta la fecha límite de presentación")
    if fecha_apertura:
        ventajas.append(f"✓ Fecha de apertura detectada: {fecha_apertura}")

    return ventajas, desventajas


    
def total_similarity(embedding1: np.ndarray, embedding2: np.ndarray, 
                           key_points1: dict, key_points2: dict) -> float:
    
    similitud_coseno = cosine_similarity(embedding1, embedding2)
    similitud_semantica = content_similarity(key_points1, key_points2)
    
    return 0.2 * similitud_coseno + 0.8 * similitud_semantica

def requires_chunking(texto: str, threshold_tokens: int = 3999) -> dict:
    total_tokens = contar_tokens(texto)
    
    if total_tokens > threshold_tokens:
        print(f"📄 Texto largo ({total_tokens} tokens). Se dividirá en chunks.")
        return chunks_key_points(texto)
    else:
        print(f"📄 Texto corto ({total_tokens} tokens). Se procesará directamente.")
        return key_points_json(texto)


def analysis_doc(ruta_archivo: str) -> tuple:
    texto = pdf_to_text(ruta_archivo)
    key_points = requires_chunking(texto)
    chunks = dividir_texto_en_chunks(texto)
    embedding = average_embedding(chunks)
    return embedding, key_points

def generate_radar_chart(key_points: dict) -> go.Figure:
    categorias = ['Requisitos', 'Plazos', 'Costes']
    valores = [
        len(key_points.get('requisitos', [])),
        len(key_points.get('plazos', [])),
        len(key_points.get('costes', []))
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=categorias,
        fill='toself',
        name=key_points.get('tema_principal', 'Documento'),
        line=dict(color='#2E86C1', width=2),
        fillcolor='rgba(33, 64, 95, 0.25)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, max(valores) + 1],
                gridcolor='#E8E8E8'
            ),
            angularaxis=dict(
                gridcolor='#E8E8E8'
            )
        ),
        showlegend=True,
        title=dict(
            text="Análisis de Componentes",
            font=dict(size=20)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def generate_comparison_chart(resultados: list) -> go.Figure:
    # Crear un DataFrame con los nombres reales de los archivos
    df = pd.DataFrame([
        {
            'Nombre del archivo': os.path.splitext(r.get('aplicacion', f"Oferta_{idx+1}.pdf"))[0],
            'Puntuación (%)': round(r['porcentaje'], 2)
        }
        for idx, r in enumerate(resultados)
    ])


    # Crear gráfico de barras con estilo elegante
    fig = px.bar(
        df,
        x='Nombre del archivo',
        y='Puntuación (%)',
        color='Puntuación (%)',
        text='Puntuación (%)',
        color_continuous_scale=['#A94442', '#800000', '#4B0000'],
        title='Comparación de las Propuestas Presentadas'
    )

    fig.update_traces(
        textposition='outside',
        marker_line_color='#333',
        marker_line_width=0.8
    )

    fig.update_layout(
        xaxis_tickangle=-30,
        yaxis=dict(range=[0, 105], title='Puntuación (%)', gridcolor='#EEE'),
        xaxis=dict(title='Documento', gridcolor='#EEE'),
        title=dict(
            font=dict(size=20, color='#333')
        ),
        paper_bgcolor='#fafafa',
        plot_bgcolor='#fafafa',
        font=dict(family='Segoe UI', size=14, color='#333')
    )

    return fig


def save_pdf(file_data, folder, filename):
    path = os.path.join(folder, filename)
    with open(path, 'wb') as f:
        f.write(file_data)
    return path


def analyze_and_compare_bid(bid_file, idx, temp_dir, pliego_embed, pliego_data):
    name = getattr(bid_file, "name", f"Oferta_{idx}.pdf")
    path = save_pdf(bid_file, temp_dir, name)

    bid_embed, bid_data = analysis_doc(path)
    score = total_similarity(pliego_embed, bid_embed, pliego_data, bid_data)
    pros, cons = pros_and_cons(pliego_data, bid_data)

    return {
        'aplicacion': name,
        'porcentaje': score,
        'ventajas': pros,
        'desventajas': cons,
        'key_points_oferta': bid_data
    }

def generate_html_pliego(pliego_data):
    criterios_html = ""
    if "criterios_adjudicacion" in pliego_data:
        criterios_html = "<ul>" + "".join(
            f"<li><strong>{c['criterio']}</strong> – {c.get('ponderacion', 'sin peso')} ({c.get('evaluacion', 'sin info')})</li>"
            for c in pliego_data["criterios_adjudicacion"]
        ) + "</ul>"

    requisitos_html = ""
    if "requisitos" in pliego_data:
        requisitos_html = "<ul>" + "".join(f"<li>{r}</li>" for r in pliego_data["requisitos"]) + "</ul>"

    plazos_html = ""
    if "plazos" in pliego_data:
        plazos_html = "<ul>" + "".join(f"<li>{p}</li>" for p in pliego_data["plazos"]) + "</ul>"

    costes_html = ""
    if "costes" in pliego_data:
        costes_html = "<ul>" + "".join(f"<li>{c}</li>" for c in pliego_data["costes"]) + "</ul>"

    codigos_cpv = ""
    if "codigo_cpv" in pliego_data:
        codigos_cpv = ", ".join(pliego_data["codigo_cpv"])

    return f"""
    <div style='background: #fff; padding: 30px; border-radius: 12px; max-width: 960px; margin: auto; border: 1px solid #ccc;'>
        <h2 style='color: #2c3e50;'>📋 Resumen del Pliego de la Licitación</h2>
        <p><strong>📌 Objeto:</strong> {pliego_data.get("objeto", "No disponible")}</p>
        <p><strong>🏢 Órgano contratante:</strong> {pliego_data.get("organo_contratante", "No disponible")}</p>
        <p><strong>💶 Presupuesto base:</strong> {pliego_data.get("presupuesto_base", "No disponible")}</p>
        <p><strong>📊 Valor estimado:</strong> {pliego_data.get("valor_estimado", "No disponible")}</p>
        <p><strong>⚙️ Procedimiento:</strong> {pliego_data.get("procedimiento", "No disponible")}</p>
        <p><strong>📍 Lugar de ejecución:</strong> {pliego_data.get("lugar_ejecucion", "No disponible")}</p>
        <p><strong>📂 Códigos CPV:</strong> {codigos_cpv}</p>
        <p><strong>📅 Fecha límite de presentación:</strong> {pliego_data.get("fecha_limite_presentacion", "No disponible")}</p>
        <p><strong>🗓️ Fecha de apertura:</strong> {pliego_data.get("fecha_apertura", "No disponible")}</p>

        <div style='margin-top: 20px;'>
            <h3 style='color: #34495e;'>📌 Requisitos</h3>
            {requisitos_html}
        </div>

        <div style='margin-top: 20px;'>
            <h3 style='color: #34495e;'>⏳ Plazos relevantes</h3>
            {plazos_html}
        </div>

        <div style='margin-top: 20px;'>
            <h3 style='color: #34495e;'>💰 Costes adicionales</h3>
            {costes_html}
        </div>
    </div>
    """


def generate_html_best_offer(offer):
    return f"""
    <div style='background: #fff; padding: 30px; border-radius: 12px; max-width: 960px; margin: auto; border: 1px solid #b22222;'>
        <div style='text-align: center; margin-bottom: 25px;'>
            <h2 style='color: #800000;'>📊 Oferta mejor valorada</h2>
            <p style='color: #555;'>Análisis objetivo basado en criterios técnicos</p>
        </div>
        <div style='background: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
            <h3 style='margin: 0; color: #800000;'>📄 {offer['aplicacion']}</h3>
            <p style='margin-top: 10px; font-weight: bold; color: #155724;'>Puntuación: {offer['porcentaje']:.2f}%</p>
        </div>
        <div style='display: flex; gap: 30px;'>
            <div style='flex: 1; border: 1px solid #d4edda; padding: 20px; background: #f8fdf8; border-radius: 8px;'>
                <h4 style='color: #155724;'>✅ Fortalezas</h4>
                <ul>{"".join(f"<li>{v}</li>" for v in offer['ventajas'][:4])}</ul>
            </div>
            <div style='flex: 1; border: 1px solid #f5c6cb; padding: 20px; background: #fdf5f6; border-radius: 8px;'>
                <h4 style='color: #721c24;'>❌ Debilidades</h4>
                <ul>{"".join(f"<li>{d}</li>" for d in offer['desventajas'][:4])}</ul>
            </div>
        </div>
    </div>
    """

def generate_html_results(results):
    html = ""
    for r in sorted(results, key=lambda x: x['porcentaje'], reverse=True):
        html += f"""
        <div style='border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 10px;'>
            <h3 style='color: #800000;'>📄 {r['aplicacion']}</h3>
            <p><strong>Puntuación:</strong> {r['porcentaje']:.2f}%</p>
            <div>
                <h4 style='color: #155724;'>✅ Fortalezas</h4>
                <ul>{"".join(f"<li>{v}</li>" for v in r['ventajas'])}</ul>
            </div>
            <div>
                <h4 style='color: #721c24;'>❌ Debilidades</h4>
                <ul>{"".join(f"<li>{d}</li>" for d in r['desventajas'])}</ul>
            </div>
        </div>
        """
    return html

def build_outputs(pliego_data, results):
    radar = generate_radar_chart(pliego_data)
    bar = generate_comparison_chart(results)
    best = max(results, key=lambda x: x['porcentaje'])
    return radar, bar, gr.HTML(generate_html_pliego(pliego_data)), gr.HTML(generate_html_results(results)), gr.HTML(generate_html_best_offer(best))


def waiting_for_process(pliego, submitted_bids, progress=gr.Progress()):
    progress(0, desc="Iniciando análisis...")

    with tempfile.TemporaryDirectory() as temp_dir:
        lic_path = save_pdf(pliego, temp_dir, "licitacion.pdf")
        progress(0.2, desc="Procesando licitación...")

        lic_embed, lic_data = analysis_doc(lic_path)

        results = []
        for idx, bid in enumerate(submitted_bids, 1):
            progress(0.2 + (0.6 * idx / len(submitted_bids)), desc=f"Analizando oferta {idx}/{len(submitted_bids)}...")
            result = analyze_and_compare_bid(bid, idx, temp_dir, lic_embed, lic_data)
            results.append(result)

    progress(0.9, desc="Generando visualizaciones...")
    return build_outputs(lic_data, results)

def crear_interfaz():
    with gr.Blocks(css="""
        body {
            background-color: #f4f4f4;
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
        }
        h1, h2, h3, h4 {
            color: #575757;
        }
        .header {
            background-color: #E3E3E3;
            padding: 30px 20px;
            border-radius: 0 0 15px 15px;
            color: white;
            text-align: center;
        }
        .upload-section {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 25px;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .result-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .boton-azul {
            background-color: #dbf7fc;
            color: #575757;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
        }
        .boton-azul:hover {
            background-color: #c8ebf1;
        }
        .upload-section label {
            font-size: 17px;
            font-weight: 500;
            color: #333333;
        }
    """) as interfaz:
        gr.HTML("""
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

            <div class="container py-4" style="background-color: #dbf7fc; border-radius: 0 0 12px 12px; color: #ffffff;">
                <div class="text-center">
                    <h1 class="display-5 fw-semibold">📊 Evaluador de Licitaciones Públicas</h1>
                    <p class="lead">Herramienta avanzada para analizar y comparar propuestas frente a una licitación oficial.</p>
                </div>
            </div>
        """)


        with gr.Row():
            with gr.Column(scale=1, elem_classes="upload-section"):
                pliego = gr.File(
                    label="📄 Documento de licitación",
                    file_types=[".pdf"],
                    type="binary"
                )
                submitted_bids = gr.Files(
                    label="📑 Ofertas para comparar",
                    file_types=[".pdf"],
                    type="binary"
                )
                boton_analizar = gr.Button(
                    "🔍 Ejecutar análisis",
                    elem_classes="boton-azul"
                )

        with gr.Tabs():
            with gr.TabItem("🏆 Mejor oferta"):
                mejor_oferta_html = gr.HTML(elem_classes="result-container")
            with gr.TabItem("📋 Resumen del Pliego de la Licitación"):
                pliego_html = gr.HTML(elem_classes="result-container")



        with gr.Row():
            grafico_radar = gr.Plot()
            grafico_comparativo = gr.Plot()

        with gr.Tabs():
            with gr.TabItem("📋 Resultados detallados"):
                resultados_html = gr.HTML(elem_classes="result-container")

        boton_analizar.click(
            fn=waiting_for_process,
            inputs=[pliego, submitted_bids],
            outputs=[grafico_radar, grafico_comparativo, pliego_html, resultados_html, mejor_oferta_html]
        )

    return interfaz


if __name__ == "__main__":
    app = crear_interfaz()
    #PARA TODOS 192.168.1.102
    #app.launch(server_name="0.0.0.0", server_port=5555, share=True)
    app.launch(server_port=5555, share=True)