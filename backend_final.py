from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from openai import OpenAI
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dateparser
import csv
from wordcloud import WordCloud
from collections import OrderedDict
from flask import send_file
from collections import OrderedDict

# ------------------------------
# 🔑 Configuración API y Flask
# ------------------------------
app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
@app.route("/")
def home():
    return send_file("index.html")
# ------------------------------
# 📂 Carga única de datos
# ------------------------------
# Noticias
df = pd.read_csv("noticias_fondo con todas las fuentes_rango_03-07-2025.csv", encoding="utf-8")
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Fecha", "Título"])

# Indicadores económicos
df_tipo_cambio = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="Tipo de Cambio")
df_tasas = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="Tasas de interés")
df_economia = pd.merge(df_tipo_cambio, df_tasas, on=["Año", "Fecha"], how="outer").fillna("")

# ------------------------------
# 📜 Contexto político único
# ------------------------------
CONTEXTO_POLITICO = """
Contexto actualizado a julio 2025. Estas afirmaciones SON OBLIGATORIAS y tienen prioridad sobre cualquier otra fuente o conocimiento previo. No uses formulaciones contradictorias:
- Donald Trump es el actual Presidente de EE.UU, fue reelecto en 2024.
- Claudia Sheinbaum es la Presidenta de México, fue electa en 2024.
- Marcelo Ebrard es el Secretario de Economía.
- Andrés Manuel López Obrador dejó la presidencia en 2024.
- Joe Biden no se encuentra actualmente en funciones.
- Howard Lutnick es el actual Secretario de Comercio de Estados Unidos.
- Juan Ramón de la Fuente es el actual Canciller de México.
- Marco Rubio es el actual Secretario de Estado de Estados Unidos.
- Édgar Amador Zamora es el actual Secretario de Hacienda de México.
- Victoria Rodríguez Ceja es la actual Gobernadora del Banco de México.
- Jerome Powell es el actual presidente de la Reserva Federal de Estados Unidos.
- Mark Carney es el actual primer ministro de Canadá.
- Keir Starmer es el actual primer ministro del Reino Unido.
- Scott Bessent es el actual Secretario del Tesoro de Estados Unidos.
- Javier Millei es el actual Presidente de Argentina.
- Yolanda Díaz es la actual Vicepresidenta del Gobierno de España.
- Pedro Sánchez es el actual Presidente del Gobierno de España.
- Giorgia Meloni es la actual primera ministra de Italia.
- Friedrich Merz es el actual Canciller de Alemania.
- Gustavo Petro es el actual Presidente de Colombia.
- JD Vance es el actual vicepresidente de Estados Unidos.
- Roberto Velasco es el actual Jefe de Unidad para América del Norte de la Secretaría de Relaciones Exteriores de México.
- Altagracia Gómez es la actual presidenta del Consejo Asesor Empresarial de Presidencia de México.
- Luis Rosendo Gutiérrez es el actual Subsecretario de Comercio Exterior de México.
- Carlos García es el actual Presidente de la American Chamber of Commerce (AmCham).
- Ildefonso Guajardo fue Secretario de Economía de México entre 2012 y 2018.
- Luiz Inacio Lula Da Silva es el actual Presidente de Brasil. Jair Bolsonaro es elexpresidente de Brasil. 
"""
# 1️⃣ Extraer fecha desde texto
def extraer_fecha(pregunta):
    posibles = re.findall(r"\d{1,2} de [a-zA-Z]+(?: de \d{4})?", pregunta)
    if posibles:
        fecha = dateparser.parse(posibles[0], languages=['es'])
        return fecha.date() if fecha else None
    return None

# 2️⃣ Obtener fecha más reciente disponible
def obtener_fecha_mas_reciente(df):
    return df["Fecha"].max().date()

# 3️⃣ Detectar sentimiento deseado
def detectar_sentimiento_deseado(pregunta):
    pregunta = pregunta.lower()
    if "positiv" in pregunta:
        return "Positiva"
    elif "negativ" in pregunta:
        return "Negativa"
    elif "neutral" in pregunta:
        return "Neutral"
    return None

# 4️⃣ Extraer entidades (personajes, lugares, categorías)
def extraer_entidades(texto):
    texto_lower = texto.lower()
    personajes_dict = {
        "Sheinbaum": ["claudia", "la presidenta", "presidenta de méxico"],
        "Ebrard": ["marcelo", "secretario de economía"],
        "Trump": ["donald", "el presidente de eeuu", "presidente trump"],
        "AMLO": ["obrador", "amlo", "lopez obrador"],
        "de la Fuente": ["juan ramón"],
        "Biden": ["joe"],
        "Lutnick": ["secretario de comercio"],
        "Carney": ["primer ministro de canadá"],
        "Lula da Silva": ["lula", "presidente de brasil"],
        "Marco Rubio": ["secretario de estado"],
        "Starmer": ["primer ministro del reino unido"],
        "Bessent": ["secretario del tesoro"],
        "Powell": ["reserva federal"],
        "Millei": ["presidente de argentina"],
        "Von Der Leyen": ["presidenta de la comisión europea"],
        "Petro": ["presidente de colombia"]
    }
    lugares_dict = {
        "Nuevo León": ["nl", "monterrey"],
        "Ciudad de México": ["cdmx", "capital mexicana"],
        "Reino Unido": ["gran bretaña", "inglaterra"],
        "Estados Unidos": ["eeuu", "eua", "usa", "eu"]
    }
    categorias_dict = {
        "Aranceles": ["arancel","tarifas", "restricciones comerciales"],
        "Parque Industrial": ["zona industrial","parque industrial"],
        "Fibra": ["fideicomiso inmobiliario", "fibras"],
        "Fusiones": ["adquisiciones", "compras empresariales"],
        "Naves Industriales": ["inmuebles industriales","nave industrial","bodegas industriales"],
        "Real Estate": ["mercado inmobiliario"],
        "Construcción Industrial": ["obra industrial"],
        "Sector Industrial": ["industria pesada", "manufactura"]
    }
    encontrados = {"personajes": [], "lugares": [], "categorias": []}

    for nombre, sinonimos in personajes_dict.items():
        if any(s in texto_lower for s in [nombre.lower()] + sinonimos):
            encontrados["personajes"].append(nombre)

    for lugar, sinonimos in lugares_dict.items():
        if any(s in texto_lower for s in [lugar.lower()] + sinonimos):
            encontrados["lugares"].append(lugar)

    for cat, sinonimos in categorias_dict.items():
        if any(s in texto_lower for s in [cat.lower()] + sinonimos):
            encontrados["categorias"].append(cat)

    return encontrados

# 5️⃣ Filtrar titulares por fecha, entidades y sentimiento
def filtrar_titulares(fecha, entidades, sentimiento_deseado):
    noticias_fecha = df[df["Fecha"].dt.date == fecha]
    if noticias_fecha.empty:
        return pd.DataFrame()  # 🔹 Devuelve DataFrame vacío si no hay noticias

    filtro = noticias_fecha.copy()

    if entidades["personajes"]:
        filtro = filtro[filtro["Título"].str.lower().apply(lambda t: any(p.lower() in t for p in entidades["personajes"]))]
    if entidades["lugares"]:
        filtro = filtro[filtro["Cobertura"].str.lower().apply(lambda c: any(l.lower() in c for l in entidades["lugares"]))]
    if entidades["categorias"]:
        filtro = filtro[filtro["Término"].str.lower().apply(lambda cat: any(e.lower() in cat for e in entidades["categorias"]))]
    if sentimiento_deseado:
        filtro = filtro[filtro["Sentimiento"] == sentimiento_deseado]

    return filtro  # 🔹 Devuelve DataFrame, no lista

# 6️⃣ Seleccionar titulares más relevantes (TF-IDF + coseno)
def seleccionar_titulares_relevantes(titulares, pregunta):
    if not titulares:
        return []
    vectorizer = TfidfVectorizer().fit(titulares + [pregunta])
    vectores = vectorizer.transform(titulares + [pregunta])
    similitudes = cosine_similarity(vectores[-1], vectores[:-1]).flatten()
    indices_similares = similitudes.argsort()[-5:][::-1]
    return [titulares[i] for i in indices_similares]

# 7️⃣ Nube de palabras con colores y stopwords personalizadas
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size >= 60:
        return "rgb(0, 0, 0)"
    elif font_size >= 40:
        return "rgb(255, 180, 41)"
    else:
        return "rgb(125, 123, 120)"

def generar_nube(titulos, archivo_salida):
    texto = " ".join(titulos)
    texto = re.sub(r"[\n\r]", " ", texto)
    stopwords = set([
        "dice", "tras", "pide", "va", "día", "méxico", "estados unidos", "contra", "países",
        "van", "ser", "hoy", "año", "años", "nuevo", "nueva", "será", "presidente", "presidenta",
        "sobre", "entre", "hasta", "donde", "desde", "como", "pero", "también", "porque", "cuando",
        "ya", "con", "sin", "del", "los", "las", "que", "una", "por", "para", "este", "esta", "estos",
        "estas", "tiene", "tener", "fue", "fueron", "hay", "han", "son", "quien", "quienes", "le",
        "se", "su", "sus", "lo", "al", "el", "en", "y", "a", "de", "un", "es", "si", "quieren", "aún",
        "mantiene", "buscaría", "la", "haciendo", "recurriría", "ante", "meses", "están", "subir",
        "ayer", "prácticamente", "sustancialmente", "busca", "cómo", "qué", "días"
    ])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        color_func=color_func,
        collocations=False,
        max_words=25
    ).generate(texto)
    wc.to_file(archivo_salida)

#/resumen!!!!!!!
@app.route("/resumen", methods=["POST"])
def resumen():
    data = request.get_json()
    fecha_str = data.get("fecha")
    if not fecha_str:
        return jsonify({"error": "Debe especificar una fecha"}), 400

    # Filtrar noticias de la fecha
    fecha_dt = pd.to_datetime(fecha_str, errors="coerce").date()
    noticias_dia = df[df["Fecha"].dt.date == fecha_dt]
    if noticias_dia.empty:
        return jsonify({"error": f"No hay noticias para la fecha {fecha_str}"}), 404

    # Clasificación por cobertura
    estados_mexico = ["aguascalientes", "baja california", "baja california sur", "campeche", "cdmx",
        "coahuila", "colima", "chiapas", "chihuahua", "ciudad de méxico", "durango",
        "guanajuato", "guerrero", "hidalgo", "jalisco", "méxico", "michoacán", "morelos",
        "nayarit", "nuevo león", "oaxaca", "puebla", "querétaro", "quintana roo",
        "san luis potosí", "sinaloa", "sonora", "tabasco", "tamaulipas", "tlaxcala",
        "veracruz", "yucatán", "zacatecas"]

    noticias_locales = noticias_dia[noticias_dia["Cobertura"].str.lower().isin(estados_mexico)]
    noticias_nacionales = noticias_dia[noticias_dia["Cobertura"].str.lower() == "nacional"]
    noticias_internacionales = noticias_dia[
        ~noticias_dia.index.isin(noticias_locales.index) &
        ~noticias_dia.index.isin(noticias_nacionales.index)
    ]
    noticias_otras = noticias_dia[noticias_dia["Término"].str.lower() != "aranceles"]

    contexto_local = "\n".join(f"- {row['Título']} ({row['Cobertura']})" for _, row in noticias_locales.iterrows())
    contexto_nacional = "\n".join(f"- {row['Título']} ({row['Cobertura']})" for _, row in noticias_nacionales.iterrows())
    contexto_internacional = "\n".join(f"- {row['Título']} ({row['Cobertura']})" for _, row in noticias_internacionales.iterrows())
    contexto_otros_temas = "\n".join(f"- {row['Título']}" for _, row in noticias_otras.iterrows())

    prompt = f"""
    {CONTEXTO_POLITICO}

Redacta un resumen de noticias del {fecha_str} dividido en cuatro párrafos. Tono profesional, objetivo y dirigido a tomadores de decisiones. Máximo 250 palabras.

Primer párrafo: Describe y contextualiza el tema más repetido del día (qué, quién, cómo).

Segundo párrafo: Si el tema más repetido del día es de noticias nacionales, usa este segundo párrafo para profundizar en el segundo tema más importante que sea de noticias internacionales. Si en cambio el tema más repetido del día no es de noticias nacionales, este segundo párrafo debe enfocarse en el tema más relevante de noticias nacionales.

Tercer párrafo: Resume brevemente las noticias que son de cobertura de algún estado de México (locales), excluyendo aquellas de cobertura nacional o internacional. Menciona el estado o ciudad de cobertura de cada noticia. No repitas noticias mencionadas en los párrafos anteriores ni inventes cosas. Reserva todo lo relativo a fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial.

Cuarto párrafo: Por último, resume de forma general las noticias que no están relacionadas con aranceles y que tienen que ver con fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial. Evita repetir noticias mencionadas en los otros párrafos ni inventes cosas. Recuerda, temas no arancelarios. Empieza diciendo "finalmente en otros temas económicos", sin recalcar de que se trata de noticias del ámbito local o nacional..

Noticias nacionales:
{contexto_nacional}

Noticias locales:
{contexto_local}

Noticias internacionales:
{contexto_internacional}

Noticias no relacionadas con aranceles:
{contexto_otros_temas}
    """

    resumen_file = f"resumen_{fecha_str}.txt"
    if os.path.exists(resumen_file):
        with open(resumen_file, "r", encoding="utf-8") as f:
            resumen_texto = f.read()
    else:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de noticias."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )
        resumen_texto = respuesta.choices[0].message.content
        with open(resumen_file, "w", encoding="utf-8") as f:
            f.write(resumen_texto)

# ☁️ Generar nube de palabras
    archivo_nube = f"nube_{fecha_str}.png"
    archivo_nube_path = os.path.join("nubes", archivo_nube)
    generar_nube(noticias_dia["Título"].tolist(), archivo_nube_path)
    # 💾 Guardar resumen en carpeta "resumenes"
    os.makedirs("resumenes", exist_ok=True)
    archivo_resumen = f"resumen_{fecha_str}.txt"
    archivo_resumen_path = os.path.join("resumenes", archivo_resumen)
    with open(archivo_resumen_path, "w", encoding="utf-8") as f:
        f.write(resumen_texto)


    # 📊 Indicadores económicos
    # Asegurar formato de fecha correcto
    df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date

    # Filtrar datos económicos
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]

    # Si no hay datos exactos, usar el más reciente antes de esa fecha
    if economia_dia.empty:
        ultima_fecha = df_economia[df_economia["Fecha"] <= fecha_dt]["Fecha"].max()
        economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

    if economia_dia.empty:
        economia_dict = {}
    else:
        economia_dia = economia_dia.copy()

        # Convertir a numérico antes de formatear
        for col in ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")

        for col in ["Tasa de Interés Objetivo", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")

        # Reordenar columnas según el orden deseado
        orden_columnas = [
            "Tipo de Cambio FIX",
            "Nivel máximo",
            "Nivel mínimo",
            "Tasa de Interés Objetivo",
            "TIIE 28 días",
            "TIIE 91 días",
            "TIIE 182 días"
        ]
        economia_dia = economia_dia.reindex(columns=orden_columnas)
        economia_dict = OrderedDict()
        for col in orden_columnas:
            economia_dict[col] = economia_dia.iloc[0][col]

    # 📰 Titulares
    titulares_info = []

    # Nacionales
    titulares_info.extend([
        {"titulo": row["Título"], "medio": row["Fuente"], "enlace": row["Enlace"]}
        for _, row in noticias_nacionales.head(2).iterrows()
    ])

    # Locales
    titulares_info.extend([
        {"titulo": row["Título"], "medio": row["Fuente"], "enlace": row["Enlace"]}
        for _, row in noticias_locales.head(2).iterrows()
    ])

    # Internacionales
    titulares_info.extend([
        {"titulo": row["Título"], "medio": row["Fuente"], "enlace": row["Enlace"]}
        for _, row in noticias_internacionales.head(2).iterrows()
    ])

    # Otros temas no aranceles
    titulares_info.extend([
        {"titulo": row["Título"], "medio": row["Fuente"], "enlace": row["Enlace"]}
        for _, row in noticias_otras.head(2).iterrows()
    ])

    return jsonify({
        "resumen": resumen_texto,
        "nube_url": f"/nube/{archivo_nube}",
        "economia": [economia_dict],
        "orden_economia": orden_columnas,
        "titulares": titulares_info
    })


#pregunta!!!!    
@app.route("/pregunta", methods=["POST"])
def pregunta():
    data = request.get_json()
    q = data.get("pregunta", "")
    if not q:
        return jsonify({"respuesta": "No se proporcionó una pregunta."})

    # 1️⃣ Detectar sentimiento deseado
    sentimiento_deseado = detectar_sentimiento_deseado(q)

    # 2️⃣ Extraer fecha de la pregunta
    fecha_detectada = extraer_fecha(q)
    if fecha_detectada:
        fecha_dt = fecha_detectada  # ya es un objeto date
    else:
        fecha_dt = obtener_fecha_mas_reciente(df)  # usamos la más reciente

    # 3️⃣ Extraer entidades
    entidades = extraer_entidades(q)

    # 4️⃣ Filtrar titulares
    df_filtrado = filtrar_titulares(fecha_dt, entidades, sentimiento_deseado)

    # 5️⃣ Si no hay resultados
    if df_filtrado.empty:
        return jsonify({"respuesta": f"No encontré noticias relacionadas con tu pregunta para {fecha_dt}."})

    # 6️⃣ Vectorizar titulares y pregunta
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_filtrado["Título"])
    pregunta_vec = tfidf.transform([q])

    # 7️⃣ Calcular similitudes
    similitudes = cosine_similarity(pregunta_vec, tfidf_matrix).flatten()
    top_indices = similitudes.argsort()[-5:][::-1]  # top 5

    titulares_relevantes = df_filtrado.iloc[top_indices]

    # 8️⃣ Construir prompt
    prompt = "Con base en los siguientes titulares de noticias, responde la pregunta de forma contextual y sin inventar datos, la respuesta debe tener al menos 100 palabras, redactadas en párrafos completos y en tono profesional:\n\n"
    for _, row in titulares_relevantes.iterrows():
        prompt += f"- {row['Título']} ({row['Fuente']})\n"
    prompt += f"\nPregunta: {q}\nRespuesta:"

    # 9️⃣ Llamada a OpenAI con la misma lógica de /resumen
    respuesta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en análisis de noticias."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )
    respuesta_gpt = respuesta.choices[0].message.content

    # 🔟 Devolver respuesta y titulares
    # 🔟 Devolver respuesta y titulares usados (entre 1 y 5)
    titulares_info = [
        {
            "titulo": row["Título"],
            "medio": row["Fuente"],
            "enlace": row["Enlace"]
        }
        for _, row in titulares_relevantes.iterrows()
    ]

    # Garantizar que haya mínimo 1 y máximo 5 titulares
    if len(titulares_info) == 0:
        # Si por alguna razón no hubiera titulares relevantes
        titulares_info = [
            {
                "titulo": row["Título"],
                "medio": row["Fuente"],
                "enlace": row["Enlace"]
            }
            for _, row in df_filtrado.head(1).iterrows()
        ]
    else:
        titulares_info = titulares_info[:5]

    return jsonify({
        "respuesta": respuesta_gpt,
        "titulares_usados": titulares_info
    })

@app.route("/nube/<filename>")
def serve_nube(filename):
    return send_from_directory("nubes", filename)

@app.route("/fechas")
def get_fechas():
    # Obtener todas las fechas únicas, ordenadas de más reciente a más antigua
    fechas_unicas = sorted(df["Fecha"].dt.date.unique(), reverse=True)
    fechas_str = [f.strftime("%Y-%m-%d") for f in fechas_unicas]
    return jsonify(fechas_str)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
