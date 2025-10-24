import streamlit as st
import json
import google.generativeai as genai
from google.api_core import exceptions

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
MODEL_NAME = "gemma-3-27b-it"
API_KEY = st.secrets["API_KEY"]

st.set_page_config(page_title="Puntos Críticos", page_icon="👨‍🏫")

# --- 2. CARGA DE DATOS ---

@st.cache_data
def cargar_ejercicios(ruta_archivo="extremos.json"):
    """Carga los ejercicios desde el archivo JSON de forma segura."""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error crítico: El archivo de base de datos '{ruta_archivo}' no se encontró.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error crítico: El formato del archivo '{ruta_archivo}' es un JSON inválido.")
        return None

EJERCICIOS = cargar_ejercicios()

# --- 3. FUNCIONES DE LA IA (CON EL PROMPT DEFINITIVO) ---

def generar_contexto_db(ejercicios):
    """Formatea la lista de ejercicios del JSON para que el modelo sepa qué puede enseñar."""
    if not ejercicios:
        return "No hay ejercicios cargados en la base de datos."
    
    contexto = "BASE DE DATOS DE EJERCICIOS QUE PUEDES ENSEÑAR:\n\n"
    for ej in ejercicios:
        contexto += f"--- EJERCICIO ID: {ej['id']} ---\n"
        contexto += f"FUNCION: {ej['function']}\n"
        contexto += f"DERIVADA FINAL: {ej['derivative']}\n"
        contexto += f"CONCLUSIÓN (OBJETIVO): {ej['conclusion']}\n"
        contexto += "------------------------\n"
    return contexto

# <--- INICIO ---
def obtener_respuesta_ia(historial, ejercicios):
    """
    Genera una respuesta como un tutor de cálculo con reglas de formato extremadamente estrictas.
    """

    system_instruction = f"""
    Eres un tutor de cálculo 1. Tu tarea es enseñar a los estudiantes a resolver problemas relacionados con los extremos relativos de funciones. Debes seguir estos pasos:

    1. **Definir los extremos relativos**: Explicar qué son los extremos relativos, incluyendo la definición de máximos y mínimos relativos.
    2. **Mostrar los pasos para encontrar extremos relativos**:
        - Encontrar la derivada de la función.
        - Resolver la ecuación de la derivada igualada a cero para encontrar los puntos críticos.

    REGLAS ESTRICTAS DE OPERACIÓN (SIN EXCEPCIÓN):
    1.  **ENSEÑANZA FOCALIZADA:** Tu conocimiento se limita ESTRICTAMENTE a los ejercicios de la base de datos. Si un usuario te pide derivar cualquier otra función o pregunta por otros temas, niégate cortésmente.
    2.  **INTERACTIVIDAD:** Debes poder responder preguntas de seguimiento sobre los pasos o las reglas de derivación utilizadas en un ejercicio.
    3.  **REGLA MAESTRA DE FORMATO (LA MÁS IMPORTANTE):**
        - Está **ABSOLUTAMENTE PROHIBIDO** escribir texto y expresiones matemáticas en la misma línea.
        - CUALQUIER expresión matemática, por pequeña que sea (una variable, un número, una función completa), debe estar en su propia línea separada dentro de un bloque matemático de Markdown.
        - El único formato matemático permitido es el de bloque: st.markdown($$ ... $$). El formato en línea ($ ... $) está prohibido.
        - Debes enumerar los pasos que sigues en cada a proceso. Paso 1, Paso 2, etc.

        **Ejemplo: Encontrar los extremos relativos de la función \( f(x) = 3x^3 - 12x^2 + 3 \)**

    ### Paso 1: Derivar la función

    La primera tarea es calcular la derivada de la función. Aplicamos la regla de la potencia a cada término:

    $$
    f'(x) = 9x^2 - 24x
    $$

    ### Paso 2: Encontrar los puntos críticos

    Para encontrar los puntos críticos, igualamos la derivada a cero:

    $$
    f'(x) = 9x^2 - 24x = 0
    $$

    Factorizamos:

    $$
    x(9x - 24) = 0
    $$

    Las soluciones son:

    $$
    x = 0 \quad \text{"y"} \quad x = \frac{"8"}{"3"}
    $$

    Por lo tanto, los puntos críticos son x = 0 y  x = 8/3.
    
    
    {generar_contexto_db(ejercicios)}
    """

    model = genai.GenerativeModel(model_name=MODEL_NAME)

    full_prompt_history = [
        {"role": "user", "parts": [system_instruction]},
        {"role": "model", "parts": ["Entendido. Mi regla principal es el formato. JAMÁS escribiré matemáticas en la misma línea que el texto. Cada expresión matemática, sin importar su tamaño, irá en su propio bloque $$...$$."]} 
    ]

    gemini_history = [
        {"role": "model" if msg["role"] == "assistant" else "user", "parts": [msg["content"]]}
        for msg in historial
    ]
    full_prompt_history.extend(gemini_history)

    try:
        response = model.generate_content(full_prompt_history)
        if not response.parts:
            return "El modelo no generó una respuesta. Esto puede ocurrir debido a los filtros de seguridad."
        return response.text
    except exceptions.PermissionDenied:
        st.error(f"Error de Permiso: API Key no válida.")
        return "No pude conectarme. Por favor, verifica tu API Key."
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al contactar la API: {e}")
        return "Lo siento, estoy teniendo problemas técnicos."

# --- 4. INTERFAZ DE STREAMLIT (SIN CAMBIOS) ---
st.title("👨‍🏫 Profesor Interactivo. Cálculo 1")
st.caption("Elige un ejercicio y te enseñaré a resolverlo. ¡Pregúntame lo que necesites!")

if API_KEY:
    genai.configure(api_key=API_KEY)

# ---AÑADIR BOTONES DE CONTROL --- 
if st.sidebar.button("Reiniciar Chat"):
    st.session_state.messages = []
    st.rerun()

if not EJERCICIOS:
    st.warning("La aplicación no puede iniciar porque la base de datos de ejercicios no se pudo cargar.")
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Ejercicios. Hallar puntos críticos")
    for ej in EJERCICIOS:
        st.sidebar.markdown(f"**ID {ej['id']}**: {ej['function']}")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu profesor de cálculo. Elige un ejercicio de la lista (por su ID) y te enseñaré a resolverlo paso a paso. ¡Puedes hacerme preguntas en cualquier momento!"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe el ID del ejercicio o haz una pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("El profesor está preparando la lección..."):
                respuesta_ia = obtener_respuesta_ia(st.session_state.messages, EJERCICIOS)
                st.markdown(respuesta_ia)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})

