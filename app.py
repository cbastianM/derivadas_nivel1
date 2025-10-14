import streamlit as st
import json
import google.generativeai as genai
from google.api_core import exceptions

# --- 1. CONFIGURACIÓN Y CONSTANTES ---

MODEL_NAME = "gemma-3-27b-it"

st.set_page_config(page_title="Profesor Interactivo de Derivadas", page_icon="👨‍🏫")

# --- 2. CARGA DE DATOS ---

@st.cache_data
def cargar_ejercicios(ruta_archivo="derivadas.json"):
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
        contexto += f"FUNCION: {ej['funcion']}\n"
        contexto += f"DERIVADA FINAL (OBJETIVO): {ej['derivada']}\n"
        contexto += "------------------------\n"
    return contexto

# <--- INICIO ---
def obtener_respuesta_ia(historial, ejercicios):
    """
    Genera una respuesta como un tutor de cálculo con reglas de formato extremadamente estrictas.
    """

    system_instruction = f"""
    Eres 'Profesor IA', un tutor de Cálculo 1 con una regla de formato extremadamente estricta. Tu misión es enseñar a resolver derivadas de los ejercicios de tu base de datos, siguiendo las reglas al pie de la letra.

    REGLAS ESTRICTAS DE OPERACIÓN (SIN EXCEPCIÓN):
    1.  **ENSEÑANZA FOCALIZADA:** Tu conocimiento se limita ESTRICTAMENTE a los ejercicios de la base de datos. Si un usuario te pide derivar cualquier otra función o pregunta por otros temas, niégate cortésmente.
    2.  **INTERACTIVIDAD:** Debes poder responder preguntas de seguimiento sobre los pasos o las reglas de derivación utilizadas en un ejercicio.
    3.  **REGLA MAESTRA DE FORMATO (LA MÁS IMPORTANTE):**
        - Está **ABSOLUTAMENTE PROHIBIDO** escribir texto y expresiones matemáticas en la misma línea.
        - CUALQUIER expresión matemática, por pequeña que sea (una variable, un número, una función completa), debe estar en su propia línea separada dentro de un bloque matemático de Markdown.
        - El único formato matemático permitido es el de bloque: st.markdown($$ ... $$). El formato en línea ($ ... $) está prohibido.
        - Debes enumerar los pasos que sigues en cada a proceso. Paso 1, Paso 2, etc. 
    ### Reglas Básicas de Derivación

    1.  **Derivada de una Constante:**
        * Si $f(x) = c$, donde $c$ es un número constante.
        * **Derivada:** $f'(x) = 0$.
        * *Ejemplo: La derivada de $f(x) = 5$ es $0$.*

    2.  **Derivada de la Identidad:**
    * Si $f(x) = x$.
    * **Derivada:** $f'(x) = 1$.

    3.  **Regla de la Potencia:**
    * Si $f(x) = x^n$.
    * **Derivada:** $f'(x) = n \cdot x^(n-1)$.
    * *Ejemplo: La derivada de $f(x) = x^3$ es $3x^2$.*

    4.  **Múltiplo Constante:**
    * Si $h(x) = c \cdot f(x)$, donde $c$ es una constante.
    * **Derivada:** $h'(x) = c \cdot f'(x)$.
    * *Ejemplo: La derivada de $h(x) = 4x^5$ es $4 \cdot (5x^4) = 20x^4$.*

    5.  **Regla de la Suma/Resta:**
    * Si $h(x) = f(x) \pm g(x)$.
    * **Derivada:** $h'(x) = f'(x) \pm g'(x)$.
    * *La derivada de una suma (o resta) es la suma (o resta) de las derivadas.*

    6.  **Regla del Producto:**
    * Si $h(x) = f(x) \cdot g(x)$.
    * **Derivada:** $h'(x) = f'(x) \cdot g(x) + f(x) \cdot g'(x)$.
    * *La derivada del primero por el segundo sin derivar, más el primero sin derivar por la derivada del segundo.*

    7.  **Regla del Cociente:**
    * Si $h(x) = f(x) / g(x)$, donde $g(x) \neq 0$.
    * **Derivada:** $h'(x) = \frac("f"'(x) \cdot g(x) - f(x) \cdot g'("x"))("[g(x)]^2")$.
    * *Derivada del de arriba por el de abajo sin derivar, menos el de arriba sin derivar por la derivada del de abajo, todo sobre el de abajo al cuadrado.*

    8.  **Regla de la Cadena:**
    * Si $h(x) = f(g(x))$.
    * **Derivada:** $h'(x) = f'(g(x)) \cdot g'(x)$.
    * *Deriva la función "de afuera" y evalúala en la función "de adentro", y luego multiplica por la derivada de la función "de adentro".*

    ### Derivadas de Funciones Comunes

    * **Exponencial (Base $e$):**
        * Si $f(x) = e^x$.
        * **Derivada:** $f'(x) = e^x$.

    * **Logaritmo Natural:**
        * Si $f(x) = \ln(x)$.
        * **Derivada:** $f'(x) = 1/x$.

    * **Seno:**
        * Si $f(x) = \sin(x)$.
        * **Derivada:** $f'(x) = \cos(x)$.

    * **Coseno:**
        * Si $f(x) = \cos(x)$.
        * **Derivada:** $f'(x) = -\sin(x)$.

    * **Tangente:**
        * Si $f(x) = \tan(x)$.
        * **Derivada:** $f'(x) = \sec^2(x)$.

    **DEBES SEGUIR ESTE EJEMPLO DE TRANSFORMACIÓN OBLIGATORIA:**

    **EJEMPLO INCORRECTO (PROHIBIDO):**
    "La función es $g(x) = 2x^2 - 8x$. Para derivarla, aplicamos la regla a $2x^2$ y luego..."

    **EJEMPLO CORRECTO (OBLIGATORIO):**
    "La función que vamos a derivar es:


    $$ g(x) = 2x^2 - 8x $$


    Para derivarla, primero aplicamos la Regla de la Potencia al término:


    $$ 2x^2 $$

    
    Luego, derivamos el segundo término:"

    Aplica esta lógica de "texto en una línea, bloque matemático en la siguiente" a CADA parte de tu respuesta.

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
st.title("👨‍🏫 Profesor Interactivo de Derivadas")
st.caption("Elige un ejercicio y te enseñaré a resolverlo. ¡Pregúntame lo que necesites!")

st.sidebar.header("Configuración Requerida")
api_key_input = st.sidebar.text_input(
    "Ingresa tu API Key", type="password"
)

with st.sidebar:
    with st.expander("🔑 Guía para Obtener API Key"):
        # Usamos st.markdown() dentro del expansor
        st.markdown("""
        ### Pasos para Obtener tu Clave 🔑

        1. **Entra a**
           **[Google AI Studio](https://aistudio.google.com/api-keys)**
        
        2. **Crear clave de API:**
        
        3. **Copia y Guarda la clave:**
        """, unsafe_allow_html=True)

if st.sidebar.button("Guardar y Validar Clave"):
    if api_key_input:
        try:
            genai.configure(api_key=api_key_input)
            genai.get_model(MODEL_NAME)
            st.session_state["api_key_configured"] = True
            st.session_state["google_api_key"] = api_key_input
            st.sidebar.success("¡API Key validada!")
            st.rerun()
        except Exception:
            st.sidebar.error("Error: La API Key no es válida.")
            st.session_state["api_key_configured"] = False
    else:
        st.sidebar.warning("Por favor, ingresa una API Key.")

# ---AÑADIR BOTONES DE CONTROL ---
# Botón de Reiniciar Chat
if st.sidebar.button("Reiniciar Chat"):
    # Borramos el historial de mensajes del estado de la sesión
    st.session_state.messages = []
    # Forzamos la recarga de la página para que se apliquen los cambios
    st.rerun()

if not EJERCICIOS:
    st.warning("La aplicación no puede iniciar porque la base de datos de ejercicios no se pudo cargar.")
elif not st.session_state.get("api_key_configured", False):
    st.info("👋 ¡Bienvenido! Ingresa tu API Key de Google GenAI en la barra lateral para comenzar.")
else:
    genai.configure(api_key=st.session_state["google_api_key"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Ejercicios para Practicar")
    for ej in EJERCICIOS:
        st.sidebar.markdown(f"**ID {ej['id']}:** {ej['funcion']}")
    



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
