# Memoria para asistentes conversacionales

## Introducción

Para el track de NLP proponemos el siguiente problema: Necesitamos crear un sistema para aumentar un asistente conversacional tipo chatbot con un modulo de memoria.

Para ayudarlxs a encarar el problema, preparamos este repo con un sistema de memoria basico basado en un retriever semantico. Recordemos que un retriever semantico puede pensarse como una funcion que, para una query Q y documentos D, devuelve la relevancia de cada documento condicionada a la query.

NO es necesario que basen su solucion en este repo, ni que sigan la estructura del mismo. Esto es solo para ayudarles a arrancar y si no les sirve ni lo miren.

## Estructura del proyecto

Si no te interesa, podes saltear directo a setup.

La carpeta principal del proyecto es `src`. Dentro de ella tienen `models`, `agents`, `datasets`, `experiments`.

- `models` tiene distintas implementaciones que les pueden servir de inspiracion. LiteLLM puede estar bueno si piensan probar modelos de distintas apis, ya que unifica su interfaz. A su vez, en caso de usar un modelo de huggingface, recomendamos el uso de Qwen3, ya que soporta reasoning y tiene muy buenas capacidades de uso de tools. Es por eso que incluimos una implementacion de QwenModel. Depende del hardware que esten usando y de su experiencia, pueden tambien implementarlo usando Ollama o VLLM.
- `agents` tiene distintas implementaciones de agentes. JudgeAgent es el que usamos para evaluar si la respuesta a una pregunta es correcta. FullContext es un agente que, dada una instancia de LongMemEval, le pasa todas las conversaciones a un modelo para que este intente responder. Si tienen acceso a un modelo con ventana de contexto muy grande, como gpt-5 o gemini, pueden intentar usarlo, aunque no lo recomendamos por no ser una solucion muy creativa y ser sumamente cara :). Finalmente RagAgente es el que implementa el modulo de rag que benchmarkeamos.
- En `datasets` tenemos una implementacion comoda que puede servirles para entender como leer el benchmark. Como veran, definimos la clase

```python
def instance_from_row(self, row):
    return LongMemEvalInstance(
        question_id=row["question_id"],
        question=row["question"],
        sessions=[
            Session(session_id=session_id, date=date, messages=messages)
            for session_id, date, messages in zip(
                row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
            )
        ],
        t_question=row["question_date"],
        answer=row["answer"],
    )
```

siguiendo la definicion del paper.

Finalmente en `experiments` esta la implementacion del experimento, junto con algunas utils que permiten cargar los modelos y modulos de memoria. En la carpeta `config` esta la configuracion que se usa para correr el expeirmento. La configuracion permite elegir que implementacion usar (fullcontext, rag, ... pueden agregar mas), que modelo usar para responder, que modelo usar como juez, etc...

## Setup

Si no usas `uv`, es muy recomendable. Podes descargarlo e instalarlo aca: https://docs.astral.sh/uv/getting-started/installation/

### Instalacion de uv y requierements

Una vez instalado, instala las dependencias usando

```sh
uv sync
```

Es posible que aun haciendo esto les salga el error de que torch no esta instalado. Eso es porque uv sync no va a instalar torch para que ustedes puedan instalarlo manualmente segun la version que necesiten (por ej con soporte para cuda)

Lo mismo con transformers.

La forma mas rapida de arreglar esto es

```
uv pip install torch transformers
```

### Descarga de datasets

Con las dependencias ya instaladas, descarga los datasets del benchmark usando

```sh
uv run scripts/download_dataset.py # Para descargar los datasets
```

o

```sh
source .venv/bin/activate # Para activar el virtual environment
python scripts/download_dataset.py
```

### Descarga de embeddings

Como vas a ver mas adelante en este ejemplo, uno de los modulos de memoria desarrollados usa embeddings. Es decir, para cada round conversacional (un mensaje de usuario seguido de uno de un agente), calculamos su embedding

[formula]

Y usamos esto para hacer retrieval dada una pregunta:

[ejemplo query, retrieval, respuesta]

Es posible que ustedes quieran usar algo similar, pero para arrancar nosotros les proveemos algunos embeddings precomputados para que puedan correr el ejemplo. Para descargar los embeddings precomputados, corran

```sh
TODO
```

### API Keys

Si eligen correr el benchmark usando una api, van a necesitar una api key. La mejor forma de hacerlo es creando un `.env` y configurando la `OPENAI_API_KEY`

### Correr el benchmark

Finalmente, para correr el benchmark:

```sh
uv run main.py
```

o

```sh
python main.py
```

### Analizar resultados

En `notebooks/rag_result_eval.ipynb` pueden ver un analisis bastante generico de los resultados. Como veran, los resultados estan segmentados segun tipo de pregunta. Esperamos que ustedes reporten los resultados de esa manera, porque los distintos tipos suelen tener distinta dificultad.
