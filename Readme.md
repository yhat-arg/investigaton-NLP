# Memoria para asistentes conversacionales

## Introducción

En un futuro cercano, los asistentes conversacionales correrán directamente *en tu celular*: modelos de LLMs pequeños, eficientes y capaces de recordar tus hábitos, tus gustos y el contexto de tus últimas conversaciones. Para que ese tipo de agentes sea posible, necesitan **módulos de memoria optimizados**: rápidos, baratos de ejecutar y sin depender de modelos gigantes.  
Este track propone justamente eso: **explorar cómo diseñar el mejor sistema de memoria para agentes conversacionales usando modelos pequeños de LLMs**, comparando distintas estrategias y evaluando su eficiencia y calidad.

Este repositorio acompaña el track de NLP del Investigathon de YHat y plantea el desafío de ampliar un asistente conversacional con un módulo de memoria.  
Incluimos una referencia básica basada en un *semantic retriever*, que servirá como baseline.

Un **retriever semántico** o **RAG** puede pensarse como una función que, dada una consulta `Q` y un conjunto de documentos `D`, calcula un embedding para cada documento y devuelve los **top-k documentos más relevantes** según su similitud con la consulta. 

La utilización de este proyecto es completamente opcional: pueden usarlo tal cual, adaptarlo o simplemente tomarlo como fuente de inspiración.

> **Nota:** la explicación completa del benchmark, el formato de las instancias y los criterios de evaluación (incluyendo cómo evaluamos correctitud y memoria) está en `benchmark_explanation.md`.


## Estructura del proyecto

La carpeta principal es `src`. Allí van a encontrar:

- **`models`**: implementaciones de referencia. `LiteLLM` simplifica la prueba de múltiples APIs al unificar su interfaz. Si trabajan con modelos de Hugging Face sugerimos Qwen3, que ofrece buen *reasoning* y soporte de *tools*; por eso incluimos `QwenModel`. Dependiendo del hardware y la experiencia, también pueden evaluar Ollama o vLLM.
- **`agents`**: distintos agentes ya configurados. `JudgeAgent` evalúa si la respuesta es correcta. `FullContextAgent` envía la instancia completa de LongMemEval a un modelo con ventana de contexto amplia (por ejemplo GPT-5 o Gemini); es una alternativa directa pero costosa y poco creativa. `RAGAgent` implementa el módulo de RAG que usamos para el benchmark.
- **`datasets`**: utilidades para cargar y representar el benchmark. Incluye la clase `LongMemEvalInstance`, alineada con la definición del paper.

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

- **`experiments`**: implementación del pipeline experimental y utilidades para cargar modelos y módulos de memoria. En `config` pueden definir qué agente usar (`fullcontext`, `rag`, etc.), qué modelo responde, qué modelo actúa como juez y otros parámetros.


## Setup

Recomendamos utilizar `uv` para gestionar el entorno. Podés descargarlo e instalarlo desde <https://docs.astral.sh/uv/getting-started/installation/>.

### Instalación de dependencias

Una vez instalado `uv`, sincronizá las dependencias:

```sh
uv sync
```

Es posible que `torch` y `transformers` no se instalen automáticamente para permitirles elegir versiones específicas (por ejemplo, con soporte CUDA). La instalacion mas basica es mediante:

```
uv pip install torch transformers
```

### Descarga de datasets

Con el entorno configurado, descargá los datasets del benchmark:

```sh
uv run scripts/download_dataset.py
```

Alternativamente, activá el entorno virtual y ejecutá el script manualmente:

```sh
source .venv/bin/activate
python scripts/download_dataset.py
```

### Descarga de embeddings

Uno de los módulos de memoria incluidos utiliza embeddings. Para cada mensaje se calcula un embedding:

[formula]

Luego se utiliza para realizar *retrieval*:

[ejemplo query, retrieval, respuesta]

Incluimos embeddings precomputados para acelerar las primeras ejecuciones, pueden encontrarlos aca: https://drive.google.com/file/d/1V2IzQVtQhpUhUCLDxFmGz2Xf4X-B-1et/view?usp=sharing

descarguenlos y ponganlos en `data/rag/embeddings/`

### API Keys

Si vas a correr el benchmark con una API externa, configurá un archivo `.env` con la variable `OPENAI_API_KEY` (o la clave que corresponda a tu proveedor).

### Correr el benchmark

Para ejecutar el benchmark:

```sh
uv run main.py
```

o bien:

```sh
python main.py
```

### Analizar resultados

En `notebooks/rag_result_eval.ipynb` encontrarás un análisis general de los resultados, segmentado por tipo de pregunta. Recomendamos reportar las métricas siguiendo esa segmentación, ya que cada categoría presenta distintos niveles de dificultad.

Para correr este notebook con el mismo env, deben hacer primero 

```sh
uv pip install ipykernel
```
