# LongMemEval - Resumen del Paper

# Preguntas que tengo del paper

- Cual es la diferencia entre LongMemEval_S y LongMemEval_M
- Cual es la diferencia entre los tres datasets [acá](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main)

# Resumen

While the length of the history is freely extensible, we provide two standard settings
for consistent comparison: 

- LONGMEMEVALS with approximately 115k tokens per problem
- LONGMEMEVALM with 500 sessions (around 1.5 million tokens).

## 3. LongMemEval

### 3.1 Formulation

Una instancia de su benchmark es una 

Una 4-upla $(S, q, t_q, a)$

Una secuencia de sesiones $S ≡
[(t_1, S_1),(t_2, S_2), ...,(t_N , S_N )]$ ordenada cronologicamente

- Acá, $S_i$ es una interaccion multi-turno entre un usuario y un asistente
- $t_i$ es el timestamp de la sesion

Ademas, plantean que cada sesion puede descomponerse en *rounds*, donde un *round* es un mensaje de usuario seguido por uno de asistente.

En la evaluación:

- Se provee $S$ al sistema, sesion por sesion
- Se provee $q$ y $t_q > t_N$, que representan la pregunta y su respectiva fecha.
- $a$ es una frase corta que indica la respuesta, pues en algunos casos la pregunta es *open-ended*

### 3.2 Benchmark Curation

El benchmark apunta a evaluar cinco cuestiones:

- **Information Extraction (IE):** Ability to recall specific information from extensive
interactive histories, including the details mentioned by either the user or the assistant.
- **Multi-Session Reasoning (MR):** Ability to synthesize the information across multiple
history sessions to answer complex questions that involve aggregation and comparison
- **Knowledge Updates (KU)**: Ability to recognize the changes in the user’s personal
information and update the knowledge of the user dynamically over time.
- **Temporal Reasoning (TR):** Awareness of the temporal aspects of user information,
including both explicit time mentions and timestamp metadata in the interactions.
- **Abstention (ABS):** Ability to identify questions seeking unknown information, i.e.,
information not mentioned by the user in the interaction history, and answer “I don’t know”.

Para hacer esto, el benchmark propone siete tipos de preguntas:

- **Single-session-user**
- **single-session-assistant**
- **Single-session-preference**
- **multi-session (MR)**
- **knowledge-update (KU)**
- **Temporal-reasoning (TR)**
- **Abstention:** No aparece como un tipo de pregunta en el dataset, pero son 30 preguntas modificadas para evaluar si el modelo puede detectar que no tiene información suficiente y abstenerse.

![image.png](attachment:5e9522dd-8ce0-44df-8331-6e069c41d39b:image.png)

Definen una ontologia con 164 atributos en cinco categorias:

- lifestyle
- belongings
- life events
- situations context
- demographic information

Para cada categoria, usan un LLM para generar attribute-focused use background paragraphs, que incluyen una discusion detallada de una experiencia del usuario. A eso le llaman *background sampling* 

![image.png](attachment:25d70b71-1363-40c8-b68f-c785bffca402:image.png)

Despues de eso, agarran un parrafo y le piden a un LLM que proponga QA pairs (pregunta y respuesta). Supuestamente, estas preguntas a veces carecen de la profundidad y diversidad necesarias, entonces un humano las filtra y reescribe. 

![image.png](attachment:8ade3985-16e8-4c5e-9086-36d4f2d603ca:image.png)

Una vez hecho esto, tienen preguntas construidas a partir de mensajes de usuarios. Lo que se hace a continuacion es generar las demás sesiones.

[PENDIENTE: Evidence Session Construction y History Compilation]

### 3.3 Metricas de Evaluación

**Question Answering:** Como las respuestas son abiertas, no sería correcto usar *exact matching* para evaluar las respuestas. Por eso proponen usar un LLM para evaluar las respuestas, lo que comunmente se denomina *LLM as a judge*. Especifican que este metodo coincide un 97% de las veces con el juicio de un humano.

**Memory Recall:** Como el benchmark tiene, para cada pregunta, una *label* de la ubicacion de la respuesta, se puede medir si el sistema encontro correctamente la justificacion a la pregunta. Para eso, proponen usar 

- Recall@k [PENDIENTE: Explicar]
- NDCG@k [PENDIENTE: Explicar]

### 3.4 LongMemEval es desafiante para sistemas comerciales

## 4. Formulacion del problema Long-Term Memory System

## 5. Experimentos realizados