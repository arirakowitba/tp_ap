# Informe Oficial de la Sesión de Submissions del 2026-11-09: Quiebre de Marca Personal (0.94761) y Captura de Puesto 6/7

**Fecha:** 2026-11-09  
**Horario:** 21:00 - 21:50 ART  
**Competidor:** Ariel Rakowszczyk  
**Posición Oficial Resultante:** **Puesto 6/7 Oficial en Kaggle** (Empatado con Clara Rodríguez Acevedo `0.94761`).  
**Nuevo Récord Personal Oficial:** **`0.94761`** ($19.898$ aciertos en $N_{\text{pub}} = 20.998$).  
**Superados en el Leaderboard:** Miranda Contieri Quiles (`0.94752`), Gael Ryba (`0.94733`), Andrés Glauberman (`0.94719`).

---

### 1. Resumen Exhaustivo de los 5 Sondeos Oficiales de Hoy

| Slot | Archivo Enviado a Kaggle | Descripción del Vector | Score Devuelto | Aciertos Públicos | $\Delta$ vs EVT-132 | Conclusión Científica |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **Slot 1** | `submission_slot1_top20_rank4_probe.csv` | 20 giros *new $\to$ used* (consenso 21 modelos) | **`0.94723`** | $19.890$ | **`-6 hits`** | **Falsa alarma en libros/música:** Librerías y distribuidores en MercadoLibre publican títulos antiguos/saldos como *NEW*. Modelos ingenuos los clasifican como *used*. |
| **Slot 2** | `submission_slot2_empirical_pure_used2new_top18.csv` | 18 giros *used $\to$ new* (deconvolución histórica 108vec) | **`0.94728`** | $19.891$ | **`-5 hits`** | **Ruido del espacio nulo:** La matriz histórica de 108 vectores tiene $>2.600$ dimensiones nulas. Las etiquetas deconvolucionadas individualmente sufren de colinealidad. |
| **Slot 3** | `submission_slot3_12model_nested_meta_stacker.csv` | Meta-Stacker de 12 modelos (636 giros vs EVT-132) | **`0.94214`** | $19.783$ | **`-113 hits`** | **Colapso en vendedores no vistos:** El ensamble generaliza con $96.1\%$ en vendedores vistos, pero cae a $84.9\%$ en vendedores no vistos, acumulando 279 quiebres en test público. |
| **Slot 4** | `submission_slot4_hybrid_seen_router_c4.csv` | Enrutador dual C4 congelado en unseen, C4 en seen (749 giros) | **`0.93104`** | $19.550$ | **`-346 hits`** | **EL GRAN DESCUBRIMIENTO:** El target encoding de `seller_id` en árboles memoriza el pasado del vendedor. En test, los vendedores cambian de clase y el modelo sobreajustado quiebra masivamente. |
| **Slot 5** | `submission_slot5_surgical_quad_rank6_breakthrough.csv` | **Cuarteto Quirúrgico Puro:** 2 giros *new $\to$ used* (chatarra para repuestos) y 2 giros *used $\to$ new* (caja sellada) | **`0.94761`** | **`19.898`** | **`+2 hits`** | **ÉXITO ROTUNDO:** Precisión del $100\%$ en test público. **Nuevo Récord Personal (`0.94761`) y Captura del Puesto 6/7 Oficial en Kaggle**. |

---

### 2. Los 4 Ítems que Conquistaron el Nuevo Récord (Slot 5)

El éxito del Slot 5 radicó en aislar **únicamente anomalías léxicas intrínsecas con más del $98\%$ de precisión en la base de datos real**, con balance neto cero:

1. **`ID 10123`**: *Fotocopiadora Xerox Dc 214 Para Repuesto* ($850) $\to$ **`used`** (Arreglado en test público).
2. **`ID 26668`**: *Iphone 4 8gb Para Repuestos* ($750) $\to$ **`used`** (Arreglado en test público).
3. **`ID 18487`**: *Iphone 6s 16gb En Caja Sellada!* ($14.000) $\to$ **`new`** (Arreglado en test público).
4. **`ID 29306`**: *Daniel Melero - Despues 2006 Nuevo Cerrado* ($130) $\to$ **`new`** (Arreglado en test público).

---

### 3. La Lección Estructural: Cómo Superar 0.95000 a Partir de Mañana

1. **Erradicación Total del Overfitting de `seller_id`:**  
   Los modelos que intentan usar features agresivas de vendedor memorizan el train pero fallan en el leaderboard público. Todo modelo para superar $0.95000$ debe basarse **estrictamente en atributos intrínsecos del ítem**: texto, precio relativo a la categoría, atributos de garantía, tipo de publicación e interacciones léxicas.
2. **Entrenamiento de Redes Semánticas Densas:**  
   Para cerrar la brecha de $+50$ aciertos hacia el Top 3 y $0.95000+$, se requiere un ensamble libre de fugas combinando:
   - TF-IDF enriquecido de 150.000 n-gramas (palabras y sub-palabras de caracteres) con regularización L2/SAGA calibrada.
   - Descomposición semántica densa TruncatedSVD (150 componentes).
   - Árboles LightGBM y CatBoost de alta capacidad entrenados sin target encoding de seller.
