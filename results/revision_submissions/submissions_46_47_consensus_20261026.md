# Tanda 46–47: consenso fuerte de tres modelos entre vendedores nuevos

Base exacta: submission 41, score público 0.94628. Fuente: auditoría round49, checkpoint Library libfile_4a128759e8888191af86e1ede299b385. Regla post hoc: vendedor ausente de train, HistGB, FastText y CatBoost de ronda30 asignan al menos 0.70 a la clase contraria a 41; excluir 41, 43 y 45. Los siete también formaban parte de la 39 masiva, cuyo resultado negativo no identifica el efecto de este subconjunto.

46 cambia siete IDs: 8532, 13583, 16436, 22635, 25426, 25656, 26861. Dirección used→new: 8532, 13583, 22635, 25426; new→used: 16436, 25656, 26861. SHA256 a619ba9ec77e1ebf5341ed97a7252eea7b951a8bcd2012a1e47aacc1766a8d64. Library libfile_eb0ba5ac94248191b9fc538da094d6d2.

47 contiene solo los cuatro used→new de 46, SHA256 e6ae2508ce1f56dc093478987ae297667685593fa55ff76a3ecbae990dfad343. Library libfile_dc085aae4d088191b229dbe3c09be9e1. Ambos CSV tienen 30.000 ID únicos y cambian exactamente siete/cuatro condiciones respecto de 41. Subir 46 primero; considerar 47 después del score para aislar el efecto agregado por dirección. No subir ambos sin revisar el primero.

Limitación: la exactitud general de consenso en train (~95% a umbral .70) no mide fixes/breaks donde 41 discrepa. La regla se seleccionó tras observar el éxito del bloque 41 y fracaso de 45; no hay validación directa del efecto incremental sobre 41. Es una microprueba exploratoria, no una mejora demostrada ni un camino suficiente para cerrar la brecha al líder. No se infieren etiquetas individuales de scores públicos y el asistente no envió submissions.

Feedback informado por Ariel: 46 = 0.94623 frente a 41 = 0.94628, delta público redondeado −0.00005 para los siete cambios. La 47 sigue pendiente; subirla permite medir el bloque used→new de cuatro y obtener por resta el efecto agregado de los tres new→used, sin atribuir etiquetas por ID. Conservar 41 como mejor mientras tanto.

Feedback informado por Ariel: 47 = 0.94628. Frente a 41, los cuatro cambios used→new tienen delta redondeado 0.00000; 46 frente a 47 = −0.00005 para los tres new→used. La 41 y la 47 empatan como mejor score público observado. Por parsimonia conservar la 41 como candidata principal; el empate no demuestra aciertos individuales y el redondeo limita recuentos exactos.
