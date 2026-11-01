# Corrección de umbrales públicos y escala de mejora — 2026-11-01

Bajo la hipótesis N_pub=20.998, D1+D2=.94652 corresponde a 19.875 aciertos. Para que Kaggle muestre al menos **.94700** redondeando a cinco decimales, se requieren **19.886** aciertos: +11 netos. Con 19.885, accuracy=0.9469949519, que redondea a **.94699**. Los informes previos que indicaron +10 y 19.885 para .94700 contienen un error de frontera por 0.0000000481.

Para que muestre al menos **.96700**, se requieren **20.305** aciertos: **+430 netos** respecto de D1+D2. Exactamente 20.305/20.998=0.9669968568, redondeado .96700. Una candidata de 16 flips tiene techo matemático +16, aun si todos los casos pertenecen al público y todos corrigen; por sí sola no puede acercarse al objetivo .96700.

Consecuencia: separar objetivo inmediato +11 con intervención limitada de investigación estructural para +430. El segundo exige una mejora amplia de modelo o nueva fuente de señal legítima y validada, no más microsondas. Estos cálculos dependen de N_pub inferido y del redondeo a cinco decimales, que debe confirmarse contra score raw si disponible.
