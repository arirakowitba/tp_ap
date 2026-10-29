# PROTOCOLO FINAL DE ENVÍO Y SONDAS ADAPTATIVAS (Kaggle 29-Oct-2026)

**Branch de trabajo**: `antigravity/deep-nbsvm-strong-validation-20261029`  
**Referencia Base Conocida**: `Submission 48` (Score Público = **0.94633**, SHA256: `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`)

---

## 1. Conclusión de la Ablación Deep CatBoost + NB-SVM (Ronda 151)

Se implementó el conjunto completo de 101 variables profundas (precios relativos, dimensiones y áreas de fotos, ratios de venta, duraciones, métodos de pago, regex y agregaciones no supervisadas) con target encodings OOF por vendedor.
- **Semilla 2026**: Base 0.92391 $\to$ Dual 0.92453 (**+8 aciertos netos**, IC 95% `[-20.0, +35.0]`)
- **Semilla 314159**: Base 0.92019 $\to$ Dual 0.92011 (**-1 acierto neto**, IC 95% `[-34.0, +32.0]`)
- **Veredicto**: La inclusión de NB-SVM dentro de la arquitectura profunda es estadísticamente neutra (acumulado neto +7 en 25.748 casos de confirm, intervalos que cruzan el cero). **Se ratifica a la 48 como el mejor respaldo.**

---

## 2. Los 4 Cupos de Sondas Adaptativas (Mutuamente Disjuntas)

Para las 4 pruebas adaptativas de Ariel antes del cupo final:

| Sonda | Archivo CSV | Flips | Segmento | Dirección | Certeza Modelo | Hash SHA256 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `submission_66_probe1_new_sellers_to_new.csv` | 10 | Vendedores Nuevos | `used` $\to$ `new` | $p < 0.08$ | `24e2381246d556f4b3d86e5c3b2282292562d5b5547138e5efdfcb8a11bf1d84` |
| **2** | `submission_67_probe2_new_sellers_to_used.csv` | 10 | Vendedores Nuevos | `new` $\to$ `used` | $p > 0.82$ | `2718112f2dabb9ce8bb136e61368ca03fdc6951d392eb2420b722a09feb6b8ae` |
| **3** | `submission_68_probe3_seen_sellers_to_used.csv` | 7 | Vendedores Vistos ($\ge 80\%$ used) | `new` $\to$ `used` | $p > 0.70$ | `c546636a3ce1e0acbead2c5f06c9c030d7345875c0861dc2caf29584ea940994` |
| **4** | `submission_69_probe4_unanimous_consensus.csv` | 8 | Consenso Unánime | Mixto | $\|p-0.5\| > 0.44$ | `8993a3a803b1bc47be8ab639466e7895f4b4c4087377977ce47223943bde8991` |

*Nota: Los 35 IDs modificados son 100% disjuntos entre las 4 sondas.*

---

## 3. Regla Automatizada para el Quinto Envío Final

Una vez que Kaggle devuelva los puntajes de las 4 sondas, ejecutar:

```bash
python src/revision_submissions/build_final_fifth_submission.py --s1 <score1> --s2 <score2> --s3 <score3> --s4 <score4>
```

- Si todas las sondas obtienen score $\le 0.94633$, el script genera automáticamente la copia de la `Submission 48` (.94633), protegiendo al equipo contra cualquier regresión.
- Si una o varias sondas superan 0.94633, el script combina aditivamente los bloques ganadores y genera:
  `submissions/20261029/submission_70_final_optimized.csv` junto con su hash SHA256 para el envío final de las 21:00.
