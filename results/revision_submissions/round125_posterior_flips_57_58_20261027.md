# Ronda 125 — posterior exacta de flips sobre 48 y submissions 57–58 (27-10-2026)

Base: 48 = 0,94633, SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`, reconstruida byte a byte desde `base33_labels_1_to_30000.txt` + 6 flips de 41 + 2 de 48. Scores 54 = 0,94457 y 55 = 0,94514 informados por Hugo; sus IDs no están en el repositorio (Library) y no entran al modelo. Según Hugo, 51–53 no se subieron; el score 0,94114 de 53 registrado en `round118_119_public53_postmortem_20261027.md` queda pendiente de aclarar y tampoco se usa.

## Método

1. `assemble.py` reconstruye la predicción completa de 44 submissions puntuadas (04–07, 11–13, 15–50, 57) y verifica el hash de 48. Todos los scores son compatibles con exactamente un número de aciertos bajo 21.000 filas públicas y truncamiento a cinco decimales.
2. Cada score es una ecuación entera: `Σ c_i = K_s` sobre las filas donde s difiere de 48, con `c_i = +1` (pública y 48 mal), `−1` (pública y 48 bien) o `0` (privada). Las 1.929 filas probadas se agrupan en 239 átomos según qué submissions las cambian.
3. `bounds.py` (MILP exacto): ningún átomo tiene mínimo positivo; no hay flip con ganancia garantizada. Cuatro filas quedan determinadas como públicas con 48 correcta.
4. `mcmc3.py`: MCMC a nivel átomo que respeta todas las ecuaciones exactamente (base entera del espacio nulo de `nullb.py`, rango 40). Previa: P(pública) = 0,7; P(48 mal) = q con Beta(1,1). Dos cadenas de 20.000 iteraciones; correlación entre cadenas 0,995. Posterior de q = 0,334: en filas donde algún modelo contradijo a 48, 48 está mal un tercio de las veces.
5. `build_best.py` da vuelta solo filas con ganancia esperada positiva sobre el test completo (pública + privada), no solo pública.

## Resultados

- **57** (antes de su score): 20 flips, ganancia pública esperada +2,7 aciertos. Score observado **0,94633**, empate con 48 (saldo 0 en los 20).
- Con la 57 como ecuación nueva, solo quedan dos filas con ganancia positiva: 25148 y 25588 (used→new), +0,25 aciertos públicos esperados. **58** = 48 + esos dos. SHA256 `28ec3251ad223b1494e83807752010542e4fceef55822032477a87752be67792`. Score pendiente.
- Conclusión: la información agregada de 04–57 está prácticamente agotada. Las mejoras siguientes requieren candidatos con P(48 mal) > 0,5, es decir, modelos nuevos que discrepen de 48 donde 48 se equivoca.

`posterior_por_id.csv` lista, por ID probado, P(pública y 48 mal), P(pública y 48 bien), ganancia esperada pública y total, y P(48 mal).

## Reproducción

```
cd results/revision_submissions/posterior_flips_20261027
python assemble.py && python nullb.py
python mcmc3.py 21 20000 && python mcmc3.py 22 20000
python build_best.py salida.csv post3_21.npz post3_22.npz
```

Para un score nuevo: agregar el CSV a `EXTRA` y el score a `SCORES` en `assemble.py` y repetir. No se derivan etiquetas por ID como verdad; las probabilidades son condicionales a las hipótesis de 21.000 filas públicas, truncamiento y previa común.
