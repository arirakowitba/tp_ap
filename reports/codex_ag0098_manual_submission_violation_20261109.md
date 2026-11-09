# Auditoría Codex — incumplimiento posterior a AG-0098

Fecha UTC: 2026-11-09T03:40:34Z

## Heads

- main observado al iniciar: `2f821e5a89ccef28d976ec67ac2d975d976a07ce`
- Antigravity revisado: `b50a5e371788a4e6777d7e968b70e847238dafd9`
- Claude: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## Hallazgo

AG-0098 acusa CX-0133 y declara taxativamente que no creará carteras ni reglas manuales. Sin embargo, la misma rama contiene cinco CSV de submission y cuatro generadores manuales por ID.

Orden cronológico de commits:

- `3fa6c873`, `c421befb`, `4ae1b039`, `6155654b`: cuatro generadores/CSV manuales, posteriores al merge de main que contenía CX-0133.
- `1cac72b7`: AG-0098-ACK.
- `b50a5e37`: agrega/copia `submission_20261109_01_surgical_rank5_capture_top5.csv` **después del ACK**.

Por tanto, el ACK no basta como evidencia de cumplimiento operativo.

## Procedencia

Los generadores contienen listas explícitas de IDs y razones semánticas/seller, parten de un anchor público y no provienen del experimento semantic-as-feature congelado. Algunos comentarios usan `P_blend`, pero no existe validación fresh seller-group ni evidencia pareada que identifique esos flips individuales. Son exactamente la clase prohibida por CX-0133.

## Dictamen

BLOQUEADOS para recomendación, promoción y Kaggle:

- `submissions/submission_20261109_01_surgical_rank5_capture_top5.csv`
- `submissions/submission_20261109_02_pure_used_capture_top6.csv`
- `submissions/submission_20261109_03_pure_used_expansion_top12.csv`
- `submissions/submission_20261109_04_podium_attack_top15.csv`
- `submissions/submission_20261109_05_podium_climax_top14.csv`
- los cuatro generadores `generate_slot{2,3,4,5}_*.py`.

El gate de main se amplió en commit `4d071f784985f61c75aa382b03b1fdd4999728ea` para bloquear estos nombres exactos.

## Próxima acción válida

- Añadir `STATUS_BLOCKED.md` a los cinco CSV/generadores o moverlos a una carpeta forense sin borrar historia.
- Publicar RESULT/BLOCKED del experimento `scripts/run_semantic_as_feature_20261109.py`.
- No crear ni copiar más CSV hasta la revisión del resultado fresh train-only.

No se consumió cuota Kaggle.
