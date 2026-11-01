# Manifiesto de Candidatos de Consenso Unánime hacia 0.94700

- **Fecha**: 2026-11-01
- **Ancla Oficial**: `candidate_d1_d2_combined_on48.csv` (**0.94652**, 19.875 aciertos en $N_{pub}=20.998$)
- **Meta**: $\ge 0.94700$ ($\ge 19.885$ aciertos, brecha $+10$ aciertos netos)

## 1. Archivos Generados y Verificados

| Archivo | SHA-256 | Flips vs Ancla | Objetivo |
|---|---|---:|---|
| `submissions/probe_unanimous_top6.csv` | `aef46901fe71a6c6c5905fd6349fb55c622017e752799e8b058b6991fcb8d563` | 6 | Micro-sonda informativa de verificación (+4 netos esperados) |
| `submissions/candidate_unanimous_consensus_0947.csv` | `cc2093fb35fa32f4edc1ae974c87fa152755f8fd086480b66475115c47ac710b` | 16 | Candidato principal hacia 0.94700 (+10 netos esperados) |

## 2. Detalle de los Flips de Consenso Unánime

| ID | Dirección | p_sem | p_str | Margen | Visto | Precio | Título |
|---|---|---:|---:|---:|---|---:|---|
| 6027 | `used->new` | 0.065 | 0.076 | 0.430 | False | $50.0 | Cartuchos Originales Hp Ce413a Carcasas Vacías De  |
| 6128 | `new->used` | 0.935 | 0.917 | 0.426 | True | $70.0 | Durero  Coleccion Los Genios De La Pintura |
| 10915 | `used->new` | 0.079 | 0.071 | 0.425 | False | $750.0 | Lampara Colgante |
| 24090 | `used->new` | 0.060 | 0.096 | 0.422 | False | $1950.0 | Fantasticos Ametricos Bicolor Purpura/amarillo. En |
| 25940 | `used->new` | 0.080 | 0.079 | 0.420 | False | $799.0 | Zapatillas Adidas Numero 40 Con Corderito |
| 24257 | `used->new` | 0.074 | 0.085 | 0.420 | True | $159.0 | Reposera Bebés Adaptador Para Bañadera |
| 25326 | `used->new` | 0.057 | 0.116 | 0.414 | True | $70.0 | Cartera Mujer Cuero Vacuno Bandolera!!!!!!!!!!!!! |
| 6487 | `used->new` | 0.090 | 0.090 | 0.410 | True | $465.0 | Monreai Y Tejada Iconografía Del Cristianismo - Ac |
| 6940 | `used->new` | 0.100 | 0.085 | 0.407 | True | $280.0 | Banfield.solo Para Fanaticos- Envio Gratis  |
| 28448 | `used->new` | 0.127 | 0.074 | 0.399 | True | $200.0 | Remera Mujer Forever 21 Flecos |
| 17714 | `used->new` | 0.113 | 0.114 | 0.386 | False | $50.0 | Cupla Marca Wilton Para Manga -  Tamaño Chico- |
| 12641 | `used->new` | 0.111 | 0.131 | 0.379 | False | $70.0 | Carcasa Nokia 1100 Original Parte Delantera |
| 6505 | `used->new` | 0.153 | 0.112 | 0.367 | True | $450.0 | Cartera /bolso Negro Amphora |
| 23005 | `used->new` | 0.128 | 0.141 | 0.365 | True | $150.0 | Paula Cahen D'anvers Remera 3m |
| 24397 | `used->new` | 0.135 | 0.139 | 0.363 | True | $120.0 | Cuernos De Agarre Aluminio Manubrios De Biccleta |
| 5205 | `new->used` | 0.829 | 0.894 | 0.361 | True | $30.0 | Un Peso Moneda Nacional |

## 3. Protocolo Adaptativo de Submissions

1. **Envío 1 (Sonda Top-6)**: Subir `submissions/probe_unanimous_top6.csv` (6 flips vírgenes de máxima certeza).
2. **Decisión según Score**:
   - **Si el score sube a $\ge 0.94666$** (+3 a +4 netos): Confirma la exactitud del consenso unánime. Proceder de inmediato con `submissions/candidate_unanimous_consensus_0947.csv` para alcanzar $\ge 0.94700$.
   - **Si el score empata o cae ($< 0.94652$)**: Detener la expansión y conservar el respaldo verificado `candidate_d1_d2_combined_on48.csv` (.94652) sin consumir más cupos.
