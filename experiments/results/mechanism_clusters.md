# OpenCure cross-disease mechanism clusters

Drugs that appear as high-score candidates across multiple unrelated 
diseases. Ranked by cluster_strength = N × mean_score × category_diversity × pathway_coherence.

| Drug | N diseases | Diversity | Mean score | Strength | Categories |
|---|---|---|---|---|---|
| Dexamethasone | 33 | 0.788 | 0.543 | 16.027 | autoimmune, cardio, gynecological, hepatic, infectious, metabolic, neurodegen, oncology, other, psychiatric, renal, respiratory |
| Testosterone | 17 | 0.765 | 0.522 | 7.829 | autoimmune, cardio, gynecological, hepatic, metabolic, oncology, other, respiratory |
| Cimetidine | 12 | 0.75 | 0.601 | 6.312 | autoimmune, cardio, hepatic, metabolic, neurodegen, oncology, psychiatric, renal |
| Tacrolimus | 14 | 0.714 | 0.547 | 5.86 | autoimmune, cardio, infectious, metabolic, neurodegen, oncology, psychiatric |
| Urea | 10 | 0.7 | 0.567 | 4.822 | autoimmune, cardio, hepatic, metabolic, neurodegen, psychiatric |
| Tretinoin | 12 | 0.583 | 0.502 | 4.769 | autoimmune, oncology, other |
| Paclitaxel | 12 | 0.5 | 0.502 | 4.52 | autoimmune, neurodegen, oncology, other |
| Estradiol | 11 | 0.636 | 0.497 | 4.474 | gynecological, hepatic, neurodegen, oncology, other, psychiatric, respiratory |
| Metformin | 8 | 0.75 | 0.529 | 3.702 | autoimmune, cardio, neurodegen, oncology, other, psychiatric |
| Fluorouracil | 10 | 0.4 | 0.517 | 3.62 | gynecological, hepatic, neurodegen, oncology, respiratory |
| histamine dihydrochloride | 8 | 0.75 | 0.509 | 3.562 | autoimmune, cardio, neurodegen, oncology, other, respiratory |
| Methylprednisolone | 8 | 0.625 | 0.514 | 3.339 | autoimmune, infectious, oncology, other, respiratory |
| Hydrocortisone | 8 | 0.625 | 0.514 | 3.338 | autoimmune, cardio, gynecological, other, respiratory |
| Prednisolone | 8 | 0.5 | 0.533 | 3.196 | autoimmune, cardio, metabolic, oncology, renal |
| Tamoxifen | 8 | 0.5 | 0.523 | 3.14 | gynecological, oncology, other |
| Cyclic Adenosine Monophosphate | 7 | 0.714 | 0.506 | 3.037 | autoimmune, oncology, other, psychiatric, respiratory |
| Acetylcysteine | 7 | 0.714 | 0.503 | 3.019 | autoimmune, neurodegen, neurological, oncology, other, respiratory |
| Carboplatin | 8 | 0.375 | 0.546 | 3.006 | oncology, other |
| Folic Acid | 7 | 0.571 | 0.541 | 2.975 | autoimmune, neurodegen, oncology |
| Epigallocatechin gallate | 8 | 0.5 | 0.495 | 2.968 | cardio, infectious, oncology, other, respiratory |

## Details

### Dexamethasone  (strength 16.027)
Hits (33): Amyotrophic lateral sclerosis, Asthma, Atherosclerosis, Atrial fibrillation, Chronic kidney disease, Coronary artery disease, Crohns disease, Cystic fibrosis, Endometriosis, Ewing sarcoma, Hypertension, Idiopathic pulmonary fibrosis, Inflammatory bowel disease, Leukemia, Liver cirrhosis, Lung cancer, Lupus, Lymphoma, Medulloblastoma, Melanoma, Neuroblastoma, Obesity, Osteoporosis, Osteosarcoma, Psoriasis, Pulmonary hypertension, Retinoblastoma, Rhabdomyosarcoma, Rheumatoid arthritis, Schizophrenia, Sepsis, Type 2 diabetes, Ulcerative colitis

Representative mechanism path:
> Dexamethasone —[downregulates]→ gene-10417 —[is downregulated by]→ Amyotrophic lateral sclerosis

### Testosterone  (strength 7.829)
Hits (17): Acromegaly, Atherosclerosis, COPD, Colorectal cancer, Cystic fibrosis, Endometriosis, Idiopathic pulmonary fibrosis, Inflammatory bowel disease, Liver cirrhosis, Multiple myeloma, Obesity, Ovarian cancer, Pancreatic cancer, Psoriasis, Pulmonary hypertension, Type 2 diabetes, Ulcerative colitis

Representative mechanism path:
> Testosterone —[is a therapeutic for]→ MESH:D042882 —[is treated by]→ DB00104 —[treats]→ Acromegaly

### Cimetidine  (strength 6.312)
Hits (12): Asthma, Bipolar disorder, Chronic kidney disease, Coronary artery disease, Crohns disease, Huntingtons disease, Hypertension, Liver cirrhosis, Obesity, Osteoporosis, Prostate cancer, Psoriasis

Representative mechanism path:
> Cimetidine —[binds]→ gene-8647 —[is bound by]→ DB01234 —[treats]→ Asthma

### Tacrolimus  (strength 5.86)
Hits (14): Asthma, Bipolar disorder, Colorectal cancer, Coronary artery disease, Crohns disease, Heart failure, Lymphoma, Melanoma, Multiple sclerosis, Osteoporosis, Psoriasis, Pulmonary hypertension, Sepsis, Ulcerative colitis

Representative mechanism path:
> Tacrolimus —[downregulates]→ gene-991 —[is downregulated by]→ DB00938 —[treats]→ Asthma

### Urea  (strength 4.822)
Hits (10): Asthma, Depression, Hypertension, Liver cirrhosis, Osteoporosis, Parkinsons disease, Psoriasis, Pulmonary hypertension, Rheumatoid arthritis, Type 2 diabetes

Representative mechanism path:
> Urea —[is a therapeutic for]→ MESH:D012871 —[is treated by]→ DB00741 —[treats]→ Asthma

### Tretinoin  (strength 4.769)
Hits (12): Breast cancer, Ewing sarcoma, Lung cancer, Lupus, Lymphoma, Melanoma, Neuroblastoma, Prostate cancer, Retinoblastoma, Rhabdomyosarcoma, Ulcerative colitis, Wilms tumor

Representative mechanism path:
> Tretinoin —[downregulates]→ gene-27338 —[is downregulated by]→ DB00563 —[treats]→ Breast cancer

### Paclitaxel  (strength 4.52)
Hits (12): Alzheimers disease, Ewing sarcoma, Glioblastoma, Inflammatory bowel disease, Lung cancer, Lymphoma, Medulloblastoma, Parkinsons disease, Retinoblastoma, Rhabdomyosarcoma, Spinocerebellar ataxia, Wilms tumor

Representative mechanism path:
> Paclitaxel —[downregulates]→ gene-3692 —[is a target of]→ DB09130 —[is a therapeutic for]→ Alzheimer's disease

### Estradiol  (strength 4.474)
Hits (11): Amyotrophic lateral sclerosis, COPD, Colorectal cancer, Depression, Endometriosis, Idiopathic pulmonary fibrosis, Liver cirrhosis, Neuroblastoma, Osteosarcoma, Retinoblastoma, Wilms tumor

Representative mechanism path:
> Estradiol —[downregulates]→ gene-22920 —[is linked to]→ Amyotrophic lateral sclerosis

### Metformin  (strength 3.702)
Hits (8): Alzheimers disease, Depression, Heart failure, Medulloblastoma, Neuroblastoma, Ovarian cancer, Pulmonary hypertension, Rheumatoid arthritis

Representative mechanism path:
> Metformin —[is a therapeutic for]→ Alzheimer's disease

### Fluorouracil  (strength 3.62)
Hits (10): Endometriosis, Glioblastoma, Idiopathic pulmonary fibrosis, Leukemia, Liver cirrhosis, Lung cancer, Lymphoma, Multiple myeloma, Multiple sclerosis, Pancreatic cancer

Representative mechanism path:
> Fluorouracil —[binds]→ gene-1558 —[is bound by]→ DB00603 —[treats]→ Endometriosis

### histamine dihydrochloride  (strength 3.562)
Hits (8): Atherosclerosis, Idiopathic pulmonary fibrosis, Inflammatory bowel disease, Lung cancer, Lupus, Multiple sclerosis, Neuroblastoma, Phenylketonuria

Representative mechanism path:
> histamine dihydrochloride —[is a therapeutic for]→ MESH:D051437 —[is treated by]→ DB00627 —[treats]→ Atherosclerosis

### Methylprednisolone  (strength 3.339)
Hits (8): COPD, Crohns disease, Lymphoma, Melanoma, Pancreatic cancer, Psoriasis, Rhabdomyosarcoma, Sepsis

Representative mechanism path:
> Methylprednisolone —[palliates]→ DOID:2531 —[is treated by]→ DB00860 —[treats]→ COPD

### Hydrocortisone  (strength 3.338)
Hits (8): Crohns disease, Cystic fibrosis, Endometriosis, Heart failure, Idiopathic pulmonary fibrosis, Lupus, Phenylketonuria, Psoriasis

Representative mechanism path:
> Hydrocortisone —[treats]→ Crohn's disease

### Prednisolone  (strength 3.196)
Hits (8): Asthma, Atherosclerosis, Chronic kidney disease, Inflammatory bowel disease, Lymphoma, Obesity, Psoriasis, Ulcerative colitis

Representative mechanism path:
> Prednisolone —[treats]→ Asthma

### Tamoxifen  (strength 3.14)
Hits (8): Breast cancer, Endometriosis, Ewing sarcoma, Multiple myeloma, Neuroblastoma, Osteosarcoma, Ovarian cancer, Prostate cancer

Representative mechanism path:
> Tamoxifen —[treats]→ Breast cancer

### Cyclic Adenosine Monophosphate  (strength 3.037)
Hits (7): Idiopathic pulmonary fibrosis, Leukemia, Pancreatic cancer, Rhabdomyosarcoma, Schizophrenia, Ulcerative colitis, Wilms tumor

Representative mechanism path:
> Cyclic Adenosine Monophosphate —[is a therapeutic for]→ MESH:D013224 —[is treated by]→ DB00860 —[treats]→ Idiopathic pulmonary fibrosis

### Acetylcysteine  (strength 3.019)
Hits (7): Amyotrophic lateral sclerosis, COPD, Epilepsy, Multiple myeloma, Parkinsons disease, Spinocerebellar ataxia, Ulcerative colitis

Representative mechanism path:
> Acetylcysteine —[is a therapeutic for]→ Amyotrophic lateral sclerosis

### Carboplatin  (strength 3.006)
Hits (8): Breast cancer, Medulloblastoma, Melanoma, Multiple myeloma, Pancreatic cancer, Prostate cancer, Retinoblastoma, Wilsons disease

Representative mechanism path:
> Carboplatin —[is a therapeutic for]→ Breast cancer

### Folic Acid  (strength 2.975)
Hits (7): Alzheimers disease, Breast cancer, Crohns disease, Inflammatory bowel disease, Lung cancer, Multiple sclerosis, Ovarian cancer

Representative mechanism path:
> Folic Acid —[is a therapeutic for]→ Alzheimer's disease

### Epigallocatechin gallate  (strength 2.968)
Hits (8): Atherosclerosis, COVID-19, Idiopathic pulmonary fibrosis, Medulloblastoma, Multiple myeloma, Osteosarcoma, Rhabdomyosarcoma, Spinocerebellar ataxia

Representative mechanism path:
> Epigallocatechin gallate —[is a therapeutic for]→ MESH:D015673 —[is treated by]→ DB01065 —[is a therapeutic for]→ Atherosclerosis