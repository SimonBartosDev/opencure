# OpenCure cross-disease mechanism clusters

Drugs that appear as high-score candidates across multiple unrelated 
diseases. Ranked by cluster_strength = N × mean_score × category_diversity × pathway_coherence.

| Drug | N diseases | Diversity | Mean score | Strength | Categories |
|---|---|---|---|---|---|
| Dexamethasone | 34 | 0.794 | 0.517 | 15.759 | autoimmune, cardio, gynecological, hepatic, infectious, metabolic, neurodegen, oncology, psychiatric, rare_genetic, renal, respiratory |
| Testosterone | 20 | 0.8 | 0.509 | 9.165 | autoimmune, cardio, gynecological, hepatic, infectious, metabolic, oncology, rare_genetic, respiratory |
| Cimetidine | 18 | 0.722 | 0.568 | 8.313 | autoimmune, cardio, hepatic, infectious, metabolic, neurodegen, psychiatric, rare_genetic, renal |
| Tacrolimus | 20 | 0.75 | 0.532 | 7.683 | autoimmune, cardio, infectious, metabolic, neurodegen, oncology, psychiatric, rare_genetic |
| Urea | 11 | 0.727 | 0.545 | 5.181 | autoimmune, cardio, hepatic, infectious, metabolic, neurodegen, psychiatric |
| Hydrocortisone | 11 | 0.727 | 0.479 | 4.551 | autoimmune, cardio, gynecological, infectious, rare_genetic, respiratory |
| Fluorouracil | 11 | 0.455 | 0.519 | 4.15 | autoimmune, gynecological, hepatic, neurodegen, oncology, respiratory |
| Estradiol | 9 | 0.667 | 0.512 | 3.842 | gynecological, hepatic, neurodegen, oncology, psychiatric, respiratory |
| Methylprednisolone | 9 | 0.556 | 0.522 | 3.654 | autoimmune, infectious, oncology, rare_genetic, respiratory |
| Folic Acid | 9 | 0.667 | 0.505 | 3.574 | autoimmune, infectious, neurodegen, oncology |
| Prednisolone | 8 | 0.625 | 0.529 | 3.437 | autoimmune, cardio, infectious, metabolic, oncology, renal |
| Acetylcysteine | 8 | 0.75 | 0.484 | 3.388 | autoimmune, neurodegen, neurological, oncology, rare_genetic, respiratory |
| Progesterone | 8 | 0.75 | 0.48 | 3.357 | gynecological, hepatic, infectious, metabolic, rare_genetic, respiratory |
| Paclitaxel | 8 | 0.625 | 0.505 | 3.283 | autoimmune, neurodegen, oncology, rare_genetic |
| Rebamipide | 9 | 0.667 | 0.496 | 3.099 | autoimmune, infectious, neurodegen, oncology, rare_genetic |
| Melatonin | 7 | 0.714 | 0.501 | 3.007 | neurodegen, neurological, oncology, psychiatric, respiratory |
| histamine dihydrochloride | 6 | 0.833 | 0.535 | 2.943 | autoimmune, cardio, neurodegen, oncology, rare_genetic, respiratory |
| Budesonide | 7 | 0.429 | 0.501 | 2.506 | autoimmune, infectious, neurodegen, oncology |
| Ouabain | 6 | 0.667 | 0.496 | 2.479 | autoimmune, infectious, oncology, respiratory |
| Tretinoin | 7 | 0.286 | 0.517 | 2.329 | autoimmune, infectious, oncology |

## Details

### Dexamethasone  (strength 15.759)
Hits (34): Amyotrophic lateral sclerosis, Asthma, Atherosclerosis, Atrial fibrillation, Chronic kidney disease, Coronary artery disease, Crohns disease, Cystic fibrosis, Duchenne muscular dystrophy, Ehlers-Danlos syndrome, Endometriosis, Gaucher disease, HIV, Hepatitis C, Hypertension, Idiopathic pulmonary fibrosis, Inflammatory bowel disease, Leukemia, Liver cirrhosis, Lung cancer, Lupus, Lymphoma, Malaria, Obesity, Osteoporosis, Psoriasis, Pulmonary hypertension, Rheumatoid arthritis, Schistosomiasis, Schizophrenia, Sepsis, Tuberculosis, Type 2 diabetes, Ulcerative colitis

Representative mechanism path:
> Dexamethasone —[downregulates]→ gene-10417 —[is downregulated by]→ Amyotrophic lateral sclerosis

### Testosterone  (strength 9.165)
Hits (20): Atherosclerosis, Atrial fibrillation, COPD, Colorectal cancer, Cystic fibrosis, Ehlers-Danlos syndrome, Endometriosis, Idiopathic pulmonary fibrosis, Inflammatory bowel disease, Liver cirrhosis, Malaria, Multiple myeloma, Neurofibromatosis, Obesity, Ovarian cancer, Pancreatic cancer, Psoriasis, Pulmonary hypertension, Type 2 diabetes, Ulcerative colitis

Representative mechanism path:
> Testosterone —[downregulates]→ gene-5111 —[is a target of]→ DB00945 —[is a therapeutic for]→ Atherosclerosis

### Cimetidine  (strength 8.313)
Hits (18): Asthma, Bipolar disorder, Chagas disease, Chronic kidney disease, Coronary artery disease, Crohns disease, Duchenne muscular dystrophy, Huntingtons disease, Hypertension, Leishmaniasis, Liver cirrhosis, Malaria, Obesity, Osteoporosis, Psoriasis, Schistosomiasis, Tuberculosis, Ulcerative colitis

Representative mechanism path:
> Cimetidine —[binds]→ gene-8647 —[is bound by]→ DB01234 —[treats]→ Asthma

### Tacrolimus  (strength 7.683)
Hits (20): Asthma, Bipolar disorder, Chagas disease, Colorectal cancer, Coronary artery disease, Crohns disease, Duchenne muscular dystrophy, Gaucher disease, Heart failure, Leishmaniasis, Lymphoma, Melanoma, Multiple sclerosis, Osteoporosis, Psoriasis, Pulmonary hypertension, Schistosomiasis, Sepsis, Tuberculosis, Ulcerative colitis

Representative mechanism path:
> Tacrolimus —[downregulates]→ gene-991 —[is downregulated by]→ DB00938 —[treats]→ Asthma

### Urea  (strength 5.181)
Hits (11): Asthma, Depression, Hepatitis C, Hypertension, Liver cirrhosis, Osteoporosis, Parkinsons disease, Psoriasis, Pulmonary hypertension, Rheumatoid arthritis, Type 2 diabetes

Representative mechanism path:
> Urea —[is a therapeutic for]→ MESH:D012871 —[is treated by]→ DB00741 —[treats]→ Asthma

### Hydrocortisone  (strength 4.551)
Hits (11): Chagas disease, Crohns disease, Cystic fibrosis, Duchenne muscular dystrophy, Endometriosis, Fabry disease, HIV, Heart failure, Idiopathic pulmonary fibrosis, Lupus, Psoriasis

Representative mechanism path:
> Hydrocortisone —[treats]→ DOID:13189 —[is treated by]→ DB00437 —[is a therapeutic for]→ Chagas disease

### Fluorouracil  (strength 4.15)
Hits (11): Endometriosis, Glioblastoma, Idiopathic pulmonary fibrosis, Leukemia, Liver cirrhosis, Lung cancer, Lymphoma, Multiple myeloma, Multiple sclerosis, Pancreatic cancer, Rheumatoid arthritis

Representative mechanism path:
> Fluorouracil —[binds]→ gene-1558 —[is bound by]→ DB00603 —[treats]→ Endometriosis

### Estradiol  (strength 3.842)
Hits (9): Amyotrophic lateral sclerosis, COPD, Colorectal cancer, Depression, Endometriosis, Idiopathic pulmonary fibrosis, Liver cirrhosis, Ovarian cancer, Pancreatic cancer

Representative mechanism path:
> Estradiol —[downregulates]→ gene-22920 —[is linked to]→ Amyotrophic lateral sclerosis

### Methylprednisolone  (strength 3.654)
Hits (9): COPD, Crohns disease, Duchenne muscular dystrophy, Lymphoma, Melanoma, Pancreatic cancer, Prostate cancer, Psoriasis, Sepsis

Representative mechanism path:
> Methylprednisolone —[palliates]→ DOID:2531 —[is treated by]→ DB00860 —[treats]→ COPD

### Folic Acid  (strength 3.574)
Hits (9): Alzheimers disease, Breast cancer, Crohns disease, Hepatitis C, Inflammatory bowel disease, Leishmaniasis, Lung cancer, Multiple sclerosis, Ovarian cancer

Representative mechanism path:
> Folic Acid —[is a therapeutic for]→ Alzheimer's disease

### Prednisolone  (strength 3.437)
Hits (8): Atherosclerosis, Chronic kidney disease, HIV, Inflammatory bowel disease, Lymphoma, Obesity, Psoriasis, Ulcerative colitis

Representative mechanism path:
> Prednisolone —[downregulates]→ gene-1027 —[inhibits]→ gene-207 —[is linked to]→ Atherosclerosis

### Acetylcysteine  (strength 3.388)
Hits (8): Amyotrophic lateral sclerosis, COPD, Epilepsy, Fabry disease, Glioblastoma, Multiple myeloma, Parkinsons disease, Ulcerative colitis

Representative mechanism path:
> Acetylcysteine —[is a therapeutic for]→ Amyotrophic lateral sclerosis

### Progesterone  (strength 3.357)
Hits (8): COPD, Cystic fibrosis, Endometriosis, HIV, Liver cirrhosis, Osteoporosis, Sickle cell disease, Tuberculosis

Representative mechanism path:
> Progesterone —[is a therapeutic for]→ COPD

### Paclitaxel  (strength 3.283)
Hits (8): Alzheimers disease, Gaucher disease, Glioblastoma, Inflammatory bowel disease, Lung cancer, Lupus, Lymphoma, Parkinsons disease

Representative mechanism path:
> Paclitaxel —[downregulates]→ gene-3692 —[is a target of]→ DB09130 —[is a therapeutic for]→ Alzheimer's disease

### Rebamipide  (strength 3.099)
Hits (9): Colorectal cancer, Crohns disease, Duchenne muscular dystrophy, Gaucher disease, Huntingtons disease, Schistosomiasis, Sepsis, Tuberculosis, Ulcerative colitis

Representative mechanism path:
> Rebamipide —[is a therapeutic for]→ MESH:D007511 —[is treated by]→ DB00945 —[treats]→ Colorectal cancer

### Melatonin  (strength 3.007)
Hits (7): Amyotrophic lateral sclerosis, Anxiety, Epilepsy, Idiopathic pulmonary fibrosis, Leukemia, Parkinsons disease, Schizophrenia

Representative mechanism path:
> Melatonin —[is a therapeutic for]→ Amyotrophic lateral sclerosis

### histamine dihydrochloride  (strength 2.943)
Hits (6): Atherosclerosis, Fabry disease, Idiopathic pulmonary fibrosis, Lung cancer, Lupus, Multiple sclerosis

Representative mechanism path:
> histamine dihydrochloride —[is a therapeutic for]→ MESH:D051437 —[is treated by]→ DB00627 —[treats]→ Atherosclerosis

### Budesonide  (strength 2.506)
Hits (7): Crohns disease, HIV, Lupus, Multiple myeloma, Multiple sclerosis, Psoriasis, Ulcerative colitis

Representative mechanism path:
> Budesonide —[is a therapeutic for]→ Crohn's disease

### Ouabain  (strength 2.479)
Hits (6): COPD, Cystic fibrosis, Leukemia, Lung cancer, Malaria, Rheumatoid arthritis

Representative mechanism path:
> Ouabain —[downregulates]→ gene-79415 —[is upregulated by]→ COPD

### Tretinoin  (strength 2.329)
Hits (7): Breast cancer, HIV, Lung cancer, Lupus, Lymphoma, Melanoma, Prostate cancer

Representative mechanism path:
> Tretinoin —[downregulates]→ gene-27338 —[is downregulated by]→ DB00563 —[treats]→ Breast cancer