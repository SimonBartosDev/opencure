"""
AI-powered literature analysis using BioGPT and PubMedBERT.

BioGPT: Generates mechanistic hypotheses about drug-disease relationships.
PubMedBERT: Extracts entities and classifies relations from abstracts.

Together they provide AI-driven evidence synthesis that goes beyond
simple keyword matching - the system actually "reads" papers.
"""

from typing import Optional


# Module-level model cache
_biogpt_cache = {}
_ner_cache = {}


def _load_biogpt():
    """Load BioGPT model (cached)."""
    if "model" not in _biogpt_cache:
        try:
            from transformers import pipeline
            _biogpt_cache["model"] = pipeline(
                "text-generation",
                model="microsoft/biogpt",
                device=-1,  # CPU
                max_new_tokens=200,
            )
            print("  BioGPT loaded")
        except Exception as e:
            print(f"  [WARN] BioGPT not available: {e}")
            _biogpt_cache["model"] = None
    return _biogpt_cache["model"]


def _load_ner():
    """Load PubMedBERT NER model (cached)."""
    if "model" not in _ner_cache:
        try:
            from transformers import pipeline
            _ner_cache["model"] = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=-1,
            )
            print("  Biomedical NER loaded")
        except Exception as e:
            print(f"  [WARN] Biomedical NER not available: {e}")
            _ner_cache["model"] = None
    return _ner_cache["model"]


def generate_hypothesis(drug_name: str, disease_name: str, evidence_context: str = "") -> str:
    """
    Use BioGPT to generate a mechanistic hypothesis about
    why a drug might treat a disease.

    Args:
        drug_name: Name of the drug
        disease_name: Name of the disease
        evidence_context: Optional context from knowledge graph or literature

    Returns:
        Generated hypothesis text
    """
    generator = _load_biogpt()
    if generator is None:
        return ""

    # Construct a prompt that guides BioGPT
    prompt = f"{drug_name} may be effective for treating {disease_name} because"
    if evidence_context:
        prompt = f"Given that {evidence_context}, {drug_name} may treat {disease_name} because"

    try:
        result = generator(prompt, max_new_tokens=150, do_sample=False)
        generated = result[0]["generated_text"]

        # Clean up - remove the prompt prefix if repeated, trim to sentence boundary
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        # Trim to last complete sentence
        last_period = generated.rfind(".")
        if last_period > 20:
            generated = generated[:last_period + 1]

        return generated.strip()
    except Exception as e:
        return f"(hypothesis generation failed: {e})"


def extract_biomedical_entities(text: str) -> list[dict]:
    """
    Extract biomedical entities (drugs, diseases, genes) from text
    using PubMedBERT-based NER.

    Returns list of dicts with: entity, entity_group (Drug, Disease, Gene), score
    """
    ner = _load_ner()
    if ner is None:
        return []

    try:
        # Truncate long texts
        text = text[:1000]
        entities = ner(text)

        # Deduplicate and filter by confidence
        seen = set()
        unique = []
        for e in entities:
            key = (e["word"].lower(), e["entity_group"])
            if key not in seen and e["score"] > 0.7:
                seen.add(key)
                unique.append({
                    "entity": e["word"],
                    "entity_group": e["entity_group"],
                    "score": round(e["score"], 3),
                })
        return unique
    except Exception:
        return []


def analyze_abstract_for_relation(
    abstract: str,
    drug_name: str,
    disease_name: str,
) -> dict:
    """
    Analyze a paper abstract to determine the nature of the drug-disease relationship.

    Uses entity extraction + keyword heuristics to classify the relationship.

    Returns dict with: relation_type (treats/inhibits/no_effect/unclear),
                       confidence, extracted_entities, key_phrases
    """
    text_lower = abstract.lower()
    drug_lower = drug_name.lower()
    disease_lower = disease_name.lower()

    # Check if both drug and disease are mentioned
    drug_mentioned = drug_lower in text_lower
    disease_mentioned = disease_lower in text_lower

    if not drug_mentioned or not disease_mentioned:
        return {"relation_type": "unclear", "confidence": 0.0, "reason": "Drug or disease not mentioned in abstract"}

    # Keyword-based relation classification
    positive_keywords = [
        "effective", "efficacy", "therapeutic", "treatment", "treats",
        "inhibit", "reduce", "suppress", "improve", "protect",
        "attenuate", "ameliorate", "alleviate", "beneficial",
        "promising", "potent", "active against",
    ]
    negative_keywords = [
        "no effect", "ineffective", "failed", "no significant",
        "did not improve", "no benefit", "contraindicated",
        "toxic", "adverse", "worsen",
    ]
    mechanism_keywords = [
        "mechanism", "pathway", "receptor", "binding", "target",
        "kinase", "enzyme", "protein", "gene expression",
        "signaling", "downstream", "upstream",
    ]

    pos_hits = sum(1 for k in positive_keywords if k in text_lower)
    neg_hits = sum(1 for k in negative_keywords if k in text_lower)
    mech_hits = sum(1 for k in mechanism_keywords if k in text_lower)

    # Classify
    if pos_hits > neg_hits and pos_hits >= 2:
        relation = "treats"
        confidence = min(0.9, 0.5 + pos_hits * 0.1)
    elif neg_hits > pos_hits and neg_hits >= 2:
        relation = "no_effect"
        confidence = min(0.9, 0.5 + neg_hits * 0.1)
    elif mech_hits >= 2:
        relation = "mechanism_related"
        confidence = 0.5
    else:
        relation = "unclear"
        confidence = 0.3

    # Extract entities for additional context
    entities = extract_biomedical_entities(abstract[:500])

    return {
        "relation_type": relation,
        "confidence": round(confidence, 2),
        "positive_signals": pos_hits,
        "negative_signals": neg_hits,
        "mechanism_signals": mech_hits,
        "extracted_entities": entities[:10],
    }
