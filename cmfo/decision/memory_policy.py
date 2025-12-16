"""
CMFO D5: Memory as Experience (Refined)
========================================
Human-like memory integration principles.

CORE PRINCIPLE:
Memory always influences, rarely mentioned.

Citation only improves understanding, never for self-reference.
"""

# Citation Policy by Intent
CITATION_POLICY = {
    'confirm': {
        'cite': False,
        'reason': 'Confirmation needs no justification'
    },
    'reference': {
        'cite': 'optional',
        'reason': 'Only if clarifies context',
        'threshold': 0.12  # Only same precedent
    },
    'question': {
        'cite': False,
        'reason': 'Memory guides implicitly, not explicitly'
    },
    'correct': {
        'cite': True,
        'reason': 'Corrections require justification',
        'threshold': 0.25  # Same or related
    },
    'conflict': {
        'cite': True,
        'reason': 'Conflicts require evidence',
        'threshold': 0.25  # Same or related
    }
}

# Dual Thresholds (d_phi)
THRESHOLD_SAME = 0.12       # Same precedent (same experience)
THRESHOLD_RELATED = 0.25    # Related experience
# > 0.25 = Irrelevant (don't cite)

# Human-like Framing
# ❌ NEVER say:
# - "Según mi memoria..."
# - "Tengo 3 entradas similares..."
# - "En mi base de datos..."
# - "Memoria #M:0001234..."

# ✅ ALWAYS frame as implicit experience:
# - "Esto contradice lo anterior"
# - "En casos similares, la conclusión fue distinta"
# - "Lo que vimos antes sugiere..."
# - "El patrón es inconsistente"

# Forgetting Policy (Weight Decay)
def memory_weight(entry, current_time):
    """
    Calculate memory weight (influence without deletion).
    
    Factors:
    - confidence: Original decision confidence
    - usage: How many times this memory influenced decisions
    - recency: Time decay (soft, not hard cutoff)
    """
    age_hours = (current_time - entry.timestamp) / 3600
    
    # Recency factor (soft decay, never zero)
    recency = 1.0 / (1.0 + age_hours / 168)  # Half-life ~1 week
    
    # Usage factor (placeholder - would track actual usage)
    usage = 1.0  # TODO: Track citation/influence count
    
    # Combined weight
    return entry.confidence * usage * recency


# Short-term Memory Size
SHORT_TERM_SIZE = 10  # ✓ Correct, don't change
# Humans sustain ~7-9 active cognitive acts
# CMFO aligned with cognitive science

if __name__ == "__main__":
    print("CMFO D5: Memory as Experience")
    print("=" * 60)
    print("\nCitation Policy:")
    for intent, policy in CITATION_POLICY.items():
        cite_str = "YES" if policy['cite'] is True else ("OPTIONAL" if policy['cite'] == 'optional' else "NO")
        print(f"  {intent:12s}: {cite_str:8s} - {policy['reason']}")
    
    print(f"\nThresholds:")
    print(f"  Same precedent:    d_phi < {THRESHOLD_SAME}")
    print(f"  Related experience: {THRESHOLD_SAME} < d_phi < {THRESHOLD_RELATED}")
    print(f"  Irrelevant:        d_phi > {THRESHOLD_RELATED}")
    
    print(f"\nShort-term size: {SHORT_TERM_SIZE} entries")
    print("\nMemory = Experience, not narration.")
