from math import log
from typing import Dict, List, Any
from d20_types import Concept, Domain

class DomainMetricsCalculator:
    """
    Calcula métricas para evaluar traducciones entre dominios
    """
    
    def calculate_all_metrics(self, source: Concept, target: Concept, conflicts: List[Dict]) -> Dict[str, float]:
        """
        Calcula todas las métricas definidas
        """
        return {
            "ontological_loss": self.calculate_ontological_loss(source, target),
            "cross_domain_coherence": self.calculate_coherence(source, target),
            "practical_utility": self.calculate_utility(target),
            "misinterpretation_risk": self.calculate_misinterpretation_risk(source, target, conflicts),
            "information_preservation": self.calculate_information_preservation(source, target)
        }
    
    def calculate_ontological_loss(self, source: Concept, target: Concept) -> float:
        """
        Calcula qué fracción de propiedades se pierden
        """
        source_props = set(source.properties.keys())
        target_props = set(target.properties.keys())
        
        # Propiedades preservadas (considerando mapeo)
        preserved = 0
        total = len(source_props)
        
        if total == 0:
            return 0.0
        
        for prop in source_props:
            # Verificar si la propiedad existe o tiene equivalente
            if prop in target_props:
                preserved += 1
            elif f"approximation_of_{prop}" in target_props:
                preserved += 0.5  # Aproximación parcial
            elif "approximation_of" in target.properties:
                preserved += 0.3  # Aproximación general
        
        loss = 1.0 - (preserved / total)
        return loss
    
    def calculate_coherence(self, source: Concept, target: Concept) -> float:
        """
        Calcula coherencia lógica entre las representaciones
        """
        # Verificar consistencia interna del target
        internal_consistency = self._check_internal_consistency(target)
        
        # Verificar consistencia con source (relajada para traducciones)
        cross_consistency = self._check_cross_consistency(source, target)
        
        # Coherencia general
        coherence = (internal_consistency + cross_consistency) / 2
        
        return coherence

    def calculate_information_preservation(self, source: Concept, target: Concept) -> float:
        # Placeholder for information preservation calculation
        # Could be 1 - loss from generic perspective
        return 1.0 - self.calculate_ontological_loss(source, target)
    
    def calculate_utility(self, target: Concept) -> float:
        """
        Calcula utilidad práctica de la representación
        """
        # Factores: implementabilidad, eficiencia, claridad
        implementability = self._estimate_implementability(target)
        efficiency = self._estimate_efficiency(target)
        clarity = self._estimate_clarity(target)
        
        utility = (implementability * efficiency * clarity) ** (1/3)
        
        # Ajustar por complejidad
        complexity = self._estimate_complexity(target)
        if complexity > 0:
            utility /= (1 + log(1 + complexity))
        
        return utility
    
    def calculate_misinterpretation_risk(self, source: Concept, target: Concept, conflicts: List[Dict]) -> float:
        """
        Calcula riesgo de que la traducción lleve a malinterpretaciones
        """
        # Base risk from conflicts (with Mitigation)
        total_severity = 0.0
        
        for c in conflicts:
            severity = c.get("severity", 0)
            c_type = c.get("type", "")
            
            # Mitigation Logic: Check if target handles this conflict type
            # e.g. conflict "infinite_to_finite" -> mitigated by "approximation_of_infinite"
            mitigation_key = f"approximation_of_{c_type.split('_to_')[0]}" # approximation_of_infinite
            
            # Alternative: general approximation flag
            handled = False
            if mitigation_key in target.properties:
                handled = True
            elif "approximation_of" in target.properties and c_type in ["infinite_to_finite", "continuous_to_discrete"]:
                # General approximation covers structural changes if explicit
                handled = True
                
            if handled:
                severity *= 0.4 # Reduce severity by 60% if explicitly handled
                
            total_severity += severity

        conflict_risk = total_severity / max(len(conflicts), 1)
        
        # Risk from ambiguity
        ambiguity = self._estimate_ambiguity(target)
        
        # Risk from domain distance
        domain_distance = self._calculate_domain_distance(source.domain, target.domain)
        
        # Combine risks
        total_risk = (conflict_risk * 0.4 + ambiguity * 0.3 + domain_distance * 0.3)
        
        return min(1.0, total_risk)

    # --- Helper Estimation Methods (Heuristics) ---

    def _check_internal_consistency(self, concept: Concept) -> float:
        # Naive check: Does it have constraints?
        if not concept.constraints: return 0.5
        return 0.9 # Assume internally consistent if generated by Arbiter

    def _check_cross_consistency(self, source: Concept, target: Concept) -> float:
        if source.domain == target.domain: return 1.0
        
        # Consistency depends on how far the domains are
        dist = self._calculate_domain_distance(source.domain, target.domain)
        return max(0.0, 1.0 - (dist * 0.5)) # e.g. dist 0.3 -> 0.85 consistency

    def _estimate_implementability(self, concept: Concept) -> float:
        if concept.domain in [Domain.COMPUTATION_DETERMINISTIC, Domain.ENGINEERING]:
            return 1.0
        if concept.domain == Domain.PHYSICS_CLASSICAL:
            return 0.8
        return 0.3 # Math/Quantum harder to implement directly

    def _estimate_efficiency(self, concept: Concept) -> float:
        return 1.0 # Default

    def _estimate_clarity(self, concept: Concept) -> float:
        if "approximation_of" in concept.properties:
            return 0.8 # Explicit approximations are clear
        return 1.0

    def _estimate_complexity(self, concept: Concept) -> float:
        return len(concept.properties) * 0.05

    def _estimate_ambiguity(self, concept: Concept) -> float:
        if "probabilistic" in concept.properties and concept.properties["probabilistic"]:
            return 0.6
        return 0.1

    def _calculate_domain_distance(self, d1: Domain, d2: Domain) -> float:
        # Simple distance matrix
        if d1 == d2: return 0.0
        
        # Math <-> CS (Close relationship)
        if (d1 in [Domain.MATHEMATICS_INFINITE, Domain.MATHEMATICS_FINITE] and 
            d2 in [Domain.COMPUTATION_DETERMINISTIC, Domain.COMPUTATION_QUANTUM]):
            return 0.3
            
        # Quantum <-> Classical (Medium-High)
        if (d1 == Domain.PHYSICS_QUANTUM and d2 == Domain.PHYSICS_CLASSICAL):
            return 0.7
            
        return 0.5
