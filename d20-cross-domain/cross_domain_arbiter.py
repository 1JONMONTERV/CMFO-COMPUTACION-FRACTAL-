import numpy as np
from typing import Dict, List, Tuple, Any
from math import log, sqrt
from d20_types import Concept, Domain, TranslationResult
from domain_metrics import DomainMetricsCalculator

class CrossDomainArbiter:
    """
    Núcleo de D20: Detecta, resuelve y explica conflictos entre dominios
    """
    
    def __init__(self):
        # Tabla de correspondencias entre dominios
        self.domain_mappings = self._initialize_domain_mappings()
        
        # Reglas de transformación específicas
        self.transformation_rules = self._initialize_transformation_rules()
        
        # Sistema de métricas
        self.metrics_calculator = DomainMetricsCalculator()
        
        # Logger para trazabilidad completa
        self.conflict_log = []

        # [D22 Integration] Dominio Maestro y [D23 Integration] Legislador
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'd22-explicit-domain'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'd23-exigences'))
        
        try:
            from d22_cmfo_domain import CMFODomain, DomainType, CMFOMetrics
            from d23_exigences import ExigenceManager, ExigenceType
            
            self.cmfo_domain = CMFODomain()
            self.legislator = ExigenceManager()
            self.d22_active = True
        except ImportError as e:
            print(f"Warning: D22/D23 modules not found ({e}). Running in legacy D20 mode.")
            self.d22_active = False
    
    def translate_concept(self, source_concept: Concept, target_domain: Domain) -> TranslationResult:
        """
        Traduce un concepto entre dominios, manejando conflictos via Legislación D23
        """
        # Paso 1: Detección de conflictos via Legislador D23
        conflicts = self._detect_domain_conflicts(source_concept, target_domain)
        
        # Paso 2: Selección de estrategia de resolución
        resolution_strategy = self._select_resolution_strategy(conflicts)
        
        # Paso 3: Aplicar transformación
        if resolution_strategy == "BLOCKED_BY_LEGISLATION":
            # Caso especial: Bloqueo explícito por contaminación
            return self._create_blocked_result(source_concept, target_domain, conflicts)

        translated_concept, transformation_trace = self._apply_transformation(
            source_concept, target_domain, resolution_strategy
        )
        
        # Paso 4: Calcular métricas (usando D22 si disponible)
        metrics = self.metrics_calculator.calculate_all_metrics(
            source_concept, translated_concept, conflicts
        )
        
        if self.d22_active:
             # Enriquecer métricas con Curvatura Ontológica D22
             curvature = self._get_d22_curvature(source_concept.domain.value, target_domain.value)
             metrics['ontological_curvature'] = curvature

        # Paso 5: Generar advertencias apropiadas
        warnings = self._generate_warnings(metrics, conflicts)
        
        # Paso 6: Calcular confianza final
        confidence = self._calculate_confidence(metrics)
        
        # Paso 7: Loggear para aprendizaje futuro
        self._log_translation(source_concept, translated_concept, metrics, conflicts)
        
        return TranslationResult(
            concept=translated_concept,
            metrics=metrics,
            warnings=warnings,
            transformation_trace=transformation_trace,
            confidence=confidence
        )
    
    def _detect_domain_conflicts(self, source: Concept, target: Domain) -> List[Dict]:
        """
        Detecta conflictos consultando al Legislador D23
        """
        if not self.d22_active:
             return self._legacy_conflict_detection(source, target)

        conflicts = []
        
        # Contexto base para el legislador
        context = {
            "involves_infinity": source.properties.get("infinite", False),
            "involves_superposition": source.properties.get("superposition", False),
            "source_domain": source.domain.value,
            "target_domain": target.value
        }

        # Consultar legislador para ambos dominios
        # Nota: Convertimos enum Domain a string que espera D23 (e.g., Domain.MATHEMATICS_INFINITE -> 'Mathematics')
        # Esto requiere un mapeo simple o usar nombres aproximados. 
        # Para esta implementación, usaremos un mapeo directo simplificado.
        
        domain_map = {
            Domain.MATHEMATICS_INFINITE: "Mathematics",
            Domain.COMPUTATION_DETERMINISTIC: "Computation",
            Domain.PHYSICS_QUANTUM: "QuantumComp",
            Domain.PHYSICS_CLASSICAL: "Physics",
            Domain.ENGINEERING: "Engineering", 
            Domain.CMFO_7D: "CMFO"
        }
        
        src_str = domain_map.get(source.domain, "Unknown")
        tgt_str = domain_map.get(target, "Unknown")
        
        legislative_report = self.legislator.get_exigences_for_domains([src_str, tgt_str], context)
        
        # 1. Analizar Conflictos Reportados por D23 (Inter-dominio)
        if "_CONFLICTS" in legislative_report:
            for conflict_desc in legislative_report["_CONFLICTS"]:
                conflicts.append({
                    "type": "legislative_conflict",
                    "severity": 0.9, # Alta severidad por defecto para tensiones legislativas
                    "description": conflict_desc,
                    "properties_at_risk": ["axiomatic_consistency"]
                })

        # 2. Analizar Violaciones de Restricciones (Intra-dominio target)
        # Si el dominio destino tiene una restricción que el concepto origen viola flagrantemente
        if tgt_str in legislative_report:
            for exigence in legislative_report[tgt_str]:
                if exigence.type.value == "constraint":
                    # Chequeo simple de keywords
                    if "terminacion" in exigence.description.lower() and source.properties.get("infinite", False):
                         conflicts.append({
                            "type": "axiom_violation",
                            "severity": 1.0,
                            "description": f"Violación de axioma soberano en destino: {exigence.description}",
                            "properties_at_risk": ["computability"]
                         })
                
                if "PROHIBIDO" in exigence.description:
                     # Chequeo de no-contaminación
                     # (Simplificado: si origen es Teología y destino es Física)
                     # En este código D20, 'source' es un Concepto científico, pero 
                     # si extendemos a Theology, esto atraparía la contaminación.
                     pass

        # Mantener reglas legacy estructurales (dimensionalidad, etc) no cubiertas explicitamente por D23 simple
        legacy = self._legacy_conflict_detection(source, target)
        # Filtrar duplicados si es necesario, pero por ahora sumamos
        # (Idealmente D23 cubriría todo, pero por seguridad sumamos)
        for l in legacy:
             # Evitar duplicar infinito vs finito si ya lo atrapó el legislador
             if l["type"] == "infinite_to_finite" and any(c["type"] == "axiom_violation" for c in conflicts):
                 continue
             conflicts.append(l)

        return conflicts

    def _legacy_conflict_detection(self, source: Concept, target: Domain) -> List[Dict]:
        """Lógica original de detección (Legacy)"""
        conflicts = []
        def has_property(c, p): return c.properties.get(p, False)

        # Regla 1: Infinito ↔️ Finito
        if has_property(source, "infinite") and target in [Domain.COMPUTATION_DETERMINISTIC, Domain.ENGINEERING]:
            conflicts.append({
                "type": "infinite_to_finite",
                "severity": 0.8,
                "description": f"Concepto infinito en dominio finito: {source.identifier}",
                "properties_at_risk": ["infinite", "unbounded", "limitless"]
            })
        
        # Regla 2: Continuo ↔️ Discreto
        if has_property(source, "continuous") and target in [Domain.COMPUTATION_DETERMINISTIC]:
            conflicts.append({
                "type": "continuous_to_discrete",
                "severity": 0.6,
                "description": "Transformación continua → discreta",
                "properties_at_risk": ["continuous", "differentiable", "smooth"]
            })
        
        # Regla 4: Alta Dimensión ↔️ Baja Dimensión
        if has_property(source, "high_dimensional") and target in [Domain.PHYSICS_CLASSICAL, Domain.ENGINEERING]:
            dimension = source.properties.get("dimensions", 7)
            if dimension > 3:
                conflicts.append({
                    "type": "dimension_reduction",
                    "severity": 0.5,
                    "description": f"Reducción de {dimension}D a 3D/4D",
                    "properties_at_risk": [f"dimension_{i}" for i in range(4, dimension+1)]
                })

        return conflicts

    def _get_d22_curvature(self, source_name: str, target_name: str) -> float:
        """Obtiene curvatura ontológica desde D22"""
        # Mapeo rápido de nombres D20 -> D22
        # (Esto seria mas robusto en prod)
        try:
            # Asumimos que CMFODomain tiene un metodo público o static
            # Accedemos a la clase CMFOMetrics importada dinamicamente
            from d22_cmfo_domain import CMFOMetrics, DomainType
            
            # Buscar enum members
            s_enum = next((m for m in DomainType if m.value in source_name.lower()), None)
            t_enum = next((m for m in DomainType if m.value in target_name.lower()), None)
            
            # Fallback mappings
            if "math" in source_name.lower(): s_enum = DomainType.MATHEMATICS
            if "comp" in source_name.lower(): t_enum = DomainType.COMPUTATION
            if "phys" in source_name.lower(): t_enum = DomainType.PHYSICS
            
            if s_enum and t_enum:
                return CMFOMetrics.ontological_curvature(s_enum, t_enum)
            return 0.5 # Default tension
        except Exception:
            return 0.5

    def _create_blocked_result(self, source: Concept, target: Domain, conflicts: List[Dict]) -> TranslationResult:
        """Crea un resultado fallido por bloqueo legislativo"""
        return TranslationResult(
            concept=Concept("BLOCKED", target, {}, "Blocked", []),
            metrics={"ontological_loss": 1.0, "practical_utility": 0.0},
            warnings=["⛔ BLOQUEO LEGISLATIVO: Violación de axiomas soberanos"] + [c["description"] for c in conflicts],
            transformation_trace=["Intento de traducción bloqueado por D23"],
            confidence=0.0
        )

    def _select_resolution_strategy(self, conflicts: List[Dict]) -> str:
        """
        Selecciona la mejor estrategia basada en los conflictos detectados
        """
        if not conflicts:
            return "direct_translation"
        
        # Clasificar por severidad
        max_severity = max((c["severity"] for c in conflicts), default=0)
        
        # [D23 Check] Si hay violación axiomática total, bloquear
        if max_severity >= 1.0:
            return "BLOCKED_BY_LEGISLATION"

        if max_severity >= 0.8:
            return "approximation_with_explicit_loss"
        elif max_severity >= 0.5:
            return "constrained_approximation"
        else:
            return "property_preserving_translation"
    
    def _apply_transformation(self, source: Concept, target: Domain, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Aplica la transformación seleccionada
        """
        trace = [f"Estrategia: {strategy}", f"Origen: {source.domain} → Destino: {target}"]
        
        # Buscar transformador específico
        transformer_key = (source.domain, target)
        
        if transformer_key in self.transformation_rules:
            transformer = self.transformation_rules[transformer_key]
            result, subtrace = transformer(source, strategy)
            trace.extend(subtrace)
            return result, trace
        
        # Transformación genérica vía espacio CMFO intermedio (Fallback)
        trace.append("Usando transformación genérica CMFO (Fallback)")
        result = self._generic_cmfo_transformation(source, target, strategy)
        
        return result, trace
    
    def _initialize_domain_mappings(self):
        return {} # Placeholder

    def _initialize_transformation_rules(self) -> Dict:
        """
        Inicializa todas las reglas de transformación específicas
        """
        return {
            # INFINITO → FINITO
            (Domain.MATHEMATICS_INFINITE, Domain.COMPUTATION_DETERMINISTIC): 
                self._transform_infinite_to_finite,
            
            (Domain.MATHEMATICS_INFINITE, Domain.ENGINEERING): 
                self._transform_infinite_to_engineering,
            
            # CUÁNTICO → CLÁSICO
            (Domain.PHYSICS_QUANTUM, Domain.COMPUTATION_DETERMINISTIC): 
                self._transform_quantum_to_classical,
            
            (Domain.PHYSICS_QUANTUM, Domain.PHYSICS_CLASSICAL): 
                self._transform_quantum_to_classical_physics,
            
            # ALTA DIMENSIÓN → BAJA DIMENSIÓN
            (Domain.CMFO_7D, Domain.PHYSICS_CLASSICAL): 
                self._transform_7d_to_3d,
            
            (Domain.CMFO_7D, Domain.ENGINEERING): 
                self._transform_7d_to_engineering,
            
            # TIEMPO CERO → TIEMPO UNO
            (Domain.MATHEMATICS_INFINITE, Domain.PHYSICS_CLASSICAL): 
                self._transform_timeless_to_temporal,
        }
    
    # === IMPLEMENTACIONES DE TRANSFORMADORES ESPECÍFICOS ===
    
    def _transform_infinite_to_finite(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Transforma concepto infinito a representación finita computable
        """
        trace = []
        
        if "natural_numbers" in source.identifier:
            # Teoría: ℕ → {0, 1, ..., N} con N "suficientemente grande"
            trace.append("Aplicando principio del 'suficientemente grande'")
            
            # Calcular N basado en aplicación práctica (mocked)
            N = 18446744073709551615 # uint64 max
            
            result = Concept(
                identifier=f"bounded_{source.identifier}",
                domain=Domain.COMPUTATION_DETERMINISTIC,
                properties={
                    "finite": True,
                    "cardinality": N,
                    "upper_bound": N,
                    "approximation_of": source.identifier,
                    # Preserved Properties
                    "countable": True,    # Finite sets are countable
                    "well_ordered": True, # Finite sets of ints are well ordered
                    "approximation_of_infinite": True # Explicit acknowledgement
                },
                formal_definition=f"{{0, 1, 2, ..., {N}}} where {N} is MAX_UINT64",
                constraints=[f"x <= {N}", "x in Z", "x >= 0"]
            )
            
            trace.append(f"Establecido límite superior N={N}")
            return result, trace
        
        raise NotImplementedError(f"Transformación no implementada para: {source.identifier}")

    def _transform_infinite_to_engineering(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Translates Infinity to a Safety Factor.
        Engineering doesn't use infinity; it uses 'Safe Operating Limits'.
        """
        trace = ["Translating Infinite Concept to Engineering Safety Limit"]
        
        # Assume "Infinity" means "Never breaks under normal load"
        # Translate to: High Safety Factor
        
        result = Concept(
            identifier=f"engineering_{source.identifier}",
            domain=Domain.ENGINEERING,
            properties={
                "finite": True,
                "safety_factor": 10.0, # 10x Max Load
                "tolerance": "1e-6",
                "approximation_of": source.identifier,
                "approximation_of_infinite": True 
            },
            formal_definition=f"Limit(L) where L >> MaxLoad",
            constraints=["Must withstand 10x nominal load"]
        )
        
        trace.append(f"Infinite converted to Safety Factor 10.0")
        return result, trace

    def _transform_quantum_to_classical(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Transforma concepto cuántico a representación clásica computable
        """
        trace = []
        
        if source.identifier == "quantum_superposition":
            trace.append("Aproximando superposición cuántica con probabilidades clásicas")
            
            # Extraer amplitudes (simplificado)
            alpha = source.properties.get("alpha", 1/sqrt(2))
            beta = source.properties.get("beta", 1/sqrt(2))
            
            result = Concept(
                identifier="classical_probabilistic_bit",
                domain=Domain.COMPUTATION_DETERMINISTIC,
                properties={
                    "deterministic": False,
                    "probabilistic": True,
                    "p0": abs(alpha)**2,
                    "p1": abs(beta)**2,
                    "approximation_of": "quantum_superposition",
                    "approximation_of_superposition": True,
                    "approximation_of_coherent": False
                },
                formal_definition="Random variable X with P(X=0)=|alpha|^2, P(X=1)=|beta|^2",
                constraints=["0 <= p0 <= 1", "0 <= p1 <= 1", "p0 + p1 = 1"]
            )
            
            trace.append(f"Traducido |psi> = {alpha}|0> + {beta}|1> -> P(0)={abs(alpha)**2:.3f}, P(1)={abs(beta)**2:.3f}")
            trace.append("ADVERTENCIA: Pérdida de coherencia y fase cuántica")
            
            return result, trace
        
        raise NotImplementedError(f"Transformación cuántica no implementada: {source.identifier}")

    def _transform_quantum_to_classical_physics(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
         raise NotImplementedError("Quantum->Classical Phys not implemented")

    def _transform_7d_to_engineering(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Translates 7D Complex State to a 'Black Box' Control System.
        Engineers don't care about the 7 dimensions, they care about Input/Output.
        """
        trace = ["Translating 7D Fractal State to Control System Box"]
        
        # Simplified Model: Treat 7D internal state as "Hidden State"
        # Expose only 3 Control dimensions
        
        coords = source.properties.get("coordinates", [])
        visible_outputs = coords[:2] if len(coords) >= 2 else []
        
        result = Concept(
            identifier=f"control_sys_{source.identifier}",
            domain=Domain.ENGINEERING,
            properties={
                "type": "BlackBox",
                "inputs": ["voltage", "current"],
                "outputs": ["pos_x", "pos_y"], # First 2 dims
                "internal_state_dim": 7, # Acknowledged but hidden
                "approximation_of": source.identifier,
                "approximation_of_high_dimensional": True # Reduce OL
            },
            formal_definition="System y = f(u, x_internal)",
            constraints=["Linearized around operating point"]
        )
        
        trace.append("Mapped 7D state to Black Box Transfer Function")
        trace.append("Warning: Nonlinearities in hidden dims ignored")
        return result, trace

    def _transform_7d_to_3d(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Proyecta concepto 7D CMFO a espacio 3D físico
        """
        trace = ["Proyectando espacio CMFO 7D a 3D físico"]
        
        coordinates = source.properties.get("coordinates", [1.618, 0.618, 2.618, 1.0, 1.272, 0.786, 1.466])
        
        # Proyección: primeros 3 componentes (simplificado)
        projected = coordinates[:3]
        
        # Calcular información perdida
        lost_dimensions = coordinates[3:]
        info_loss = sqrt(sum(x**2 for x in lost_dimensions))
        
        result = Concept(
            identifier=f"projected_{source.identifier}",
            domain=Domain.PHYSICS_CLASSICAL,
            properties={
                "dimensions": 3,
                "coordinates": projected,
                "projected_from_7d": True,
                "information_loss": info_loss,
                "lost_dimensions": lost_dimensions
            },
            formal_definition=f"Projection(V7) -> V3",
            constraints=["Solo aproximación", "Pérdida dimensional inevitable"]
        )
        
        trace.append(f"Coordenadas originales 7D: {coordinates}")
        trace.append(f"Proyección 3D: {projected}")
        trace.append(f"Pérdida informacional: {info_loss:.3f}")
        
        return result, trace
    
    def _transform_timeless_to_temporal(self, source: Concept, strategy: str) -> Tuple[Concept, List[str]]:
        """
        Añade dimensión temporal a objeto matemático atemporal
        """
        trace = ["Añadiendo dimensión temporal a objeto matemático atemporal"]
        
        result = Concept(
            identifier=f"temporal_{source.identifier}",
            domain=Domain.PHYSICS_CLASSICAL,
            properties={
                **source.properties,
                "temporal": True,
                "static": False,
                "evolves": True,
                "original_atemporal": True
            },
            formal_definition=f"{source.formal_definition} AND t in R+",
            constraints=source.constraints + ["t >= 0", "d/dt exists"]
        )
        
        trace.append("Objeto ahora depende del tiempo: f(x) -> f(x,t)")
        trace.append("ADVERTENCIA: Esta es una interpretación física, no matemática")
        
        return result, trace
    
    def _generic_cmfo_transformation(self, source: Concept, target: Domain, strategy: str) -> Concept:
        # Fallback
        return Concept(
            identifier=f"generic_{source.identifier}",
            domain=target,
            properties=source.properties.copy(),
            formal_definition=source.formal_definition,
            constraints=source.constraints
        )

    # === MÉTRICAS Y CONFIANZA ===
    
    def _generate_warnings(self, metrics: Dict[str, float], conflicts: List[Dict]) -> List[str]:
        """
        Genera advertencias apropiadas basadas en métricas
        """
        warnings = []
        
        # Advertencia por pérdida ontológica alta
        if metrics.get("ontological_loss", 0) > 0.5:
            warnings.append(f"⚠️ ALTA PÉRDIDA ONTOLÓGICA ({metrics['ontological_loss']:.1%})")
            warnings.append("   Se han perdido propiedades esenciales del concepto original")
        
        # Advertencia por riesgo de malinterpretación
        if metrics.get("misinterpretation_risk", 0) > 0.3:
            warnings.append(f"⚠️ RIESGO DE MALINTERPRETACIÓN ({metrics['misinterpretation_risk']:.1%})")
            warnings.append("   Esta traducción puede llevar a conclusiones erróneas")
        
        # Advertencia por severidad de conflictos
        if conflicts:
            max_severity = max(c["severity"] for c in conflicts)
            if max_severity > 0.7:
                warnings.append(f"⚠️ CONFLICTO ONTOLÓGICO GRAVE (severidad: {max_severity:.1%})")
                for conflict in conflicts:
                    if conflict["severity"] > 0.7:
                        warnings.append(f"   • {conflict['description']}")
        
        # Si no hay advertencias, añadir confirmación
        if not warnings:
            warnings.append("✓ Traducción ontológicamente segura")
        
        return warnings
    
    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """
        Calcula confianza general en la traducción
        """
        # Fórmula: confianza = coherencia * (1 - riesgo) * (1 - pérdida/2)
        coherence = metrics.get("cross_domain_coherence", 0.5)
        risk = metrics.get("misinterpretation_risk", 0.3)
        loss = metrics.get("ontological_loss", 0.2)
        
        confidence = coherence * (1 - risk) * (1 - loss/2)
        
        # Ajustar por utilidad práctica
        utility = metrics.get("practical_utility", 1.0)
        confidence *= min(utility, 2.0)  # Capar en 2.0 para no exagerar
        
        return max(0.0, min(1.0, confidence))
    
    def _log_translation(self, source, target, metrics, conflicts):
        self.conflict_log.append({
            "source": source.identifier,
            "target": target.identifier,
            "conflicts": conflicts,
            "metrics": metrics
        })
    
    def _calculate_practical_bound(self, source) -> int:
        return 18446744073709551615
