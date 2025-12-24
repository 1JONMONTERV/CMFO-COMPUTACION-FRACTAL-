"""
ByteConstraintGraph: Sistema de propagación de restricciones a nivel byte

Implementa AC-3 (Arc Consistency 3) adaptado para restricciones SHA-256
sobre el nonce de Bitcoin.
"""

from typing import Set, List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ============================================================================
# DOMINIO DE BYTES
# ============================================================================

class ByteDomain:
    """Dominio de valores válidos para un byte [0-255]"""
    
    def __init__(self, values: Optional[Set[int]] = None):
        if values is None:
            # Dominio completo por defecto
            self.values = set(range(256))
        else:
            # Validar que todos los valores están en [0, 255]
            assert all(0 <= v <= 255 for v in values), "Valores fuera de rango [0, 255]"
            self.values = set(values)
    
    def __len__(self) -> int:
        return len(self.values)
    
    def __contains__(self, value: int) -> bool:
        return value in self.values
    
    def __iter__(self):
        return iter(sorted(self.values))
    
    def is_empty(self) -> bool:
        """Retorna True si el dominio está vacío (inconsistencia)"""
        return len(self.values) == 0
    
    def is_singleton(self) -> bool:
        """Retorna True si el dominio tiene un solo valor"""
        return len(self.values) == 1
    
    def get_singleton_value(self) -> Optional[int]:
        """Retorna el valor único si es singleton, None en caso contrario"""
        if self.is_singleton():
            return next(iter(self.values))
        return None
    
    def intersect(self, other: 'ByteDomain') -> 'ByteDomain':
        """Intersección de dominios"""
        return ByteDomain(self.values & other.values)
    
    def restrict(self, new_values: Set[int]) -> bool:
        """
        Restringe el dominio a new_values.
        Retorna True si el dominio cambió.
        """
        old_size = len(self.values)
        self.values &= new_values
        return len(self.values) < old_size
    
    def copy(self) -> 'ByteDomain':
        """Copia profunda del dominio"""
        return ByteDomain(self.values.copy())
    
    def __repr__(self) -> str:
        if self.is_empty():
            return "ByteDomain(EMPTY)"
        elif self.is_singleton():
            return f"ByteDomain({{{self.get_singleton_value():#04x}}})"
        elif len(self.values) <= 5:
            vals = ', '.join(f'{v:#04x}' for v in sorted(self.values))
            return f"ByteDomain({{{vals}}})"
        else:
            return f"ByteDomain({len(self.values)} values)"


# ============================================================================
# NODO DEL GRAFO
# ============================================================================

@dataclass
class ByteNode:
    """Nodo del grafo representando una posición de byte"""
    
    position: int  # Posición en el mensaje (0-79 para Bitcoin header)
    domain: ByteDomain = field(default_factory=ByteDomain)
    incoming: List['ByteConstraint'] = field(default_factory=list)
    outgoing: List['ByteConstraint'] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"ByteNode(pos={self.position}, domain={self.domain})"


# ============================================================================
# RESTRICCIONES
# ============================================================================

class ByteConstraint(ABC):
    """Clase base para restricciones entre bytes"""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes: List[ByteNode] = []
    
    @abstractmethod
    def propagate(self) -> bool:
        """
        Propaga la restricción, reduciendo dominios.
        Retorna True si algún dominio cambió.
        """
        pass
    
    @abstractmethod
    def is_satisfied(self) -> bool:
        """Retorna True si la restricción está satisfecha"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class FixedValueConstraint(ByteConstraint):
    """Restricción: byte debe tener un valor fijo"""
    
    def __init__(self, node: ByteNode, value: int):
        super().__init__(f"byte[{node.position}] = {value:#04x}")
        self.node = node
        self.value = value
        self.nodes = [node]
    
    def propagate(self) -> bool:
        """Restringe el dominio a un solo valor"""
        return self.node.domain.restrict({self.value})
    
    def is_satisfied(self) -> bool:
        return self.node.domain.is_singleton() and \
               self.node.domain.get_singleton_value() == self.value


class RangeConstraint(ByteConstraint):
    """Restricción: byte debe estar en un rango"""
    
    def __init__(self, node: ByteNode, min_val: int, max_val: int):
        super().__init__(f"byte[{node.position}] ∈ [{min_val:#04x}, {max_val:#04x}]")
        self.node = node
        self.min_val = min_val
        self.max_val = max_val
        self.nodes = [node]
        self.allowed_values = set(range(min_val, max_val + 1))
    
    def propagate(self) -> bool:
        """Restringe el dominio al rango"""
        return self.node.domain.restrict(self.allowed_values)
    
    def is_satisfied(self) -> bool:
        return all(v in self.allowed_values for v in self.node.domain)


class XORConstraint(ByteConstraint):
    """Restricción: a ⊕ b = c"""
    
    def __init__(self, node_a: ByteNode, node_b: ByteNode, node_c: ByteNode):
        super().__init__(f"byte[{node_a.position}] ⊕ byte[{node_b.position}] = byte[{node_c.position}]")
        self.node_a = node_a
        self.node_b = node_b
        self.node_c = node_c
        self.nodes = [node_a, node_b, node_c]
    
    def propagate(self) -> bool:
        """Propaga restricción XOR"""
        changed = False
        
        # a ⊕ b = c  →  a = b ⊕ c, b = a ⊕ c
        
        # Reducir dominio de c
        new_c = set()
        for a in self.node_a.domain:
            for b in self.node_b.domain:
                new_c.add(a ^ b)
        if self.node_c.domain.restrict(new_c):
            changed = True
        
        # Reducir dominio de a
        new_a = set()
        for b in self.node_b.domain:
            for c in self.node_c.domain:
                new_a.add(b ^ c)
        if self.node_a.domain.restrict(new_a):
            changed = True
        
        # Reducir dominio de b
        new_b = set()
        for a in self.node_a.domain:
            for c in self.node_c.domain:
                new_b.add(a ^ c)
        if self.node_b.domain.restrict(new_b):
            changed = True
        
        return changed
    
    def is_satisfied(self) -> bool:
        """Verifica si la restricción está satisfecha"""
        for a in self.node_a.domain:
            for b in self.node_b.domain:
                c = a ^ b
                if c in self.node_c.domain:
                    return True
        return False


class ADDConstraint(ByteConstraint):
    """Restricción: (a + b + carry_in) mod 256 = result, con carry_out"""
    
    def __init__(self, node_a: ByteNode, node_b: ByteNode, 
                 node_result: ByteNode, 
                 node_carry_in: ByteNode, 
                 node_carry_out: ByteNode):
        super().__init__(f"byte[{node_a.position}] + byte[{node_b.position}] + carry")
        self.node_a = node_a
        self.node_b = node_b
        self.node_result = node_result
        self.node_carry_in = node_carry_in
        self.node_carry_out = node_carry_out
        self.nodes = [node_a, node_b, node_result, node_carry_in, node_carry_out]
    
    def propagate(self) -> bool:
        """Propaga restricción ADD con carry"""
        changed = False
        
        # Calcular valores válidos para result y carry_out
        new_result = set()
        new_carry_out = set()
        
        for a in self.node_a.domain:
            for b in self.node_b.domain:
                for c_in in self.node_carry_in.domain:
                    sum_val = a + b + c_in
                    r = sum_val & 0xFF
                    c_out = (sum_val >> 8) & 0x01
                    
                    if r in self.node_result.domain:
                        new_result.add(r)
                        new_carry_out.add(c_out)
        
        if self.node_result.domain.restrict(new_result):
            changed = True
        
        if self.node_carry_out.domain.restrict(new_carry_out):
            changed = True
        
        # Propagación inversa: reducir dominios de a, b, carry_in
        new_a = set()
        new_b = set()
        new_carry_in = set()
        
        for a in self.node_a.domain:
            for b in self.node_b.domain:
                for c_in in self.node_carry_in.domain:
                    sum_val = a + b + c_in
                    r = sum_val & 0xFF
                    c_out = (sum_val >> 8) & 0x01
                    
                    if r in self.node_result.domain and c_out in self.node_carry_out.domain:
                        new_a.add(a)
                        new_b.add(b)
                        new_carry_in.add(c_in)
        
        if self.node_a.domain.restrict(new_a):
            changed = True
        if self.node_b.domain.restrict(new_b):
            changed = True
        if self.node_carry_in.domain.restrict(new_carry_in):
            changed = True
        
        return changed
    
    def is_satisfied(self) -> bool:
        """Verifica si la restricción está satisfecha"""
        for a in self.node_a.domain:
            for b in self.node_b.domain:
                for c_in in self.node_carry_in.domain:
                    sum_val = a + b + c_in
                    r = sum_val & 0xFF
                    c_out = (sum_val >> 8) & 0x01
                    
                    if r in self.node_result.domain and c_out in self.node_carry_out.domain:
                        return True
        return False


# ============================================================================
# GRAFO DE RESTRICCIONES
# ============================================================================

class ByteConstraintGraph:
    """Grafo de restricciones a nivel byte con propagación AC-3"""
    
    def __init__(self, num_bytes: int = 80):
        """
        Inicializa el grafo con num_bytes nodos.
        Por defecto: 80 bytes (Bitcoin header)
        """
        self.num_bytes = num_bytes
        self.nodes: Dict[int, ByteNode] = {}
        self.constraints: List[ByteConstraint] = []
        
        # Crear nodos para cada byte
        for i in range(num_bytes):
            self.nodes[i] = ByteNode(position=i)
        
        # Nodos especiales para carries
        self.carry_nodes: Dict[str, ByteNode] = {}
    
    def add_constraint(self, constraint: ByteConstraint):
        """Añade una restricción al grafo"""
        self.constraints.append(constraint)
        
        # Actualizar incoming/outgoing de los nodos
        for node in constraint.nodes:
            if constraint not in node.outgoing:
                node.outgoing.append(constraint)
            if constraint not in node.incoming:
                node.incoming.append(constraint)
    
    def get_or_create_carry_node(self, name: str, initial_domain: Optional[Set[int]] = None) -> ByteNode:
        """Obtiene o crea un nodo de carry"""
        if name not in self.carry_nodes:
            if initial_domain is None:
                initial_domain = {0, 1}  # Carry es 0 o 1
            node = ByteNode(position=-1, domain=ByteDomain(initial_domain))
            self.carry_nodes[name] = node
        return self.carry_nodes[name]
    
    def propagate_ac3(self, max_iterations: int = 1000) -> bool:
        """
        Propagación AC-3 (Arc Consistency 3).
        Retorna True si hay solución, False si inconsistencia.
        """
        queue = list(self.constraints)
        iteration = 0
        
        while queue and iteration < max_iterations:
            iteration += 1
            constraint = queue.pop(0)
            
            # Propagar restricción
            if constraint.propagate():
                # Si algún dominio cambió, verificar si está vacío
                for node in constraint.nodes:
                    if node.domain.is_empty():
                        return False  # Inconsistencia detectada
                
                # Re-encolar restricciones afectadas
                for node in constraint.nodes:
                    for affected_constraint in node.incoming:
                        if affected_constraint != constraint and affected_constraint not in queue:
                            queue.append(affected_constraint)
        
        # Si llegamos aquí, hay solución (o max_iterations alcanzado)
        return True
    
    def get_space_size(self, byte_positions: List[int]) -> int:
        """Calcula el tamaño del espacio de búsqueda para las posiciones dadas"""
        space_size = 1
        for pos in byte_positions:
            if pos in self.nodes:
                space_size *= len(self.nodes[pos].domain)
        return space_size
    
    def get_reduction_factor(self, byte_positions: List[int]) -> float:
        """Calcula el factor de reducción vs espacio completo"""
        num_bytes = len(byte_positions)
        full_space = 256 ** num_bytes
        reduced_space = self.get_space_size(byte_positions)
        
        if reduced_space == 0:
            return float('inf')
        
        return full_space / reduced_space
    
    def is_value_in_space(self, byte_positions: List[int], values: List[int]) -> bool:
        """Verifica si un conjunto de valores está en el espacio reducido"""
        assert len(byte_positions) == len(values), "Longitudes no coinciden"
        
        for pos, val in zip(byte_positions, values):
            if pos in self.nodes:
                if val not in self.nodes[pos].domain:
                    return False
        
        return True
    
    def __repr__(self) -> str:
        return f"ByteConstraintGraph({self.num_bytes} bytes, {len(self.constraints)} constraints)"
