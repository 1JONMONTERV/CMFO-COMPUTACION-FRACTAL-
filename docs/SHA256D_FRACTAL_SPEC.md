# SHA-256d Fractal: Especificación Formal

**Fecha:** 17 de Diciembre, 2025  
**Objetivo:** Implementar SHA-256d con salida idéntica a Bitcoin usando arquitectura fractal reversible  
**Principio:** Misma función, otra representación computacional

---

## 1. Declaración de Equivalencia

### 1.1 Garantía de Identidad
```
Para todo header H de 80 bytes:
  FractalSHA256d(H) = SHA256d(H)  [bit-exacto]
```

### 1.2 Diferencias Arquitectónicas

| Aspecto | SHA-256d Estándar | SHA-256d Fractal |
|---------|-------------------|------------------|
| **Entrada** | 80 bytes (header) | 80 bytes (header) |
| **Salida** | 32 bytes (digest) | 32 bytes (digest) |
| **Proceso** | Irreversible, caja negra | Reversible, trazable |
| **Estado interno** | 8 words (256 bits) | 1024 posiciones A/B |
| **Rondas** | 64 | 64 (fractal equivalentes) |
| **Verificación** | N/A | Traza completa por ronda |

---

## 2. Arquitectura del Estado Fractal

### 2.1 Layout de 1024 Posiciones

```
Estado Fractal = 1024 celdas [A/B]

Distribución:
  [0-511]   : Estado de trabajo (512 bits)
              - [0-255]   : Estado H (8 words × 32 bits)
              - [256-511] : Estado W (message schedule)
  
  [512-1023]: Ancillas y trazabilidad (512 bits)
              - [512-767] : Ancillas para operaciones reversibles
              - [768-1023]: Etiquetas de origen/propagación
```

### 2.2 Representación de Words

Cada word de 32 bits se distribuye en 32 celdas consecutivas:
```
Word[i] = {celda[32i], celda[32i+1], ..., celda[32i+31]}
```

### 2.3 Etiquetas de Trazabilidad

Cada celda tiene:
```python
class FractalCell:
    value: bool        # A (True) o B (False)
    origin: BitSet     # Qué bits de entrada contribuyeron
    round: int         # En qué ronda se modificó
    operation: str     # Qué operación la generó
```

---

## 3. Mapeo del Circuito Booleano SHA-256

### 3.1 Operaciones Primitivas

#### 3.1.1 XOR Reversible (Feynman Gate)
```
Entrada: a, b, ancilla=0
Salida:  a, a⊕b, ancilla

Fractal:
  apply_xor_fractal(pos_a, pos_b, pos_out):
    out.value = a.value XOR b.value
    out.origin = a.origin ∪ b.origin
    out.operation = "XOR"
```

#### 3.1.2 AND Reversible (Toffoli Gate)
```
Entrada: a, b, c
Salida:  a, b, c⊕(a∧b)

Fractal:
  apply_and_fractal(pos_a, pos_b, pos_c):
    c.value = c.value XOR (a.value AND b.value)
    c.origin = a.origin ∪ b.origin ∪ c.origin
    c.operation = "TOFFOLI"
```

#### 3.1.3 NOT Reversible (Pauli-X)
```
Entrada: a
Salida:  ¬a

Fractal:
  apply_not_fractal(pos_a):
    a.value = NOT a.value
    a.operation = "NOT"
```

#### 3.1.4 ROTR (Rotate Right) Reversible
```
ROTR_n(x) en fractal:
  Para word de 32 bits en posiciones [i, i+31]:
    Permutación cíclica de celdas
    Preserva trazas de origen
```

### 3.2 Funciones SHA-256 en Dominio Fractal

#### 3.2.1 Ch (Choose)
```
Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z)

Fractal:
  1. temp1 = AND_fractal(x, y)
  2. temp2 = NOT_fractal(x)
  3. temp3 = AND_fractal(temp2, z)
  4. result = XOR_fractal(temp1, temp3)
```

#### 3.2.2 Maj (Majority)
```
Maj(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)

Fractal:
  1. temp1 = AND_fractal(x, y)
  2. temp2 = AND_fractal(x, z)
  3. temp3 = AND_fractal(y, z)
  4. temp4 = XOR_fractal(temp1, temp2)
  5. result = XOR_fractal(temp4, temp3)
```

#### 3.2.3 Σ₀ (Sigma 0)
```
Σ₀(x) = ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)

Fractal:
  1. temp1 = ROTR_fractal(x, 2)
  2. temp2 = ROTR_fractal(x, 13)
  3. temp3 = ROTR_fractal(x, 22)
  4. temp4 = XOR_fractal(temp1, temp2)
  5. result = XOR_fractal(temp4, temp3)
```

#### 3.2.4 Σ₁ (Sigma 1)
```
Σ₁(x) = ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)

Fractal: [similar a Σ₀]
```

#### 3.2.5 σ₀ (sigma 0 - message schedule)
```
σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)

Fractal: [similar, SHR usa shift sin wrap]
```

#### 3.2.6 σ₁ (sigma 1 - message schedule)
```
σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)

Fractal: [similar]
```

### 3.3 Adición Modular Reversible

```
ADD_mod_2^32(a, b) reversible:

Método: Ripple-carry con ancillas
  Para cada bit i (0 a 31):
    1. sum[i] = a[i] ⊕ b[i] ⊕ carry[i-1]
    2. carry[i] = MAJ(a[i], b[i], carry[i-1])
  
  Ancillas: carry[0..31] se descomputan al final
```

---

## 4. Algoritmo SHA-256d Fractal

### 4.1 Inicialización

```python
def init_fractal_state():
    state = FractalState(1024)
    
    # H inicial (constantes SHA-256)
    H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, ...]
    
    # Cargar en posiciones [0-255]
    for i, h_val in enumerate(H):
        load_word_to_fractal(state, i*32, h_val)
    
    # Ancillas en cero
    for i in range(512, 1024):
        state[i].value = False
        state[i].origin = BitSet()
    
    return state
```

### 4.2 Message Schedule (Expansión W)

```python
def expand_message_fractal(state, block):
    # W[0..15] = block (16 words de 32 bits)
    for t in range(16):
        load_word_to_fractal(state, 256 + t*32, block[t])
    
    # W[16..63] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]
    for t in range(16, 64):
        w_t_2  = get_word_from_fractal(state, 256 + (t-2)*32)
        w_t_7  = get_word_from_fractal(state, 256 + (t-7)*32)
        w_t_15 = get_word_from_fractal(state, 256 + (t-15)*32)
        w_t_16 = get_word_from_fractal(state, 256 + (t-16)*32)
        
        s0 = sigma0_fractal(state, w_t_15)
        s1 = sigma1_fractal(state, w_t_2)
        
        temp1 = add_mod_fractal(state, s1, w_t_7)
        temp2 = add_mod_fractal(state, s0, w_t_16)
        w_t = add_mod_fractal(state, temp1, temp2)
        
        store_word_to_fractal(state, 256 + t*32, w_t)
```

### 4.3 Ronda de Compresión Fractal

```python
def compression_round_fractal(state, t, K_t):
    # Leer estado actual
    a, b, c, d, e, f, g, h = read_working_vars(state)
    W_t = get_word_from_fractal(state, 256 + t*32)
    
    # T1 = h + Σ₁(e) + Ch(e,f,g) + K[t] + W[t]
    S1 = Sigma1_fractal(state, e)
    ch = Ch_fractal(state, e, f, g)
    
    temp1 = add_mod_fractal(state, h, S1)
    temp2 = add_mod_fractal(state, temp1, ch)
    temp3 = add_mod_fractal(state, temp2, K_t)
    T1 = add_mod_fractal(state, temp3, W_t)
    
    # T2 = Σ₀(a) + Maj(a,b,c)
    S0 = Sigma0_fractal(state, a)
    maj = Maj_fractal(state, a, b, c)
    T2 = add_mod_fractal(state, S0, maj)
    
    # Actualizar variables
    h_new = g
    g_new = f
    f_new = e
    e_new = add_mod_fractal(state, d, T1)
    d_new = c
    c_new = b
    b_new = a
    a_new = add_mod_fractal(state, T1, T2)
    
    # Escribir de vuelta
    write_working_vars(state, a_new, b_new, c_new, d_new, 
                              e_new, f_new, g_new, h_new)
```

### 4.4 SHA-256d Completo

```python
def sha256d_fractal(header_80_bytes):
    # Primera pasada SHA-256
    digest1 = sha256_fractal(header_80_bytes)
    
    # Segunda pasada SHA-256
    digest2 = sha256_fractal(digest1)
    
    return digest2

def sha256_fractal(message):
    state = init_fractal_state()
    
    # Padding (igual que SHA-256 estándar)
    padded = pad_message(message)
    
    # Procesar cada bloque de 512 bits
    for block in chunk_512(padded):
        # Expansión del mensaje
        expand_message_fractal(state, block)
        
        # 64 rondas de compresión
        for t in range(64):
            compression_round_fractal(state, t, K[t])
        
        # Actualizar H
        update_hash_values(state)
    
    # Descomputar ancillas
    decompute_ancillas(state)
    
    # Extraer digest (256 bits)
    digest = extract_digest(state)
    
    return digest
```

---

## 5. Trazabilidad y Verificación

### 5.1 Traza por Ronda

```python
def get_round_trace(state, round_num):
    trace = {
        'round': round_num,
        'modified_cells': [],
        'operations': []
    }
    
    for i, cell in enumerate(state.cells):
        if cell.round == round_num:
            trace['modified_cells'].append({
                'position': i,
                'value': cell.value,
                'origin': cell.origin,
                'operation': cell.operation
            })
    
    return trace
```

### 5.2 Verificación de Bit-Exactitud

```python
def verify_fractal_sha256d():
    # Headers reales de Bitcoin
    test_headers = load_bitcoin_headers(count=10000)
    
    for header in test_headers:
        # SHA-256d estándar (librería)
        expected = hashlib.sha256(
            hashlib.sha256(header).digest()
        ).digest()
        
        # SHA-256d fractal
        result = sha256d_fractal(header)
        
        # Verificar bit a bit
        assert result == expected, f"Mismatch on header {header.hex()}"
    
    print("✓ Bit-exactitud verificada en 10,000 headers")
```

### 5.3 Invariantes de Consistencia

```python
def verify_trace_consistency(state):
    # Verificar que cada celda tiene origen válido
    for cell in state.cells:
        assert cell.origin.is_valid()
        assert cell.round >= 0
    
    # Verificar que ancillas están descomputadas
    for i in range(512, 768):
        assert state[i].value == False
    
    print("✓ Traza consistente")
```

---

## 6. Ventajas de la Arquitectura Fractal

### 6.1 Lo que SÍ garantizamos

✓ **Bit-exactitud:** Salida idéntica a SHA-256d estándar  
✓ **Reversibilidad:** Proceso completamente reversible con ancillas  
✓ **Trazabilidad:** Propagación de cada bit visible por ronda  
✓ **Paralelización:** Operaciones fractales acelerables en GPU  
✓ **Verificabilidad:** Auditoría externa contra SHA real

### 6.2 Lo que NO prometemos (evita conflictos)

- No afirmamos "más rápido siempre"
- No afirmamos "sin costo computacional"
- No reemplazamos SHA-256 en general (solo implementación alternativa)

### 6.3 Casos de Uso

1. **Minería Bitcoin con trazabilidad completa**
2. **Análisis de propagación de bits en SHA-256**
3. **Educación: visualización de SHA-256 interno**
4. **Investigación: circuitos reversibles a escala**
5. **GPU/CUDA: paralelización masiva de rondas**

---

## 7. Próximos Pasos de Implementación

### Fase 1: Operadores Primitivos
- [ ] Implementar XOR/AND/NOT reversibles en Python
- [ ] Implementar ROTR/SHR fractales
- [ ] Implementar ADD mod 2^32 con ancillas

### Fase 2: Funciones SHA-256
- [ ] Ch, Maj, Σ₀, Σ₁, σ₀, σ₁ en dominio fractal
- [ ] Message schedule expansion
- [ ] Compression round

### Fase 3: SHA-256d Completo
- [ ] Integración de 64 rondas
- [ ] Padding y chunking
- [ ] Doble hash (SHA-256d)

### Fase 4: Verificación
- [ ] Test con headers Bitcoin reales
- [ ] Validación bit-exacta (10,000+ casos)
- [ ] Benchmarks de rendimiento

### Fase 5: Optimización GPU
- [ ] Port a C++/CUDA
- [ ] Paralelización de operaciones
- [ ] Integración con `cmfo_jit.dll`

---

**Estado:** ESPECIFICACIÓN COMPLETA ✓  
**Siguiente:** Implementación de operadores primitivos
