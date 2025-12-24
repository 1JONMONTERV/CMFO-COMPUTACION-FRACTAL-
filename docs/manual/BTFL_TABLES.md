
# Tablas de Verdad: Lógica Ternaria Fractal Balanceada (BTFL)

**Referencia Técnica**: CMFO-BTFL-1.0
**Contexto**: Definición formal de operadores para sistemas de 3 estados `{-1, 0, +1}`.

## 1. Definiciones Fundamentales

El sistema utiliza tres estados base, interpretados geométricamente:
- `+1`: **Verdad / Presencia / Constructivo**
- ` 0`: **Superposición / Fase Neutra / Indeterminado**
- `-1`: **Falsedad / Ausencia / Destructivo / Inverso**

Los operadores se definen mediante ecuaciones continuas que permiten interpolación (fractalidad), pero aquí se presentan los valores discretos fundamentales.

### Fórmulas Generatrices
- **XOR Fractal**: `clamp(x + y - xy·|x+y|/2)` (Saturado en [-1, 1])
- **AND Fractal**: `(x·y)·(1 - |x-y|/2)`
- **OR Fractal**: `(x+y-xy)·(1 - |x+y-xy|/2)`

---

## 2. Tablas de Verdad

### 2.1 Tabla AND (Intersección Coherente)
*Nota: Se comporta como una coincidencia de signo estricta.*

| A   | B   | Resultado (AND) | Interpretación |
|:---:|:---:|:---------------:|:---------------|
| -1  | -1  | **+1**          | Coincidencia Negativa -> Verdad (Doble negación) |
| -1  |  0  |  0              | Absorción por Neutro |
| -1  | +1  |  0              | Cancelación de Opuestos |
|  0  | -1  |  0              | Absorción |
|  0  |  0  |  0              | Neutro |
|  0  | +1  |  0              | Absorción |
| +1  | -1  |  0              | Cancelación |
| +1  |  0  |  0              | Absorción |
| +1  | +1  | **+1**          | Coincidencia Positiva -> Verdad |

> **Observación**: A diferencia del booleano `False AND False = False`, en BTFL `(-1) AND (-1) = +1`. Esto indica "Coherencia de Fase": dos señales negativas resuenan constructivamente.

### 2.2 Tabla XOR (Diferencia Sincronizada)

| A   | B   | Resultado (XOR) | Interpretación |
|:---:|:---:|:---------------:|:---------------|
| -1  | -1  | **-1**          | Igualdad Negativa -> Falso (Sin diferencia) |
| -1  |  0  | **-1**          | (?) Estado especial de transición |
| -1  | +1  |  0              | Máxima diferencia -> Neutralizado por simetría |
|  0  | -1  | **-1**          | Transición |
|  0  |  0  |  0              | Igualdad Neutra |
|  0  | +1  | **+1**          | (?) |
| +1  | -1  |  0              | Máxima diferencia -> Neutralizado |
| +1  |  0  | **+1**          | Transición |
| +1  | +1  | **+1**          | **Divergencia Fractal**: "Constructivo + Constructivo = Constructivo" |

> **Observación Crítica**: XOR(1,1) = 1 rompe la paridad clásica (usualmente 0). En CMFO, dos verdades se refuerzan en lugar de anularse, a menos que se opere en módulo. Esta tabla refleja la *dinámica de fluidos* de la fórmula, no un anillo de polinomios estándar.

### 2.3 Tabla OR (Unión Amortiguada)

| A   | B   | Resultado (OR)  | Interpretación |
|:---:|:---:|:---------------:|:---------------|
| -1  | -1  | **+1.5** (Saturado a +1) | Resonancia fuerte (falla de saturación en fórmula cruda) |
| -1  |  0  | -0.5            | Amortiguación |
| -1  | +1  | +0.5            | Equilibrio positivo |
|  0  | -1  | -0.5            | Amortiguación |
|  0  |  0  |  0              | Nada |
|  0  | +1  | +0.5            | Señal débil |
| +1  | -1  | +0.5            | Equilibrio |
| +1  |  0  | +0.5            | Señal débil |
| +1  | +1  | +0.5            | **Saturación**: La unión de dos máximos se auto-limita (0.5) |

> **Nota de Implementación**: La fórmula de OR `(x+y-xy)...` produce valores fuera de {-1,0,1} exactos. Requiere una etapa de *Colapso* o *Clamp* final para discretizar si se usa en lógica digital pura.

## 3. Conclusión de Integración

El sistema BTFL **no es isomórfico** al álgebra booleana clásica. Es un sistema de **Resonancia de Onda**:
- `AND` detecta **Coherencia** (Fase igual).
- `XOR` detecta **Energía Neta** (Sumatoria con pérdidas).
- `OR` detecta **Presencia** con efectos de saturación no lineal.

Para la integración con "Niblex de 11 bits", se recomienda usar los valores continuos para el procesamiento (NPU) y colapsar solo al final (Lectura).
