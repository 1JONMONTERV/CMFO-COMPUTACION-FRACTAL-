# CMFO-FRACTAL-ALGEBRA 1.1
## Especificación Matemática Cerrada y Ampliada para Universo Medible de 1024 bits

### 0) Notación y dominio
*   **Universo binario**:
    $$ \mathcal{U}_{1024} := \{0,1\}^{1024} $$
*   **Descomposición en niblex (4 bits)**:
    $$ x \in \mathcal{U}_{1024} \iff x=(n_0,\dots,n_{255}),\quad n_i\in\{0,1\}^4 $$
*   **Niveles multiescala (9 niveles)**:
    $$ \ell=0..8,\quad N_\ell = 256/2^\ell $$
    donde $N_\ell$ es el número de “unidades” (niblex equivalentes) en el nivel $\ell$.

---

### 1) Estructuras canónicas: clase y espejo

#### 1.1 Espejo niblex (involución base)
Para $n \in \{0,1\}^4$:
$$ M_4(n)=\neg n \implies M_4(M_4(n))=n $$

#### 1.2 Canonización por niblex (forma normal local)
Orden lexicográfico $\le_{\text{lex}}$. Defina:
$$ C(n)=\min_{\text{lex}}\{n,M_4(n)\} $$
$$ b(n)=\mathbf{1}_{\{n>_{\text{lex}}M_4(n)\}} $$
Reconstrucción exacta:
$$ n = M_4^{\,b(n)}(C(n)) $$
donde $M_4^{0}=\text{Id}, M_4^{1}=M_4$.

#### 1.3 Clase canónica (octagonal) y bandera espejo
Defina una proyección de clase:
$$ \kappa:\{0,1\}^4\to\mathbb{Z}_8 $$
que asigna a cada $C(n)$ una clase $c=\kappa(C(n))\in\{0,\dots,7\}$.
Entonces cada niblex queda representado por:
$$ \nu(n)=(c,b)\in\mathbb{Z}_8\times\mathbb{Z}_2 $$

Resultado: el “átomo” CMFO no es el bit; es el par $(c,b)$ (octágono + espejo), y esto escala naturalmente.

---

### 2) Álgebra binaria CMFO sobre 1024 bits

#### 2.1 Operadores booleanos base (cierre universal)
Para $x,y \in \mathcal{U}_{1024}$:
$x\oplus y,\ \ x\wedge y,\ \ \neg x,\ \ x\vee y,\ \ \text{XNOR},\ \text{NAND},\ \text{NOR}$
Universalidad: $\{\oplus,\wedge,\neg\}$ genera cualquier función booleana.

#### 2.2 Operadores primarios CMFO (nuevos, estructurales)

**(A) Canonizador global**
$$ \mathcal{C}(x)=(C(n_0),\dots,C(n_{255})) $$
$$ \mathcal{B}(x)=(b(n_0),\dots,b(n_{255}))\in\{0,1\}^{256} $$
Reconstrucción componente a componente:
$$ x = \mathcal{M}^{\mathcal{B}(x)}(\mathcal{C}(x)) $$
donde $\mathcal{M}^{\mathbf{b}}$ aplica $M_4$ en cada niblex donde $\mathbf{b}_i=1$.

**(B) Operador “niblex-lift” (subir de bits a estados)**
$$ \mathcal{N}(x) = (\nu(n_0),\dots,\nu(n_{255}))\in(\mathbb{Z}_8\times\mathbb{Z}_2)^{256} $$

**(C) Operadores de re-empaquetado por escala**
Para pasar de nivel $\ell$ a $\ell+1$ (agrupación por pares):
$$ \mathrm{Pack}_\ell:\ (u_0,\dots,u_{N_\ell-1})\mapsto (U_0,\dots,U_{N_{\ell+1}-1}) $$
donde cada $U_j$ depende solo de $(u_{2j},u_{2j+1})$.
El desempaquetado se define en la versión reversible (sección 3.2).

**(D) Operador de permutación estructural**
Sea $\pi$ una permutación de bloques (por ejemplo, de 256 niblex o de bloques en nivel $\ell$):
$$ \mathrm{Perm}_\pi(x) = (n_{\pi(0)},\dots,n_{\pi(255)}) $$
Esto define un universo “de permutaciones” sin ambigüedad.

**(E) Operadores geométricos discretos (útiles en pipelines)**
*   Rotación circular por niblex: $\mathrm{Rot}_k$
*   Corrimiento por niblex: $\mathrm{Sh}_k$
*   Inversión por bloque: $\mathrm{Rev}$
*   Intercalado Morton/Z-order por bloque: $\mathrm{Morton}$

---

### 3) Renormalización multiescala: dos versiones (resumen y reversible)

#### 3.1 Renormalización-resumen $\mathcal{R}^{\text{sum}}_\ell$ (para medición, búsqueda, índices)
Trabaja sobre unidades (niblex o estados por nivel). Para cada par:
$$ u^{(\ell+1)}_j = \rho_\ell(u^{(\ell)}_{2j},u^{(\ell)}_{2j+1}) $$
Condición CMFO obligatoria (conmutación con espejo):
$$ \rho_\ell(M(u),M(v))=M(\rho_\ell(u,v)) $$
Esto se implementa como microtabla restringida (16×16 a nivel niblex, o 16×16 sobre $(c,b)$ si trabaja en estados).

#### 3.2 Renormalización-reversible $\mathcal{R}^{\text{rev}}_\ell$ (compresión sin pérdida)
Defina resumen y residual:
$$ r^{(\ell+1)}_j=\rho_\ell(u_{2j},u_{2j+1}),\qquad e^{(\ell+1)}_j=\eta_\ell(u_{2j},u_{2j+1}) $$
con reconstrucción exacta:
$$ (u_{2j},u_{2j+1})=\mathrm{Expand}_\ell(r_j,e_j) $$
y conmutación espejo en ambos canales:
$$ \rho_\ell(Mu,Mv)=M\rho_\ell(u,v),\qquad \eta_\ell(Mu,Mv)=\eta_\ell(u,v) $$

---

### 4) Segmentación determinista y estado mínimo por segmento

#### 4.1 Invariante vectorial local (ventana)
Para ventana de $W$ niblex, defina:
$$ U(i)=(H_c(i),\ p_b(i),\ E_t(i),\ \Delta(i))\in\mathbb{R}^4 $$
donde:
*   $H_c$: entropía del histograma de clases $c\in\mathbb{Z}_8$
*   $p_b$: proporción/paridad de espejos
*   $E_t$: energía de transición (cambios locales)
*   $\Delta$: variación espectral local (DFT8 sobre histograma)

Corte:
$$ \text{Corte en } i \iff \|U(i+1)-U(i)\|_2 \ge \tau $$

#### 4.2 Estado mínimo por segmento (cerrado, combinable O(1))
Para un segmento $S$:
$$ \mathrm{State}(S)=(c^*,\ b^*,\ L,\ E,\ \sigma) $$
*   $c^*$: clase dominante (moda)
*   $b^*$: paridad (o ratio cuantizado)
*   $L$: longitud (en niblex)
*   $E$: energía normalizada
*   $\sigma$: fase espectral principal (armónico dominante)

Composición de estados (monoidal, O(1)):
$$ \mathrm{State}(S_1\|S_2) = \Big( \mathrm{merge}_c,\ b_1\oplus b_2,\ L_1+L_2,\ \frac{L_1E_1+L_2E_2}{L_1+L_2},\ \sigma_1\oplus_8 \sigma_2 \Big) $$

---

### 5) Tabla procedural como insumo del álgebra

#### 5.1 Definición
$$ T:\mathcal{I}\to\mathcal{U}_{1024},\qquad T(i)=G(i) $$
donde $G$ se construye con operadores CMFO.

#### 5.2 Universo de permutaciones (formal, sin ambigüedad)
Si el objetivo es “todas las permutaciones” de 256 niblex:
$$ \mathcal{I}=\{0,\dots,256!-1\},\quad \pi=\mathrm{Unrank}(i)\in S_{256} $$
$$ T(i)=\mathrm{Perm}_{\pi}(\mathrm{Seed}) $$

#### 5.3 Tabla como acelerador
Catálogo procedural de microtablas $\rho_\ell,\eta_\ell$, patrones de pack, índices, y bases espectrales.

---

### 6) Multi-métricas CMFO (multumétricas)

#### 6.1 Métrica Hamming (base)
$$ d_H(x,y)=\sum_{k=1}^{1024}\mathbf{1}_{\{x_k\ne y_k\}} $$

#### 6.2 Métrica por niblex (más estructural)
$$ d_N(x,y)=\sum_{i=0}^{255} d_4(n_i,m_i) $$

#### 6.3 Métrica multiescala ponderada (núcleo CMFO)
$$ d_{\text{MS}}(x,y)=\sum_{\ell=0}^{8} w_\ell\ d_\ell(x^{(\ell)},y^{(\ell)}) $$
con $w_\ell=\varphi^{-\ell}$.

#### 6.4 Métrica espectral octagonal + espejo
$$ d_{\text{spec}}(x,y)=\sum_{\ell=0}^8 \alpha_\ell \sum_{m=0}^7 \left|\hat{h}^{(\ell)}_m(x)-\hat{h}^{(\ell)}_m(y)\right| $$

#### 6.5 Métrica por segmentación (estructura de cortes)
$$ d_{\text{seg}}(x,y)=\mathrm{DTW}\big(\mathrm{StateSeq}(x),\ \mathrm{StateSeq}(y)\big) $$

#### 6.6 Kernel de similitud
$$ K(x,y)=\exp\left(-\gamma\, d(x,y)\right) $$

---

### 7) Mapa de medición $\Phi$ mínimo (90D exactas)

#### 7.1 $\Phi_{90}$ (cerrado, exacto)
Por nivel $\ell=0..8$, defina 10 invariantes:
1.  Entropía de clases $I_1$
2.  Sesgo espejo $I_2$
3.  Energía transición $I_3$
4.  Autocorrelación 1 $I_4$
5.  Autocorrelación 2 $I_5$
6.  Amplitud espectral dominante $I_6$
7.  Fase espectral dominante $I_7$
8.  Complejidad LZ local $I_8$
9.  Entropía de permutación $I_9$
10. Asimetría de segmentos $I_{10}$

$$ \Phi_{90}(x)=(I_k^{(\ell)}(x))_{\ell=0..8,\ k=1..10} $$

#### 7.2 Isometría de espejo
Existe $P$ tal que $\Phi(M(x))=P\,\Phi(x)$.

---

### 8) Ecuaciones finales de coherencia

#### 8.1 Conmutación espejo de operadores primarios
$$ \mathcal{C}(M(x))=\mathcal{C}(x),\qquad \mathcal{B}(M(x))=\mathbf{1}\oplus \mathcal{B}(x) $$

#### 8.2 Compatibilidad de segmentación (estabilidad)
$$ \mathrm{Seg}(M(x))=\mathrm{Seg}(x) $$
$$ \mathrm{State}(M(S))=\Pi(\mathrm{State}(S)) $$

#### 8.3 Renormalización coherente multiescala
$$ (x^{(\ell)},e^{(\ell+1)})=\mathcal{R}^{\text{rev}}_{\ell+1}(x^{(\ell-1)}) $$

#### 8.4 Acción (costo) multiescala natural
$$ \mathcal{A}(x)=\sum_{\ell=0}^8 w_\ell\,\mathcal{E}_\ell(x^{(\ell)}) + \sum_{\ell=0}^7 \lambda_\ell\,\mathcal{I}_{\ell\leftrightarrow \ell+1}(x) $$

---

### 9) Suite CMFO ampliada

#### 9.1 Motor de aceleración booleana
$$ F(x,y)=\mathrm{Rebuild}\Big( T_F\big(\mathcal{N}(x),\mathcal{N}(y)\big)\Big) $$

#### 9.2 Índice multiescala
$$ \mathrm{Key}(x)=\mathrm{Hash}\big(\mathrm{Quant}(\Phi_{90}(x))\big) $$

#### 9.3 Deduplicación / near-duplicate
$$ \text{Duplicado si } d_{\text{MS}}<\epsilon_1\ \wedge\ d_{\text{seg}}<\epsilon_2\ \wedge\ d_{\text{spec}}<\epsilon_3 $$

#### 9.4 Compresión estructural
$$ \mathrm{Compress}(x)=\Big(\mathrm{Seg}(x),\ \{(r_i,e_i)\}_i\Big) $$

#### 9.5 Detección de anomalías
$$ A(x)=1-\max_{y\in\mathcal{D}}K(x,y) $$

---

### 10) Validación PASS/FAIL
1.  Involución espejo: $M(M(x))=x$ (100%)
2.  Canonización reversible: reconstrucción exacta (100%)
3.  Conmutación espejo (renormalización): $\rho_\ell(Mu,Mv)=M\rho_\ell(u,v)$ (100%)
4.  $\Phi$ isométrica: Error $< 10^{-6}$
5.  Reversible: Decompress(Compress(x)) = x (100%)
