import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

function Feature({ title, description, icon }) {
  return (
    <div className={styles.featureCol}>
      <div className={styles.featureIcon}>{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function CodeExample({ title, language, code }) {
  return (
    <div className={styles.codeExample}>
      <h3>{title}</h3>
      <pre className={styles.codeBlock}>
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  );
}

export default function Home() {
  return (
    <Layout
      title="CMFO–UNIVERSE v∞"
      description="Motor de Computación Fractal Determinista"
    >
      {/* Hero Section */}
      <header className={styles.heroBanner}>
        <div className="container">
          <div className={styles.heroContent}>
            <div className={styles.heroText}>
              <h1 className={styles.title}>CMFO–UNIVERSE</h1>
              <p className={styles.subtitle}>
                Motor de Cálculo Fractal • Álgebra T⁷ • Física de Solitones
              </p>
              <p className={styles.tagline}>
                Computación determinista basada en geometría fractal φ (Phi)
              </p>
              <div className={styles.buttons}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/intro"
                >
                  📚 Documentación
                </Link>
                <Link
                  className="button button--secondary button--lg"
                  to="https://github.com/1jonmonterv/cmfo-universe"
                >
                  ⭐ GitHub
                </Link>
              </div>
            </div>
            <div className={styles.heroVisual}>
              <img src="/cmfo-universe/img/soliton_hero.png" alt="Soliton Visualization" className={styles.heroImage} />
            </div>
          </div>
        </div>
      </header>

      <main className={styles.mainSection}>
        {/* Features */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Capacidades de Nivel Máximo</h2>
            <div className={styles.featuresGrid}>
              <Feature
                icon="🌊"
                title="Física de Solitones"
                description="Simulación exacta de colisiones Kink-Antikink en el campo Sine-Gordon con preservación de carga topológica."
              />
              <Feature
                icon="🔗"
                title="Interoperabilidad Total"
                description="SDKs nativos para Python, Node.js y C++. Integración fluida vía FFI y Bindings."
              />
              <Feature
                icon="⚡"
                title="CUDA Phase 2"
                description="Kernels de GPU con acople N-Cuerpos para simular emergencia fractal masiva."
              />
              <Feature
                icon="🧪"
                title="Tests de Estrés"
                description="100,000+ operaciones tensoriales sin degradación numérica."
              />
              <Feature
                icon="📦"
                title="Empaquetado Profesional"
                description="Instalable vía pip con estructura moderna de paquete Python."
              />
              <Feature
                icon="🎯"
                title="φ-Logic"
                description="Lógica continua basada en la proporción áurea con operaciones reversibles."
              />
            </div>
          </div>
        </section>

        {/* Code Examples */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Código en Acción</h2>
            <p className={styles.sectionSubtitle}>
              Ejemplos de uso en múltiples lenguajes
            </p>

            <div className={styles.codeExamples}>
              <CodeExample
                title="🐍 Python"
                language="python"
                code={`from cmfo import CMFOCore

core = CMFOCore()
print(f"Phi: {core.phi()}")

a = [1.0] * 7
b = [2.0] * 7
result = core.tensor7(a, b)`}
              />

              <CodeExample
                title="⚙️ C++"
                language="cpp"
                code={`#include "cmfo_core.h"

cmfo::Vector7 a = {1,1,1,1,1,1,1};
cmfo::Vector7 b = {2,2,2,2,2,2,2};
cmfo::Vector7 c = a ^ b;
c.print();`}
              />

              <CodeExample
                title="🟢 Node.js"
                language="javascript"
                code={`const cmfo = require('./cmfo');

console.log('Phi:', cmfo.phi());

const a = [1,1,1,1,1,1,1];
const b = [2,2,2,2,2,2,2];
const result = cmfo.tensor7(a, b);`}
              />
            </div>
          </div>
        </section>

        {/* Benchmarks */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Benchmarks de Estrés</h2>
            <div className={styles.benchmarkGrid}>
              <div className={styles.benchmarkCard}>
                <div className={styles.benchmarkNumber}>100,000+</div>
                <div className={styles.benchmarkLabel}>Operaciones Tensoriales</div>
                <div className={styles.benchmarkDesc}>Sin degradación numérica</div>
              </div>
              <div className={styles.benchmarkCard}>
                <div className={styles.benchmarkNumber}>1,000+</div>
                <div className={styles.benchmarkLabel}>Matrices Invertidas</div>
                <div className={styles.benchmarkDesc}>Detección robusta de singularidades</div>
              </div>
              <div className={styles.benchmarkCard}>
                <div className={styles.benchmarkNumber}>3</div>
                <div className={styles.benchmarkLabel}>Lenguajes Soportados</div>
                <div className={styles.benchmarkDesc}>Python, C++, Node.js</div>
              </div>
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Arquitectura Fractal 7D</h2>
            <p className={styles.architectureText}>
              Todo el conocimiento se deriva desde una estructura geométrica exacta basada en la <strong>proporción áurea φ</strong>.
              El CMFO reemplaza la lógica booleana con <strong>φ-Logic</strong>, permitiendo estados continuos y operaciones reversibles.
            </p>
            <div className={styles.architectureFeatures}>
              <div className={styles.archFeature}>
                <strong>Núcleo C</strong>
                <p>Álgebra vectorial y matricial exacta en dimensión 7</p>
              </div>
              <div className={styles.archFeature}>
                <strong>CUDA Kernels</strong>
                <p>Aceleración GPU con SM86 optimizado</p>
              </div>
              <div className={styles.archFeature}>
                <strong>Precisión Exacta</strong>
                <p>Determinante e inversa 7×7 sin errores</p>
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className={styles.ctaSection}>
          <div className="container">
            <h2>Únete a la Revolución Fractal</h2>
            <p>Explora la documentación completa y comienza a experimentar con CMFO</p>
            <div className={styles.ctaButtons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro"
              >
                📖 Leer Teoría Completa
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="https://github.com/1jonmonterv/cmfo-universe"
              >
                💻 Ver Código Fuente
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
