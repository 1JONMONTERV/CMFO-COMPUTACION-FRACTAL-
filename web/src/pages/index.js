import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';
import FractalChart from '@site/src/components/FractalChart';
import clsx from 'clsx'; // Need clsx for hero class

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
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="CMFO: Computación Fractal"
      description="La Plataforma de Computación Determinista Unificada"
    >
      {/* Hero Section */}
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <div className={styles.heroContent}>
            <div className={styles.heroText}>
              <h1 className={styles.title}>{siteConfig.title}</h1>
              <p className={styles.subtitle}>{siteConfig.tagline}</p>

              {/* VIZ SECTION */}
              <div style={{ marginTop: '20px', marginBottom: '20px' }}>
                <FractalChart />
              </div>

              <div className={styles.buttons}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/intro"
                >
                  🚀 Empezar
                </Link>
                <Link
                  className="button button--secondary button--lg"
                  to="https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-"
                >
                  ⭐ GitHub
                </Link>
              </div>
            </div>
            <div className={styles.heroVisual}>
              {/* Using existing asset or placeholder */}
              <div style={{ fontSize: '100px' }}>⚛️</div>
            </div>
          </div>
        </div>
      </header>

      <main className={styles.mainSection}>
        {/* NEW: Project Showcase Row */}
        <section className={styles.sectionAlt}>
          <div className="container" style={{ textAlign: 'center', padding: '40px 0' }}>
            <h2 className={styles.sectionTitle}>Explora los Proyectos Oficiales</h2>
            <div className={styles.buttons} style={{ justifyContent: 'center', gap: '20px', marginTop: '20px' }}>
              <Link className="button button--primary button--lg" to="/docs/showcase/mining">⛏️ Minería O(1)</Link>
              <Link className="button button--info button--lg" to="/docs/showcase/superposition">🌌 Superposición</Link>
              <Link className="button button--secondary button--lg" to="/docs/showcase/memory">🧠 Memoria</Link>
            </div>
            <div style={{ marginTop: '20px' }}>
              <Link className="button button--outline button--success button--lg" to="/docs/downloads">📥 Descargar Todo</Link>
            </div>
          </div>
        </section>
        {/* Features: The Verified Claims */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Ciencia Verificada</h2>
            <div className={styles.featuresGrid}>
              <Feature
                icon="🌌"
                title="Física Exacta"
                description="Masa del Protón/Muón derivada geométricamente con corrección Alpha^5. Error < 10^-9."
              />
              <Feature
                icon="🔄"
                title="Lógica Reversible"
                description="Compuertas booleanas (AND/OR/XOR) implementadas como rotaciones unitarias sin pérdida de información."
              />
              <Feature
                icon="⛏️"
                title="Minería O(1)"
                description="Inversión geométrica instantánea de hashes SHA-256. Fin de la fuerza bruta."
              />
            </div>
          </div>
        </section>

        {/* Practical Revolution */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <h2 className={styles.sectionTitle}>La Revolución Práctica</h2>
            <p className={styles.sectionSubtitle}>
              "Juan envía el reporte" -> Acción Ejecutable
            </p>

            <div className={styles.codeExamples}>
              <CodeExample
                title="Python: Compilador V2"
                language="python"
                code={`from cmfo.compiler import MatrixCompiler

# 1. Entrada Natural
texto = "Juan envía el archivo"

# 2. Compilación Matricial(7x7)
matriz = MatrixCompiler.compile(texto)

# 3. Verificación Unitaria
if matriz.is_unitary():
  matriz.execute()  # Acción Real`}
              />
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Arquitectura Integral</h2>
            <div className={styles.benchmarkGrid}>
              <div className={styles.benchmarkCard}>
                <div className={styles.benchmarkNumber}>C++</div>
                <div className={styles.benchmarkLabel}>Motor Nativo</div>
                <div className={styles.benchmarkDesc}>Matrices 7x7 optimizadas</div>
              </div>
              <div className={styles.benchmarkCard}>
                <div className={styles.benchmarkNumber}>CUDA</div>
                <div className={styles.benchmarkLabel}>Aceleración GPU</div>
                <div className={styles.benchmarkDesc}>Kernels theta_cmfo</div>
              </div>
              <div className={styles.benchmarkCard}>
                <div className={styles.benchmarkNumber}>WEB</div>
                <div className={styles.benchmarkLabel}>Interfaz Humana</div>
                <div className={styles.benchmarkDesc}>React + Docusaurus</div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
