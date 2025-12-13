import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

function Feature({ title, description }) {
  return (
    <div className={styles.featureCol}>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

export default function Home() {
  return (
    <Layout
      title="CMFO–UNIVERSE v∞"
      description="Motor de Computación Fractal Determinista"
    >
      <header className={styles.heroBanner}>
        <div className="container">
          <h1 className={styles.title}>CMFO–UNIVERSE v∞</h1>
          <p className={styles.subtitle}>
            Motor de Cálculo Fractal • Álgebra T⁷ • Física de Solitones
          </p>

          <div className={styles.buttons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/intro"
            >
              Documentación Maestra
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="https://github.com/1jonmonterv/cmfo-universe"
            >
              GitHub (v1.0 Release)
            </Link>
          </div>
        </div>
      </header>

      <main className={styles.mainSection}>
        <section className={styles.section}>
          <h2>Nuevas Capacidades (Maximum Level)</h2>
          <div className={styles.featuresGrid}>
            <Feature
              title="Física de Solitones"
              description="Simulación exacta de colisiones Kink-Antikink en el campo Sine-Gordon, con preservación de carga topológica y visualización GIF."
            />
            <Feature
              title="Interoperabilidad Total"
              description="SDKs nativos para Python (Ciencia), Node.js (Web) y C++ (Alto Rendimiento). Integración fluida vía FFI y Bindings."
            />
            <Feature
              title="CUDA Phase 2"
              description="Kernels de GPU actualizados con acople N-Cuerpos (Kuramoto-like) para simular emergencia fractal masiva."
            />
          </div>
        </section>

        <section className={styles.sectionAlt}>
          <h2>Benchmarks de Estrés</h2>
          <p>
            El sistema ha sido sometido a pruebas de estrés de "Nivel Máximo":
          </p>
          <ul>
            <li><strong>Flood Test:</strong> 100,000+ operaciones tensoriales sin degradación numérica.</li>
            <li><strong>Stability Search:</strong> Inversión robusta de matrices T⁷ aleatorias.</li>
            <li><strong>Cross-Platform:</strong> Ejecución consistente en CPU y GPU.</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Arquitectura Fractal 7D</h2>
          <p>
            Todo el conocimiento se deriva desde una estructura geométrica exacta basada en la proporción áurea φ.
            El CMFO reemplaza la lógica booleana con <strong>φ-Logic</strong>, permitiendo estados continuos y operaciones reversibles.
          </p>
        </section>

        <section className={styles.ctaSection}>
          <h2>Únete a la Revolución Fractal</h2>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro"
          >
            Leer Teoría Completa
          </Link>
        </section>
      </main>
    </Layout>
  );
}
