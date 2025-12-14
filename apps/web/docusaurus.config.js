// docusaurus.config.js
// Configuración oficial del sitio CMFO–UNIVERSE v∞
// Documentación científica, computación fractal y GPU kernels

module.exports = {
  title: 'CMFO–UNIVERSE v∞',
  tagline: 'Computación, Física y Lenguaje Unificados en un Autómata Fractal 7D',

  // Dominio del sitio (GitHub Pages)
  url: 'https://1jonmonterv.github.io',
  baseUrl: '/cmfo-universe/',
  trailingSlash: false,

  // Configuración GitHub Pages
  organizationName: '1jonmonterv',     // Tu usuario de GitHub
  projectName: 'cmfo-universe',        // Nombre exacto del repo
  deploymentBranch: 'gh-pages',        // Rama donde se publicará

  // Tratamiento de errores
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Iconos
  favicon: 'img/favicon.ico',

  // Localización
  i18n: {
    defaultLocale: 'es',
    locales: ['es'],
  },

  // Presets y estructura
  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/docs',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  // Tema y navegación
  themeConfig: {
    navbar: {
      title: 'CMFO–UNIVERSE v∞',
      logo: {
        alt: 'CMFO Logo',
        src: 'img/logo.svg',
      },
      items: [
        { to: '/docs/intro', label: 'Documentación', position: 'left' },
        {
          href: 'https://github.com/1jonmonterv/cmfo-universe',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      copyright: `CMFO–UNIVERSE v∞ © ${new Date().getFullYear()} • Sistema Fractal Unificado`,
    },
  },
};

