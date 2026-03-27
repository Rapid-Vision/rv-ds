import { defineConfig } from 'vitepress'

export default defineConfig({
  srcDir: 'docs',
  title: 'rv-ds',
  base: "/rv-ds/",
  description: 'Minimal docs for using rv-ds and writing plugins',
  themeConfig: {
    nav: [
      { text: 'Getting Started', link: '/' },
      { text: 'Extractors', link: '/extractors/builtins' },
      { text: 'Exporters', link: '/exporters/builtins' }
    ],
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Getting Started', link: '/' }
        ]
      },
      {
        text: 'Extractors',
        items: [
          { text: 'Built-in Extractors', link: '/extractors/builtins' },
          { text: 'Writing A Custom Extractor', link: '/extractors/custom' }
        ]
      },
      {
        text: 'Exporters',
        items: [
          { text: 'Built-In Exporters', link: '/exporters/builtins' },
          { text: 'Writing A Custom Exporter', link: '/exporters/custom' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Rapid-Vision/rv-ds' }
    ]
  }
})
