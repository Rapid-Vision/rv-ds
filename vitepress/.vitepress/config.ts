import { defineConfig } from 'vitepress'

export default defineConfig({
  srcDir: 'docs',
  title: 'rv-ds',
  description: 'Two-stage RV dataset export framework with streaming extractor/exporter plugins',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started' },
      { text: 'Plugin Dev', link: '/plugins/overview' },
      { text: 'Reference', link: '/reference/schema-dataset' }
    ],
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Getting Started', link: '/getting-started' },
          { text: 'Installation', link: '/installation' },
          { text: 'Quickstart', link: '/quickstart' }
        ]
      },
      {
        text: 'CLI',
        items: [
          { text: 'Overview', link: '/cli/overview' },
          { text: 'export Command', link: '/cli/export-command' },
          { text: 'Options Reference', link: '/cli/options-reference' }
        ]
      },
      {
        text: 'Concepts',
        items: [
          { text: 'Data Model', link: '/concepts/data-model' },
          { text: 'Streaming Model', link: '/concepts/streaming-model' },
          { text: 'Feature Contracts', link: '/concepts/feature-contracts' }
        ]
      },
      {
        text: 'Built-in Plugins',
        items: [
          { text: 'Built-in Extractors', link: '/extractors/builtins' },
          { text: 'Built-in Exporters', link: '/exporters/builtins' }
        ]
      },
      {
        text: 'Plugin Development',
        items: [
          { text: 'Overview', link: '/plugins/overview' },
          { text: 'Extractor API', link: '/plugins/extractor-api' },
          { text: 'Exporter API', link: '/plugins/exporter-api' },
          { text: 'Options and Validation', link: '/plugins/options-and-validation' },
          { text: 'SDK Helpers', link: '/plugins/sdk-helpers' },
          { text: 'Custom Features', link: '/plugins/custom-features' }
        ]
      },
      {
        text: 'Examples',
        items: [
          { text: 'Common Workflows', link: '/examples/common-workflows' },
          { text: 'Custom Extractor', link: '/examples/custom-extractor' },
          { text: 'Custom Exporter', link: '/examples/custom-exporter' },
          { text: 'Preview Exporters', link: '/examples/preview-exporters' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Dataset Schema', link: '/reference/schema-dataset' },
          { text: 'Sample Schema', link: '/reference/schema-sample' },
          { text: 'Instance Schema', link: '/reference/schema-instance' },
          { text: 'Errors', link: '/reference/errors' }
        ]
      },
      {
        text: 'Troubleshooting',
        items: [
          { text: 'Troubleshooting', link: '/troubleshooting' },
          { text: 'FAQ', link: '/faq' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Rapid-Vision/rv-export' }
    ]
  }
})
