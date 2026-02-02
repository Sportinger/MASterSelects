import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // Allow unused vars with _ prefix (common pattern for catch blocks)
      '@typescript-eslint/no-unused-vars': ['error', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_|^e$',
      }],
      // Downgrade any to warning - will fix incrementally
      '@typescript-eslint/no-explicit-any': 'warn',
      // Allow empty catch blocks (they often intentionally swallow errors)
      'no-empty': ['error', { allowEmptyCatch: true }],
      // React Hooks - keep rules-of-hooks as error (critical)
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      // React Compiler rules - downgrade to warnings (many false positives with existing patterns)
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/refs': 'warn',
      'react-hooks/preserve-manual-memoization': 'warn',
      'react-hooks/purity': 'warn',
      'react-hooks/invariant': 'warn',
      // React Refresh - allow non-component exports (common for constants/types)
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    },
  },
])
