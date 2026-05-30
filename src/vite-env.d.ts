/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_FRONTEND_URL?: string;
  // more env variables can be defined here
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
