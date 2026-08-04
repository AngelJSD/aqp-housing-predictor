# Sitio del proyecto — aqp-housing-predictor

Sitio estático (Astro) que presenta y explica en español el pipeline
MLOps del repo, para un público de ingenieros de software: arquitectura,
recorrido por el pipeline, decisiones técnicas con evidencia y stack.
No duplica el `README.md` de la raíz (comandos de `docker compose`,
etc.) — para correr el pipeline en sí, ver ese README.

## Desarrollo local

```bash
npm install
npm run dev        # http://localhost:4321
```

```bash
npm run build      # genera ./dist (salida 100% estática)
npm run preview    # sirve ./dist para probar el build de producción
```

## Estructura

```
site/
├── public/
│   ├── architecture.svg      # copia del diagrama de la raíz del repo
│   └── diagrams/              # 3 diagramas nuevos (mismo estilo visual)
└── src/
    ├── layouts/                # BaseLayout, PipelineLayout (stepper)
    ├── components/              # Nav, Footer, DiagramFigure, DecisionCard
    └── pages/
        ├── index.astro, arquitectura.astro, stack.astro, decisiones.mdx
        └── pipeline/            # index.astro + 5 sub-páginas .mdx
```

Si `architecture.svg` cambia en la raíz del repo, hay que volver a
copiarlo a `site/public/architecture.svg` a mano — no hay symlink ni
build step que los mantenga sincronizados.

## Deploy a Cloudflare Pages

Proyecto Pages conectado por Git (push a `main` → deploy automático):

| Config                | Valor       |
| ---------------------- | ----------- |
| Root directory         | `site`      |
| Framework preset        | Astro       |
| Build command           | `npm run build` |
| Build output directory  | `dist`      |

No hace falta un adapter SSR de Astro ni `wrangler.toml` — la salida es
estática. Probado localmente con el simulador real de Cloudflare Pages:

```bash
npm run build
npx wrangler pages dev dist
```

Deploy manual (sin conectar el repo), útil para una primera prueba:

```bash
npx wrangler pages deploy dist
```
