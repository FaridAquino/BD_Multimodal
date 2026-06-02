# Guía de contribución

## Flujo de ramas
- `main`: siempre estable y desplegable. Protegida (ver abajo).
- `develop`: integración. Los features se mergean aquí primero.
- Features: `feat/<area>-<tema>` (p.ej. `feat/text-spimi`, `feat/image-kmeans`).
- Fixes: `fix/<area>-<tema>`.

## Ciclo de trabajo
1. Toma un issue y asígnatelo (con su milestone y label).
2. Crea tu rama desde `develop`.
3. Commits con convención: `feat(text): tokenización + stopwords`.
4. Abre un PR contra `develop` usando la plantilla. Vincula `Closes #N`.
5. CI debe pasar (ruff + pytest).
6. **Al menos 1 review aprobado.** El Tech Lead revisa core/infra/db obligatoriamente (CODEOWNERS).
7. Merge tipo *squash*. El issue se cierra solo al mergear.

## Reglas de oro
- No cambies firmas en `src/core/` sin acuerdo previo (issue con label `infra`).
- Nada de merge directo a `main`.
- Cada quien abre y cierra sus propios issues -> evidencia de contribución para la rúbrica.

## Configuración de branch protection (la activa el Tech Lead en GitHub)
Settings → Branches → Add rule, para `main` y `develop`:
- Require a pull request before merging  ✓
- Require approvals: 1  ✓
- Require review from Code Owners  ✓
- Require status checks to pass (CI)  ✓
