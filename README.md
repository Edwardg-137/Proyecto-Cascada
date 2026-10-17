# cascada-control

Software de control del KSR-01 (Проект «Каскад»): teleoperación de dos seguidores
SO-ARM100 (servos STS3215) a partir de la pose de la herramienta modelo, con escala
cartesiana 8:1.

## Cadena

pose del modelo → delta respecto a la referencia → posición / escala (8 por defecto,
escalar o por eje) → zona muerta → filtro 1€ → límites de velocidad, aceleración y
espacio de trabajo → IK del SO-ARM100 → objetivos articulares → `Bus`.

- **Orientación:** se copia 1:1, sin escalar (`orientation: copy`). El brazo tiene
  5 ejes: la IK cumple la posición y aproxima la orientación lo que el brazo permite.
- **Fuerza:** nunca se escala con `position_scale` (no implementada todavía).
- **Embrague:** suelto, el seguidor se queda quieto. Al pisarlo se re-indexa y no hay salto.
- **Watchdog:** si no llega ninguna muestra de pose en `watchdog_timeout_ms`, el seguidor se detiene.
- **Fuente de pose:** el método de medición está POR DEFINIR. Por ahora solo
  existe `ScriptedPoseSource`, detrás de la interfaz `PoseSource`.
- **Bus:** por ahora solo hay `MockBus` y `MujocoBus` (opcional). No hay todavía driver Feetech real.

La geometría sale del MJCF oficial (mujoco_menagerie), ver
`cascada_control/kinematics/models/so_arm100/SOURCE.md`.

## Uso

```
uv sync                 # numpy, pyyaml, pytest
uv sync --extra sim     # añade MuJoCo para tests/sim
uv run pytest
```

Configuración: `config/teleop.yaml` (los valores marcados POR VALIDAR no están medidos).
Seguridad: `docs/safety.md`.
