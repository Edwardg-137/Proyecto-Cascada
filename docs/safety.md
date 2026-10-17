# Seguridad de software – teleoperación

**Los interbloqueos de software nunca reemplazan al e-stop por hardware.** El e-stop
debe cortar la potencia de los motores y del cautín por sí solo; el software, como mucho,
lee su estado. Ningún interbloqueo se desactiva "para probar": si estorba, se simula.

Cada regla es una función pura en `cascada_control/safety` (`Decision(allowed, reason, value)`).

| Regla | Comportamiento | Prueba |
|---|---|---|
| Watchdog | Si no llega ninguna muestra de pose en más de `watchdog_timeout_ms` (50 ms, POR VALIDAR), el seguidor mantiene la última consigna y queda enclavado: para volver a transmitir hay que soltar y volver a pisar el embrague, y entonces se re-indexa. | `test_watchdog_rule`, `test_watchdog_trips_in_configured_time` |
| Error de bus | Un `BusError` es un evento de seguridad: se detiene y se enclava igual que el watchdog. | `test_bus_error_is_safety_event` |
| Embrague | Suelto: el seguidor se queda quieto. Al pisarlo, las poses actuales del modelo y del seguidor pasan a ser la nueva referencia, así que no hay salto. Es una parada controlada, no una parada de emergencia. | `test_clutch_release_and_reengage_no_jump` |
| Espacio de trabajo | Caja del TCP por brazo. `reject` (por defecto) rechaza el comando y se queda en el último punto válido. `clamp` recorta, pero informa en `status`. La caja es provisional: hay que tomarla del CAD vigente. | `test_workspace_reject_and_clamp`, `test_workspace_reject_holds_inside`, `test_workspace_clamp_mode` |
| Límites articulares | Rangos del MJCF. Si la IK no converge o se sale de un rango, se rechaza el comando y el seguidor mantiene la posición. | `test_joint_limits` |
| Velocidad y aceleración | Límites sobre el TCP, con frenado a tiempo y sin sobrepaso. | `test_limiter_respects_speed_and_accel_and_lands` |

## Falta (no está hecho ni medido)

- Zonas prohibidas (cables, cámara, soportes), temperatura del cautín, extracción de humos, ESD y cambio de herramienta.
- Pruebas físicas registradas con fecha: el e-stop corta motores y cautín, y el watchdog detiene el brazo real en ≤ el tiempo configurado. **No se hicieron.**
- La parada por watchdog mantiene la última consigna, sin rampa de frenado. Hay que validarlo en hardware sin carga.
