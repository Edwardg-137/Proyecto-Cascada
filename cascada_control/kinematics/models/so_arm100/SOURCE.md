# Origen del modelo

- Archivo: `so_arm100.xml` (sin cambios; las mallas `assets/*.stl` NO se incluyen).
- Repositorio: https://github.com/google-deepmind/mujoco_menagerie
- Ruta: `trs_so_arm100/so_arm100.xml`
- Commit: `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35` (HEAD de `main`, descargado 2026-10-17).
  Último commit que tocó `trs_so_arm100/`: `71f066ad0be9cd271f7ed58c030243ef157af9f4` (2026-07-04).
- Changelog del modelo: 2025-06-09 (límites articulares ajustados al robot real).
- Derivado por Menagerie del URDF oficial de TheRobotStudio:
  https://github.com/TheRobotStudio/SO-ARM100/blob/4e9c5588d8a8415b6a6c2142a0ce8c32207cf3e9/URDF/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf
- Licencia: Apache-2.0 (`LICENSE` en esta carpeta).

La cinemática (`cascada_control.kinematics`) lee posiciones, orientaciones, ejes y
rangos articulares directamente de este archivo; no hay longitudes escritas a mano.
