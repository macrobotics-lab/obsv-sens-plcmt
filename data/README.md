# Data file conventions

- Each `.csv` file corresponds to a single trial run.
- Units are consistent across all folders.

## Header row description

- `Time`: Time [s]
- `Pressure_setpoint`: Pressure input [kPa]
- `Measured_pressure`: Pressure measured during the trial [kPa]
- `MarkerX_x`, `MarkerX_y`, `MarkerX_z`: 3D position of marker X [mm], for X = 1 to 21.
  - The robot segment has 3 chambers (A, B, C):
    - Chamber A: markers 1-7 (base to tip)
    - Chamber B: markers 8-14
    - Chamber C: markers 15-21
- `SpineX_x`, `SpineX_y`, `SpineX_z`: 3D position of the Xth centroid [mm], formed by markers (X, X+7, X+14), for X = 1 to 7.
- `Spine2DX_x`, `Spine2DX_y`: 2D position of the Xth projected centroid onto plane of best fit [mm], for X = 1 to 7.

## File naming

- `impulse` folder:
  Files are named `xkPa.csv`, where x is the constant pressure setpoint [kPa] applied to chamber A before an external impulse is applied.
- `sine` folder:
  Files are named `AxkPa_wy.csv`, where:
  - x is the amplitude [kPa]
  - y is the frequency [rad/s]
  - The pressure input [kPa] is p_set = x *(sin(y* t) + 1)
- `step` folder:
  Files are named `xkPa.csv`, where x is the step input pressure value [kPa].
