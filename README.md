# TMArucoAccuracyApp

Qt Widgets app for measuring ArUco marker xyz error using:
- Techman cobot for commanded offsets
- TMVisionary API 1202 for marker xyz JSON

## What this version follows closely

### Techman side
This version follows the reference tool more closely:
- commands can be wrapped as `TMSCT,<len>,<body>,*XX\r\n`
- default ACK token is `$TMSCT,4,1,OK,*5C`
- default move flow is:
  - send offset body
  - send `var_run=true`
  - send `ScriptExit()`
- `MoveDone` is waited as the completion token

Default offset body in the UI:

```text
1,var_offset_x={dx}\r\nvar_offset_y={dy}\r\nvar_offset_z={dz}
```

You can change this to match your drag-and-drop program exactly.

### TMVisionary side
This version requests API `1202` using the same packet style as the vision tool:
- 16-byte header with magic `0x5A`
- payload = JSON UTF-8
- parser expects fields discussed in the conversation:
  - `type`
  - `pub_nsec`
  - `data_nsec`
  - `width`, `height`
  - `found`
  - `marker_count`
  - `markers[]`
  - per marker: `id`, `center`, `c0..c3`, `has_xyz`, `xyz`

## Build
Open `TMArucoAccuracyApp.pro` in Qt Creator, or build with qmake.

## Main flow
1. Connect Techman and TMVisionary
2. Capture baseline marker mean (`ref0`)
3. For each offset:
   - send Techman offset command(s)
   - wait `MoveDone`
   - collect N valid API 1202 samples
   - compute mean/std
   - compute ref at current pose
   - write `samples.csv` and `summary.csv`

## Important notes
- Units must be consistent between Techman offsets and TMVisionary xyz.
- If your Techman program uses different variable names or sequence, edit them in the UI.
- If pose feedback is available, enable it and provide the pose request command.


Version v13 changes: measure flow uses initial home from robot P1, then per-point sends offset vars first, run=1, waits MoveDone, samples marker, then reset=1 only between subsequent points. Legacy mode sends each offset variable as a separate command.


## Robot -> Camera default TF

This build defaults to the confirmed 3D mapping:

- X_rb -> +Z_cm
- Y_rb -> -X_cm
- Z_rb -> -Y_cm

For camera-to-marker reference prediction, the app uses:

ref_i = ref0 - R_cm_rb * d_rb

So for robot offsets (dx, dy, dz) in robot base:

- x_ref = x0 + dy
- y_ref = y0 + dz
- z_ref = z0 - dx

`ref0` is captured automatically from the first Home baseline capture.
