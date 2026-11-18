# GRDL-TE Fit-for-Purpose Scenarios & Ground Truth

Each subdirectory defines an apples-to-apples operational evaluation scenario
indexed by:

- **`modality`**: Sensor phenomenology (`SAR`, `EO`, `MSI`, `HSI`, `IR`)
- **`region_type`**: Background/clutter regime (`littoral`, `open_ocean`, `agricultural`, `urban`, `coastal_temperate`, `calibration_range`)
- **`signature_type`**: Detectable phenomenon or target class (`maritime_vessel`, `homogeneous_roi_and_edges`, `land_cover_classes`, `trihedral_corner_reflector`)
- **`evaluation_task`**: `detection`, `geolocation`, `radiometric_quality`, or `segmentation`

Every `scenario.yaml` pairs a sensor collection specification with a labeled
`ground_truth.geojson` and acceptance thresholds so candidate components
(`grdl-sar`, `grdl-eo`) and workflows (`grdl-runtime`) can be evaluated
quantitatively on both mission effectiveness and runtime performance.

