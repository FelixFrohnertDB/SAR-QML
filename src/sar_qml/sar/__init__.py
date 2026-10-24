"""Geometry-aligned classification of synthetic SAR phase histories.

Supporting code for the two SAR notebooks:

* ``geometry`` -- the sampled omega-k surface of one acquisition, the fixed
  query f*_geom derived from it, and the label-independent channel phase;
* ``gates`` -- single-qubit rotations and the azimuth Fourier readout;
* ``common`` -- reproducible seeds, exactly balanced labels, balanced accuracy
  and printing of validity checks;
* ``checkpoint`` -- configuration hash (``RUN_ID``) and the resumable
  per-task result CSV;
* ``capacity_matched`` -- scene generator, reduced-state classifier, training,
  experiments, checks and statistics of
  ``notebooks/fig_07_sar_geometry_classification.ipynb``;
* ``double_reuploading`` -- the same for the double data re-uploading circuit
  of ``notebooks/fig_08_double_reuploading_geometry.ipynb``.
"""
