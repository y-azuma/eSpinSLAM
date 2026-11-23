# Cited-source audit: eSfO front end

This audit makes the hierarchy-level-3 evidence used for the adapted eSfO
front end reviewable without network access.  It does **not** make the online,
per-chunk eSpinSLAM adaptation exact.

## Source identity

- Elias Elms, Gabriel Latif, and Tat-Jun Chin, *Event-based
  Structure-from-Orbit*, CVPR 2024 / arXiv:2405.06216.
- Audited PDF: arXiv version, 10 PDF pages, SHA-256
  `7ad59e3495d37a0772e8d2274db9202a0c1601f1434779aaa4ad3b826d6caa00`.
- Locations below use PDF page numbers (not the printed footer alone).

## Offline-verifiable evidence map

The following quoted fragments total fewer than 25 source words.  Equations
and table cells are identified by location rather than reproduced at length.

| Reproduction item | Auditable source location | Source evidence |
|---|---|---|
| Same-polarity density and mean | PDF p.4, Sec. 4.1, Eq. (2), paragraph immediately below | Eq. (2) contains `I(p_i=p_q)`; the paragraph says **“same polarity as ei”**. |
| Density radius `lambda=7` | PDF p.6, Sec. 7.1, Table 2, row “Local event neighborhood distance” | Table cell value `7`. |
| Polarity-agnostic HDBSCAN | PDF p.4, Sec. 4.2, first paragraph | **“disregarding polarity”**. |
| HDBSCAN `minPts=10`, selection epsilon `5` | PDF p.6, Sec. 7.1, Table 2, HDBSCAN rows | Table cell values `10` and `5`. |
| Head/tail sample count `N_sigma=5` | PDF p.4, Sec. 4.2, descriptor paragraph; PDF p.6, Table 2 | Descriptor uses **“last and respectively first”** events; table cell value `5`. |
| Forward gate and `phi=30` | PDF p.4, Sec. 4.2, Eq. (3) and preceding paragraph; PDF p.6, Table 2 | **“spatio-temporal hemisphere of radius”**; Eq. (3) defines the gate; table cell value `30`. |

## What this source does not settle

eSfO clusters an offline event stream.  eSpinSLAM instead says that this
front end is adapted to approximately 1,000-event online chunks and a global
track history.  Therefore the constants above are cited inputs, while their
per-chunk use is **Ambiguous (adapted)** rather than Exact.  In particular:

- neither source states a time-to-pixel scale in Euclidean `(t,x,y)`;
- simultaneous current-chunk tracklets competing for one online track are not
  defined by eSfO;
- applying `minPts=10` independently to small chunks can behave differently
  from clustering the complete stream.

The frozen primary time coordinate is seconds multiplied by `1`.  With
`phi=30`, that literal reading permits a time difference of almost 30 seconds
when spatial displacement is small, so an association can cross multiple
rotations.  This is a known limitation, not a reason to tune the scale after
observing results.

If these cited constants fail the predeclared D-003 minimum output on the dev
scenes, the result is reported as a reproduction failure.  They are not
reopened as a performance sweep.
