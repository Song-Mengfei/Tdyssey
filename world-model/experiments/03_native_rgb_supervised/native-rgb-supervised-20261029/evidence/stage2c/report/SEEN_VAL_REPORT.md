# Independent seen-letter validation data and frozen indices

Recorded 162/162 trajectories, 10368/10368 controls; H1=9882/9882, H4=9396/9396.
40% moving /45% near-static thresholds describe validation distribution only; no frames are removed and no validation resampling is allowed.

| Letter | Raw controls | H1 windows | H4 windows | H1 moving | H1 near-static | Diagnostic target |
|---|---:|---:|---:|---:|---:|---|
| A | 576 | 549 | 522 | 0.6338797814207651 | 0.3224043715846995 | True |
| B | 576 | 549 | 522 | 0.6029143897996357 | 0.3570127504553734 | True |
| C | 576 | 549 | 522 | 0.604735883424408 | 0.36065573770491804 | True |
| E | 576 | 549 | 522 | 0.5846994535519126 | 0.366120218579235 | True |
| F | 576 | 549 | 522 | 0.5901639344262295 | 0.3697632058287796 | True |
| H | 576 | 549 | 522 | 0.6466302367941712 | 0.29690346083788705 | True |
| I | 576 | 549 | 522 | 0.5919854280510018 | 0.3588342440801457 | True |
| J | 576 | 549 | 522 | 0.6102003642987249 | 0.31693989071038253 | True |
| L | 576 | 549 | 522 | 0.6739526411657559 | 0.28233151183970856 | True |
| M | 576 | 549 | 522 | 0.5591985428051002 | 0.39344262295081966 | True |
| O | 576 | 549 | 522 | 0.5701275045537341 | 0.3806921675774135 | True |
| P | 576 | 549 | 522 | 0.6757741347905283 | 0.2859744990892532 | True |
| S | 576 | 549 | 522 | 0.6029143897996357 | 0.3533697632058288 | True |
| T | 576 | 549 | 522 | 0.697632058287796 | 0.2550091074681239 | True |
| U | 576 | 549 | 522 | 0.6520947176684881 | 0.28051001821493626 | True |
| V | 576 | 549 | 522 | 0.5446265938069217 | 0.4262295081967213 | True |
| X | 576 | 549 | 522 | 0.5282331511839709 | 0.43169398907103823 | True |
| Y | 576 | 549 | 522 | 0.599271402550091 | 0.3697632058287796 | True |

Natural/conditional/combined/development panel distributions, missing slots and temporal correlation are in indices/INDEX_REPORT.md and indices.json.
All raw contact×motion, actual spaces, four signed wall gaps and dual-wall requested-corner gaps are retained in h1/real-transition-statistics.csv and motion-statistics.json. H4 endpoint/path motion remains separate from H1 labels.
No optimizer, denoiser, training iterator or model prediction was run. Saved-state example rendering, when requested, performs zero physics controls.
