# Capital-letter policy videos

These are the actual recorded attempts from the first frozen official CNN policy
sweep: all 26 capitals plus a T with a 10%-shorter top bar. Every shape starts from
common scene 100000 with the same policy noise seed. No adaptation was performed.
T and Y succeeded; the other shapes reached the 300-control limit. One scene per
shape does not establish a reliable success rate or the policy's general limit.

Click an MP4 to open its GitHub file page, then use its raw/download control to
watch it locally. Videos play at 10 frames per second, including initial frame
zero, independent of inference time. These 27 curated clips are committed directly
to the repository and total **12,025,483 bytes (about 11.5 MiB)**. Future raw runs,
datasets and checkpoints stay outside Git.

| Shape | Outcome | Controls | Best IoU | Final IoU | Video |
| --- | --- | --- | --- | --- | --- |
| T | Solved | 128 | 0.9241 | 0.9241 | [MP4](media/official-alphabet-2026-10-19/T-100000.mp4) |
| A | Step limit | 300 | 0.5433 | 0.1848 | [MP4](media/official-alphabet-2026-10-19/A-100000.mp4) |
| B | Step limit | 300 | 0.3695 | 0.2728 | [MP4](media/official-alphabet-2026-10-19/B-100000.mp4) |
| C | Step limit | 300 | 0.2755 | 0.0645 | [MP4](media/official-alphabet-2026-10-19/C-100000.mp4) |
| D | Step limit | 300 | 0.8154 | 0.4172 | [MP4](media/official-alphabet-2026-10-19/D-100000.mp4) |
| E | Step limit | 300 | 0.3129 | 0.2453 | [MP4](media/official-alphabet-2026-10-19/E-100000.mp4) |
| F | Step limit | 300 | 0.1749 | 0.1542 | [MP4](media/official-alphabet-2026-10-19/F-100000.mp4) |
| G | Step limit | 300 | 0.3782 | 0.2302 | [MP4](media/official-alphabet-2026-10-19/G-100000.mp4) |
| H | Step limit | 300 | 0.2261 | 0.1311 | [MP4](media/official-alphabet-2026-10-19/H-100000.mp4) |
| I | Step limit | 300 | 0.4058 | 0.1898 | [MP4](media/official-alphabet-2026-10-19/I-100000.mp4) |
| J | Step limit | 300 | 0.1876 | 0.1165 | [MP4](media/official-alphabet-2026-10-19/J-100000.mp4) |
| K | Step limit | 300 | 0.7512 | 0.0620 | [MP4](media/official-alphabet-2026-10-19/K-100000.mp4) |
| L | Step limit | 300 | 0.3015 | 0.0000 | [MP4](media/official-alphabet-2026-10-19/L-100000.mp4) |
| M | Step limit | 300 | 0.3322 | 0.1233 | [MP4](media/official-alphabet-2026-10-19/M-100000.mp4) |
| N | Step limit | 300 | 0.4352 | 0.2607 | [MP4](media/official-alphabet-2026-10-19/N-100000.mp4) |
| O | Step limit | 300 | 0.3384 | 0.1099 | [MP4](media/official-alphabet-2026-10-19/O-100000.mp4) |
| P | Step limit | 300 | 0.2332 | 0.2291 | [MP4](media/official-alphabet-2026-10-19/P-100000.mp4) |
| Q | Step limit | 300 | 0.2599 | 0.1725 | [MP4](media/official-alphabet-2026-10-19/Q-100000.mp4) |
| R | Step limit | 300 | 0.4032 | 0.2456 | [MP4](media/official-alphabet-2026-10-19/R-100000.mp4) |
| S | Step limit | 300 | 0.5755 | 0.5755 | [MP4](media/official-alphabet-2026-10-19/S-100000.mp4) |
| U | Step limit | 300 | 0.3023 | 0.1137 | [MP4](media/official-alphabet-2026-10-19/U-100000.mp4) |
| V | Step limit | 300 | 0.3422 | 0.1713 | [MP4](media/official-alphabet-2026-10-19/V-100000.mp4) |
| W | Step limit | 300 | 0.3919 | 0.2577 | [MP4](media/official-alphabet-2026-10-19/W-100000.mp4) |
| X | Step limit | 300 | 0.2896 | 0.1684 | [MP4](media/official-alphabet-2026-10-19/X-100000.mp4) |
| Y | Solved | 235 | 0.9121 | 0.9121 | [MP4](media/official-alphabet-2026-10-19/Y-100000.mp4) |
| Z | Step limit | 300 | 0.4648 | 0.1616 | [MP4](media/official-alphabet-2026-10-19/Z-100000.mp4) |
| T_narrow | Step limit | 300 | 0.8717 | 0.7751 | [MP4](media/official-alphabet-2026-10-19/T_narrow-100000.mp4) |

Several attempts got close and then lost alignment. D reached about 90% target
coverage, K 86%, and the narrower T 93%; all missed the unchanged success rule
requiring coverage above 95%. Best IoU and final IoU retain that distinction.

See [full validation and near-miss evidence](VALIDATION.md#completed-frozen-policy-alphabet-sweep-2026-10-19)
and [geometry/reset protocol](OFFICIAL_SHAPES.md). The local HTML gallery also has
best-frame screenshots; it is generated from saved reports with
`scripts/summarize_shapes.py`. The complete report and screenshots remain outside
Git. [Video provenance](media/official-alphabet-2026-10-19/videos.json) records exact
sizes, frame counts and SHA-256 hashes for these unchanged MP4 files.
