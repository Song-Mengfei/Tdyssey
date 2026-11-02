# Third-party sources and notices

## Project provenance

First-party working sources came from six local Tdyssey workspaces derived from [Song-Mengfei/Tdyssey](https://github.com/Song-Mengfei/Tdyssey). Source profiles record original Git HEADs and file hashes. No original author credit has been replaced by the repository organizer's name.

## External repositories

| Project | Pinned commit | Role / redistribution treatment |
| --- | --- | --- |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | `5ba07ac6661db573af695b419a7947ecb704690f` | Official baseline / simulator dependency. Recorded MIT text retained in `licenses/`; external clone excluded. |
| [GPC](https://github.com/han20192019/gpc_code) | `a96146e568826ef46134f73c22fdf59a675e7264` | Native WM/reference policy. No root license file found in the inspected pinned checkout. Source is referenced/fetched separately; no relicensing is implied. |
| [DPPO](https://github.com/irom-princeton/dppo) | `cc7234ad7ff39a8f32de3af903606723a16f0648` | Implementation reference, not vendored; upstream terms were not independently audited during packaging. |
| [DIWA](https://github.com/acl21/diwa) | `980d1ed5b893b7e33a732b89ee0916b0554e32d6` | Research reference, not a drop-in dependency; upstream terms were not independently audited during packaging. |

Exact copied upstream source duplicates found in saved freezes are represented by immutable source references in `manifests/source-map.json`. First-party adapters and adaptations remain in scope and retain their comments/attributions. This mechanical exact-match separation is not a complete legal audit of all derived code.

Model weights and datasets have separate provenance and may have separate terms. A Git source reference is not permission to redistribute those assets. Research-paper references are retained in categorized bibliography evidence; paper PDFs are not bundled.
