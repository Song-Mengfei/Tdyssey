# Stage2A independent sampler review

Static review passed. The frozen1B baseline is `57fecfc81b7e6ef025b1154102e7c7a6e7ab370aa495ba14796052c3b42ec900`; the18-letter adapter is `38fc876f8c182a28f3b1a5e067c6a49e349367cb2485dbf85aa6e91f72858f11`. All non-init sampler methods retain exact source text. Original constants, recovery limits and phases match. Only the approved-letter guard and side-sign index expression changed, with three added mapping constants.

The old six indices remain T0/L1/O2/C3/X4/B5. The new indices are A6/E7/F8/H9/I10/J11/M12/P13/S14/U15/V16/Y17. The18-letter whitelist rejects all sealed letters. All 10 inspected core source rows match the1B source identity.

This is not complete collection readiness approval: the final collector, frozen162 requests, ledger and failure exits still require review. Reviewer constructed0 environments and executed0 controls, models, optimizer updates or server jobs.
