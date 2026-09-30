# DEIC-Inventory

> ⚠️ **The v17 paper has been withdrawn (2026-09-30).** An audit of our own analyses found that
> several central claims of v17 do not hold as stated; most importantly, the conditional-process
> (moderated mediation) result is not established. We checked our main result ourselves, could not
> confirm it in the form claimed, and rewrote it in the form that holds. Corrected papers will be
> posted here. Everything below describes the May 2026 release, kept as a record; do not cite
> results from it.

Materials of a questionnaire study on empathy, psychopathy, dissociation and childhood adversity:
a Russian-language online sample (N = 1,426 analysed), the item bank, and the analysis scripts of
the May 2026 release.

## Live artifacts

- **Anonymized dataset (OSF, v1.5):** https://osf.io/y43zs/ (DOI: 10.17605/OSF.IO/Y43ZS)
- **Test platform:** https://deic-c2ed4.web.app/ru/ (RU), https://deic-c2ed4.web.app/en/ (EN)

## What's here — the May 2026 record

- `paper/source_ru/`, `paper/source_en/` — LaTeX sources of v17. The PDFs are withdrawn.
- `analysis/` — the v17 analysis scripts. They reproduce the numbers v17 printed; several of those
  numbers, and some of what v17 concluded from them, have since been corrected.
- `analysis/claims_provenance.tsv` — the v17 map from claims to source files, as it stood in May 2026.
- `instrument/questions_wave1_5.json` — the 156-item question bank (131 wave-1 items and 25 wave-1.5
  modules).
- `instrument/psychometrics.json` — the product layer, including four clusters (Witness / Defended /
  Wounded / Operator). They are cuts through a continuous distribution, not types.

## Dataset citation

```
Bodrov, A. (2026). DEIC-Inventory: anonymized dataset (N = 1470).
OSF. https://doi.org/10.17605/OSF.IO/Y43ZS
```

## Contact

deic.survey@gmail.com — questions, replication, collaboration

## License

- Paper text and figures: CC-BY-4.0
- Code and instrument data: MIT
