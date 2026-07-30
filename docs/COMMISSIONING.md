# Site Commissioning Checklist — DicomRTTool

For a medical physicist commissioning DicomRTTool at their own institution
before its output is allowed to inform a clinical decision.

DicomRTTool is a reference implementation, not a cleared clinical device. The
badge on any index listing certifies that a specific commit met a review
standard — **it is not a substitute for local commissioning.** Treat this the
way you would treat commissioning any TPS feature: nothing is trusted until
your site has reproduced it on your own data, on your own systems.

Work through every section. Sections 1–2 are automated and take minutes;
sections 3–6 are the ones that actually matter, and they cannot be automated
because they depend on your scanners, your naming conventions, and your TPS.

---

## 1. Fix the version under test

Clinical use is supported **only from tagged releases**, never from `main`.

- [ ] Installed from a tagged release, not a branch or a working checkout
      ```bash
      pip install DicomRTTool==<version>
      ```
- [ ] Version recorded here: `________________`
- [ ] Installed dependency versions captured (`pip freeze > commissioning_env.txt`)
      and filed with this checklist
- [ ] The environment you commissioned is the environment clinical work will
      run in — same Python, same OS, same dependency set. If your production
      environment differs in any of these, commission that one instead.

## 2. Run the automated validation

This converts a synthetic CT + RT structure set whose true geometry is known
in closed form, and compares every mask against ground truth computed by
sub-voxel quadrature — not against another rasterizer.

```bash
pip install -r requirements-conformance.txt
python validation/run_validation.py
```

- [ ] All structures report PASS
- [ ] `validation/validation_report.md` filed with this checklist
- [ ] **Threshold disclosures read.** The report marks any structure whose
      tolerance was relaxed relative to the published conformance defaults.
      At the time of writing, `cube` is relaxed (Dice 0.99 → 0.98, volume
      error 0.03 → 0.04) because the rasterizer's boundary-inclusive polygon
      fill adds roughly a half-voxel of volume on each flat face. Confirm you
      accept every relaxation the report lists, or re-run with stricter
      thresholds: `--config your_thresholds.yaml`
- [ ] Environment table in the report matches section 1

A PASS here means the conversion path reproduces analytic geometry on
synthetic data. It says nothing about your scanners or your TPS. Continue.

## 3. Geometry on your own data

Synthetic data cannot exercise your acquisition and export chain. Use a real
(de-identified) study from **each** scanner and export path you intend to use.

- [ ] Round-trip a known structure set: DICOM → mask → NIfTI, and confirm
      voxel spacing, image origin, and orientation match the source series
- [ ] Slice ordering correct: confirm superior/inferior is not flipped, by
      checking a structure whose position is unambiguous (e.g. a lung apex)
- [ ] Non-uniform slice spacing: if any of your protocols produce variable
      slice gaps, verify one of those studies explicitly
- [ ] Anisotropic voxels: verify a study with in-plane spacing ≠ slice
      thickness
- [ ] Compare a computed structure volume against the volume your TPS reports
      for the same structure. Record the agreement: `________ %`
- [ ] Volume agreement is within a tolerance your site has decided in advance
      and written down here: `________ %`

## 4. Structure naming

Wrong names are the most likely clinical failure mode, and the quietest.

- [ ] Your site's structure naming conventions are represented in the
      association map you will use in production
- [ ] TG-263 target names verified where applicable
- [ ] Case, whitespace, and special-character variants your scanners actually
      emit are all mapped (export a real structure-name inventory and check it
      against your map — do not do this from memory)
- [ ] An unmapped structure name produces a visible, unmistakable outcome —
      confirm what your workflow does when a name is *not* matched, and that
      it does not silently produce an empty or mislabelled mask

## 5. The write-back paths

**This is the section that makes DicomRTTool clinically adjacent.** Complete
it if you will use either path; skip only if you can state that neither will
ever be used at your site.

### `prediction_array_to_RT()` — segmentation into a structure set

- [ ] A generated structure set **imports cleanly into your TPS** without
      warnings or geometry errors
- [ ] Contours land in the correct anatomical position — verify visually,
      overlaid on the planning CT, for at least one structure per anatomical
      site you intend to support
- [ ] The frame of reference and referenced SOP instances link correctly to
      the intended image series
- [ ] Volumes reported by your TPS for imported structures agree with the
      source mask within your stated tolerance
- [ ] Your clinical workflow requires review and editing by a qualified
      person before any generated contour is used for planning, and this is
      documented in the workflow, not just assumed

### `rewrite_RT()` — renaming ROIs in place

- [ ] Confirm the in-place write behaviour: **this function overwrites the
      structure set it is given.** Your workflow must operate on a copy, or
      have a verified backup, before this is run
- [ ] A rename is verified end to end: rename a structure, re-import to the
      TPS, confirm the intended structure and only that structure changed
- [ ] Confirm what happens when an association matches nothing, and when two
      associations could match the same ROI

## 6. Failure behaviour

A tool that fails loudly is safer than one that guesses.

- [ ] Malformed or truncated DICOM produces an error, not a silent partial
      result
- [ ] A structure set referencing a series that is absent produces an error
- [ ] An empty or all-background prediction array does not silently produce
      an empty structure set that could be mistaken for a valid one
- [ ] Someone other than the person who ran these checks has confirmed the
      failure cases above

---

## Sign-off

Commissioning is not complete until every box above is either checked or
explicitly marked not-applicable with a reason.

| | |
|---|---|
| Version commissioned | |
| Commissioned by | |
| Date | |
| Reviewed by (second physicist) | |
| Scope — paths approved for use | |
| Scope — paths NOT approved | |
| Local volume-agreement tolerance | |
| Accepted threshold relaxations | |

## Re-commissioning

Repeat this checklist when any of the following change:

- The DicomRTTool version — in full for any change to conversion, geometry,
  or the RT write paths; sections 1–2 at minimum otherwise
- Python, `pydicom`, `SimpleITK`, `numpy`, `opencv`, or `scikit-image`
  versions in your production environment
- Your TPS version, or its DICOM import configuration
- A new scanner, protocol, or export path
- Your structure naming conventions

Nothing here expires on a fixed schedule, but a commissioning record older
than the software it describes is not a commissioning record.
