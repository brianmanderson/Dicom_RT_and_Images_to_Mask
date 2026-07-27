# Examples

Runnable walkthroughs of DicomRTTool on public data. The [project
README](../README.md) is the API reference; these notebooks are the end-to-end
story.

| Notebook | What it covers |
|----------|----------------|
| [`01_DICOM_to_NIfTI_Dataset.ipynb`](01_DICOM_to_NIfTI_Dataset.ipynb) | The full pipeline — download, discover, survey, QC, ROI normalization, resampling, metadata, anonymized NIfTI export, and verification. |

## The data

Notebook 01 uses TCIA's
[**Pancreatic-CT-CBCT-SEG**](https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/)
collection: 40 pancreatic SBRT patients, each with a planning CT, structure sets,
and an **RT dose** grid — so the walkthrough exercises structures *and* dose on
real clinical data rather than images alone.

It is licensed **CC BY 4.0**; cite the collection if you publish with it. The
notebook downloads a subset — the planning CT plus every structure and dose
object per patient, roughly **3.3 GB for the default 30 patients**. Lower
`N_PATIENTS` for a faster first run.

## Running them

```bash
pip install DicomRTTool tcia_utils SimpleITK pandas matplotlib
```

Then open the notebook and run top to bottom. The first cell installs the
dependencies too, so Colab works with nothing set up locally.

## A note on what these write

Everything the notebooks produce lands under `Examples/pancreatic_ct_cbct/`,
which `.gitignore` excludes — along with `anonymization_key.json`, the file that
maps hashes back to MRNs. **That key is re-identification data:** keep it
access-controlled, never commit it, and never ship it alongside the imaging.
