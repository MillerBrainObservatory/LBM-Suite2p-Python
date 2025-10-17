---
bibliography:
  - references.bib
---

# Decay Time Constants (Tau) in Calcium Imaging Spike Inference Pipelines

Understanding calcium indicator decay kinetics (the time constant **τ**) is essential for accurate spike inference in calcium imaging pipelines.  
Most deconvolution algorithms assume an **exponential decay model**, requiring τ (in seconds) as an input.  
This document summarizes *in vivo* decay constants used or recommended for common pipelines — including **FOOPSI**, **OASIS**, **Suite2p**, **CaImAn**, and **CASCADE**:contentReference[oaicite:0]{index=0}.

---

## FOOPSI (Fast Non-Negative Deconvolution)

**FOOPSI** (Vogelstein et al., 2010) introduced model-based spike inference with a fixed exponential decay constant τ:contentReference[oaicite:1]{index=1}.  
In practice, *in vivo* implementations typically assume **τ ≈ 1.0 s** for GECIs such as **GCaMP3/5**, representing the slower kinetics of early indicators.  
This foundational method demonstrated that constraining calcium transient decay improves spike inference accuracy ([Vogelstein 2010](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423)).

---

## OASIS (Online Active Set Method) and CNMF Deconvolution

**OASIS** (Friedrich et al., 2017) improved FOOPSI for online/batch deconvolution using an AR(1)/AR(2) model with specified τ:contentReference[oaicite:2]{index=2}.  
Typical *in vivo* τ settings match indicator kinetics:

| **Indicator** | **Typical τ (s)** | **Reference** |
|---------------|-------------------|----------------|
| GCaMP6f | 0.5–0.7 | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| GCaMP6m | 1.0–1.3 | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| GCaMP6s | 1.5–2.0 | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| GCaMP7f | 0.45 | [Rupprecht 2025](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators) |
| GCaMP8f/m | 0.2–0.3 | [Rupprecht 2025](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators) |
| GCaMP8s | 0.5 | [Rupprecht 2025](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators) |

These reflect progressively faster GECIs, with newer sensors (GCaMP7/8) decaying several times faster than GCaMP6 variants.

---

## Suite2p (Spike Deconvolution Module)

The **Suite2p** deconvolution module incorporates OASIS internally and provides recommended τ values for GCaMP indicators:contentReference[oaicite:3]{index=3}.  
Default settings from Suite2p documentation are:

| **Indicator** | **Recommended τ (s)** | **Source** |
|---------------|------------------------|-------------|
| GCaMP6f | 0.7 | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |
| GCaMP6m | 1.0 | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |
| GCaMP6s | 1.25–1.5 | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |

Users often extrapolate for newer indicators:  
**GCaMP7f ≈ 0.7 s** and **GCaMP7s ≈ 1.0 s**, consistent with *in vivo* measurements by [Dana et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609268/).

Suite2p also used **τ = 0.7 s** for **jRGECO1a**, matching the GCaMP6f default ([GitHub Issue #233](https://github.com/MouseLand/suite2p/issues/233)).  
Overall, Suite2p’s τ settings align with measured *in vivo* kinetics for each GECI.

---

## CaImAn (CNMF-E Pipeline)

**CaImAn** (Giovannucci et al., 2019) employs CNMF-based source extraction with similar τ parameters to Suite2p:contentReference[oaicite:4]{index=4}.  
Typical recommendations:

| **Indicator** | **τ (s)** | **Notes** |
|---------------|------------|------------|
| GCaMP6f | 0.4–0.7 | Captures fast decay; often fixed at 0.4 s |
| GCaMP6m | 1.0–1.2 | Intermediate decay |
| GCaMP6s | 1.5–2.0 | Slow indicator kinetics |

Friedrich et al. (2017) showed that an **AR(1)** model with τ≈0.4 s adequately captured GCaMP6f’s decay, while MLSpike and CNMF implementations reported similar optimal τ values ([Friedrich 2017](https://pubmed.ncbi.nlm.nih.gov/28291787/)).

---

## CASCADE (Supervised Deep Learning)

**CASCADE** (Rupprecht et al., 2021; 2025) uses supervised deep learning trained on ground-truth spike data:contentReference[oaicite:5]{index=5}.  
Unlike analytical models, CASCADE does **not** require a fixed τ parameter — instead, its network implicitly learns decay dynamics from training data.

The original CASCADE (2021) model was trained mostly on **GCaMP6 datasets**, corresponding to **τ ≈ 1–2 s**, while retrained models for **GCaMP8** data captured faster decays (**τ ≈ 0.3–0.5 s**):contentReference[oaicite:6]{index=6}.  
This retraining substantially improved inference accuracy for modern fast indicators.

---

## Summary Table

| **Pipeline** | **Uses τ?** | **Typical Range (s)** | **Notes** |
|---------------|-------------|-----------------------|------------|
| FOOPSI | Yes | ~1.0 | Fixed exponential decay model |
| OASIS / CNMF | Yes | 0.3–2.0 | User-tuned per GECI |
| Suite2p | Yes | 0.7–1.5 | Built-in per indicator |
| CaImAn | Yes | 0.4–2.0 | Similar to Suite2p; CNMF-based |
| CASCADE | Implicit | learned (~0.3–2.0) | Learned via training data |

---

## Notes and References

Recommended τ values above correspond to *in vivo* two-photon recordings at ~30 Hz frame rates.  
In vitro measurements (e.g., GCaMP6f half-decay ≈ 0.07 s at 37°C) can differ markedly.  
Choosing τ too large merges spikes; too small yields noisy estimates — the optimal value matches measured transient kinetics under each condition:contentReference[oaicite:7]{index=7}.

**Primary Sources:**  
[Vogelstein 2010 (FOOPSI)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423)  
[Friedrich 2017 (OASIS)](https://pubmed.ncbi.nlm.nih.gov/28291787/)  
[Pnevmatikakis 2016 (CNMF)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423)  
[Suite2p Documentation](https://suite2p.readthedocs.io/en/latest/settings.html)  
[Dana 2019 (GCaMP7)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609268/)  
[Rupprecht 2021 & 2025 (CASCADE)](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators)
