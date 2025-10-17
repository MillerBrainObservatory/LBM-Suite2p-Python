# Decay Time Constants (Tau) in Calcium Imaging Spike Inference Pipelines

Most calcium imaging pipelines model neural activity as an exponential decay process following each spike, making τ a key hyperparameter.

This guide summarizes *in vivo* τ values used or recommended by leading spike inference algorithms — **FOOPSI**, **OASIS**, **Suite2p**, **CaImAn**, and **CASCADE** — as reported in peer-reviewed literature and official documentation:contentReference[oaicite:0]{index=0}.

```{figure} ./_images/tau_comparison_overview.png
:alt: Comparison of GCaMP decay constants
:figclass: full-width
:name: fig-tau-overview

Decay time constants (τ) for commonly used GCaMP variants in *in vivo* spike inference pipelines.
```

---

## FOOPSI (Fast Non-Negative Deconvolution)

**FOOPSI** ([Vogelstein 2010](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423)) introduced probabilistic inference of spike trains using an exponential calcium model:  
```{math}
C_t = \gamma C_{t-1} + A s_t
```
where \( \gamma = e^{-\Delta t / \tau} \) defines the decay rate.

In *in vivo* work, FOOPSI typically assumes **τ ≈ 1.0 s** for GCaMP3–GCaMP5 indicators:contentReference[oaicite:1]{index=1}, which display relatively slow kinetics.  
This constant was fixed across datasets and used to calibrate spike likelihood.

```{figure} ./_images/foopsi_model.png
:alt: FOOPSI calcium model
:figclass: full-width
:name: fig-foopsi-model

Illustration of FOOPSI’s exponential decay model, where τ defines calcium transient length.
```

---

## OASIS and CNMF (Pnevmatikakis & Friedrich, 2016–2017)

**OASIS** improved upon FOOPSI by introducing a convex optimization solver for **AR(1)/AR(2)** calcium models ([Friedrich 2017](https://pubmed.ncbi.nlm.nih.gov/28291787/)).  
τ is explicitly passed as a parameter in seconds, often derived from indicator kinetics:contentReference[oaicite:2]{index=2}.

| **Indicator** | **τ (s)** | **Notes** | **Reference** |
|---------------|------------|------------|----------------|
| GCaMP6f | 0.5 – 0.7 | Fast kinetics | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| GCaMP6m | 1.0 – 1.3 | Medium decay | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| GCaMP6s | 1.5 – 2.0 | Slow kinetics | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| GCaMP7f | ~0.45 | Very fast | [Rupprecht 2025](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators) |
| GCaMP8f/m | 0.2 – 0.3 | Modern fast sensors | [Rupprecht 2025](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators) |
| GCaMP8s | ~0.5 | Slow variant | [Rupprecht 2025](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators) |

::::{grid}
:::{grid-item-card} GCaMP6 Family
:columns: 4
Fast (6f): 0.5 s Medium (6m): 1.1 s Slow (6s): 1.8 s
:::
:::{grid-item-card} GCaMP7 Family
:columns: 4
7f: 0.45 s 7s: 1.0 s 7c: 0.8 s
:::
:::{grid-item-card} GCaMP8 Family
:columns: 4
8f: 0.25 s 8m: 0.3 s 8s: 0.5 s
:::
::::

```{figure} ./_images/tau_families.png
:alt: Decay constants across GCaMP families
:figclass: full-width
:name: fig-tau-families

Approximate *in vivo* decay constants (τ) across GCaMP6–8 families from Pnevmatikakis 2016 and Rupprecht 2025.
```

---

## Suite2p Spike Deconvolution

**Suite2p**’s deconvolution module wraps OASIS, using τ as an internal parameter:contentReference[oaicite:3]{index=3}.  
Default values in Suite2p documentation are:

| **Indicator** | **Recommended τ (s)** | **Reference** |
|---------------|------------------------|----------------|
| GCaMP6f | 0.7 | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |
| GCaMP6m | 1.0 | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |
| GCaMP6s | 1.25 – 1.5 | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |

In *in vivo* datasets from [Dana 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609268/), these τ values accurately reproduce spike rates for PbN and cortical neurons.  
Suite2p applies the same τ≈0.7 s default for red indicators like **jRGECO1a** ([GitHub #233](https://github.com/MouseLand/suite2p/issues/233)).

```{figure} ./_images/suite2p_tau_deconv.png
:alt: Suite2p spike deconvolution and τ parameter
:figclass: full-width
:name: fig-suite2p-tau

Suite2p spike deconvolution showing effect of τ parameter on inferred spike rates.
```

---

## CaImAn (CNMF-E Pipeline)

**CaImAn** ([Giovannucci 2019](https://elifesciences.org/articles/38173)) uses a similar CNMF model to Suite2p but allows τ tuning within its CNMF-E fitting stage:contentReference[oaicite:4]{index=4}.

| **Indicator** | **τ (s)** | **Implementation Notes** |
|---------------|------------|---------------------------|
| GCaMP6f | 0.4 – 0.7 | Default for fast decay |
| GCaMP6m | 1.0 – 1.2 | Standard for medium kinetics |
| GCaMP6s | 1.5 – 2.0 | Slow indicator decay |

In most datasets, **τ=0.4 s** captures fast transients well, while slower indicators require τ > 1 s for stability ([Friedrich 2017](https://pubmed.ncbi.nlm.nih.gov/28291787/)).

```{figure} ./_images/caiman_tau_fit.png
:alt: CaImAn CNMF-E τ fitting
:figclass: full-width
:name: fig-caiman-tau

CaImAn CNMF-E fitting showing decay component corresponding to τ for different indicators.
```

---

## CASCADE (Deep Learning Inference)

**CASCADE** ([Rupprecht 2021](https://www.nature.com/articles/s41593-021-00938-1)) uses supervised networks trained on simultaneous electrophysiology and calcium data.  
τ is *not* explicitly specified but implicitly learned from training data:contentReference[oaicite:5]{index=5}.

- Original 2021 models were trained on GCaMP6 data (τ≈1–2 s).  
- Updated 2025 versions retrained on **GCaMP8f/m/s** achieved optimal inference with **effective τ ≈ 0.3–0.5 s**:contentReference[oaicite:6]{index=6}.

```{figure} ./_images/cascade_tau_learned.png
:alt: CASCADE learned decay constants
:figclass: full-width
:name: fig-cascade-tau

CASCADE’s neural network implicitly learns τ during training, adapting to GCaMP family kinetics.
```

---

## Summary

| **Pipeline** | **Uses τ?** | **Range (s)** | **Method** | **Reference** |
|---------------|-------------|----------------|-------------|----------------|
| FOOPSI | Yes | ~1.0 | Fixed exponential | [Vogelstein 2010](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| OASIS / CNMF | Yes | 0.3 – 2.0 | AR(1/2) model | [Pnevmatikakis 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) |
| Suite2p | Yes | 0.7 – 1.5 | OASIS internal | [Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) |
| CaImAn | Yes | 0.4 – 2.0 | CNMF-E fit | [Giovannucci 2019](https://elifesciences.org/articles/38173) |
| CASCADE | Implicit | 0.3 – 2.0 | Learned dynamics | [Rupprecht 2021](https://www.nature.com/articles/s41593-021-00938-1) |

::::{grid}
:::{grid-item-card} Key Takeaways
:columns: 12

- τ defines calcium transient decay and sets temporal resolution of spike inference  
- Optimal τ depends on both **indicator kinetics** and **frame rate**  
- Pipelines like Suite2p and CaImAn require τ tuning per GECI  
- CASCADE bypasses explicit τ by learning it implicitly  
- GCaMP8 series are ~3× faster than GCaMP6  
:::
::::

---

## Notes

All τ values summarized here reflect *in vivo* mammalian calcium imaging (typically ~30 Hz frame rate).  
In vitro or temperature-controlled decay times (e.g., 37 °C) can be >10× shorter.  
Choosing an incorrect τ biases both spike amplitude and inferred firing rate:contentReference[oaicite:7]{index=7}.

```{figure} ./_images/tau_effect_on_spikes.png
:alt: Effect of τ choice on inferred spikes
:figclass: full-width
:name: fig-tau-spike-bias

Effect of τ selection on inferred spike amplitude and frequency.
```

---

**Primary Sources:**  
[Vogelstein 2010 (FOOPSI)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) ·  
[Pnevmatikakis 2016 (CNMF)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) ·  
[Friedrich 2017 (OASIS)](https://pubmed.ncbi.nlm.nih.gov/28291787/) ·  
[Giovannucci 2019 (CaImAn)](https://elifesciences.org/articles/38173) ·  
[Suite2p Docs](https://suite2p.readthedocs.io/en/latest/settings.html) ·  
[Dana 2019 (GCaMP7)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609268/) ·  
[Rupprecht 2021 & 2025 (CASCADE)](https://www.researchgate.net/publication/391114094_Spike_inference_from_calcium_imaging_data_acquired_with_GCaMP8_indicators)
````
