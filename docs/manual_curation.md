(manual_curation_gui)=
# Dataset Curation

Users are always encouraged to evaluate ROI's after any processing run.
There are a few differing approaches depending on the experimental question.

These approaches fall on two sides of a spectrum:
- Some users opt for an "accept everything" approach which sets all thresholds low enough to pass ROI criterion and use post-hoc correlation measures to narrow down cell counts.
- Other users use parameters that ideally get ~95% accuracy, and bridge the remaining ~5% through manually accepting and rejecting the misses.

Which approach you take may depend on the experimental question. Do you care about **sparesely active cells**, i.e. cells that may have 1 significant firing event over the course of the entire session?
If so, you may wish to "accept everything" 


[Trace Activity](./_images/trace_activity_gui.png)

(summary_images_gui)=
## Summary Images

Suite2p gives you several summary images to use when deciding accepted/rejected cells.

