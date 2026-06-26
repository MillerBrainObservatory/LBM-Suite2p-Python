selector_to_html = {"a[href=\"#suite2p-gui\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Suite2p GUI<a class=\"headerlink\" href=\"#suite2p-gui\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#compute-f-f\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Compute \u0394F/F<a class=\"headerlink\" href=\"#compute-f-f\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#load-results\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load Results<a class=\"headerlink\" href=\"#load-results\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#what-s-next\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">What\u2019s Next<a class=\"headerlink\" href=\"#what-s-next\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#quickstart\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Quickstart<a class=\"headerlink\" href=\"#quickstart\" title=\"Link to this heading\">#</a></h1><p><a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html\"><strong>User Guide</strong></a> |\n<a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/mbo_utilities/array_types.html#quick-reference\"><strong>Supported Filetypes</strong></a> |\n<a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/LBM-Suite2p-Python/api.html\"><strong>API Reference</strong></a> |\n<a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/\"><strong>MBO Hub</strong></a></p><p>Suite2p-based calcium imaging pipeline for Light Beads Microscopy data.</p>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};
