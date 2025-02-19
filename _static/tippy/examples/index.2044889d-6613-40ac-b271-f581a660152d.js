selector_to_html = {"a[href=\"notebooks/s2p_binaryfile.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Exploring Suite2p.io.BinaryFile<a class=\"headerlink\" href=\"#exploring-suite2p-io-binaryfile\" title=\"Link to this heading\">#</a></h1>", "a[href=\"#tutorial\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Tutorial<a class=\"headerlink\" href=\"#tutorial\" title=\"Link to this heading\">#</a></h1>", "a[href=\"notebooks/suite2p_segmentation.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>Suite2p Segmentation<a class=\"headerlink\" href=\"#suite2p-segmentation\" title=\"Link to this heading\">#</a></h1>"}
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
