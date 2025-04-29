from suite2p import default_ops as s2p_default_ops

def default_ops(metadata=None, ops=None, nplanes=1):
    """
    Returns default ops for Suite2P processing on Light Beads Microscopy datasets.

    Parameters
    ----------
    metadata : dict, optional
        Metadata dictionary containing information about the dataset.
    ops : dict, str or Path, optional
        Path to or dict of suite2p ops.
    nplanes : int, optional
        Number of z-planes that will be processed by Suite2p, may differ from the number of z-planes in the dataset.

    Returns
    -------
    dict
        Default ops for Suite2P processing.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> metadata = mbo.get_metadata("D://demo//raw_data//raw_file_00001.tif")  # noqa
    >>> ops = lsp.default_ops(metadata=metadata)  # noqa
    >>> # No ops are passed, so the default ops are used.
    >>> lsp.run_plane(
    >>>    ops=ops,
    >>>    input_tiff="D://demo//raw_data//raw_file_00001.tif",
    >>>    save_path="D://demo//results",
    >>>    save_folder="v1"
    >>> )
    """
    if ops is None:
        ops = s2p_default_ops()

    # typical neuron ~16 microns
    ops["fs"] = metadata["frame_rate"]
    ops["nplanes"] = 1
    ops["nchannels"] = 1
    ops["do_bidiphase"] = 0
    ops["do_regmetrics"] = True

    # suite2p iterates each plane and takes ops['dxy'][i] where i is the plane index
    ops["dx"] = [metadata["pixel_resolution"][0]]
    ops["dy"] = [metadata["pixel_resolution"][1]]

    return ops
