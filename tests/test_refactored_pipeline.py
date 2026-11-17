"""pipeline() dispatch: planar input goes to run_plane, volumetric to run_volume."""

from unittest.mock import MagicMock, patch

from mbo_utilities import LazyArray

from lbm_suite2p_python import pipeline


def _lazy_array(nz):
    arr = MagicMock(spec=LazyArray)
    arr.nz = nz
    arr.shape = (100, 1, nz, 64, 64)
    arr.metadata = {}
    arr.filenames = ["test.tif"]
    return arr


@patch("lbm_suite2p_python.run_lsp.run_plane")
def test_planar_input_goes_to_run_plane(mock_run_plane, tmp_path):
    arr = _lazy_array(nz=1)
    pipeline(arr, save_path=tmp_path)
    mock_run_plane.assert_called_once()
    assert mock_run_plane.call_args.kwargs["input_data"] is arr


@patch("lbm_suite2p_python.run_lsp.run_volume")
def test_volumetric_input_goes_to_run_volume(mock_run_volume, tmp_path):
    arr = _lazy_array(nz=5)
    pipeline(arr, save_path=tmp_path, planes=[1, 3])
    kwargs = mock_run_volume.call_args.kwargs
    assert kwargs["input_data"] is arr
    assert kwargs["planes"] == [1, 3]


@patch("lbm_suite2p_python.run_lsp.run_volume")
def test_parameters_forwarded_to_run_volume(mock_run_volume, tmp_path):
    pipeline(
        _lazy_array(nz=5),
        save_path=tmp_path,
        dff_percentile=30,
        norm_method="zscore",
        accept_all_cells=True,
        cell_filters=[{"name": "test"}],
    )
    kwargs = mock_run_volume.call_args.kwargs
    assert kwargs["dff_percentile"] == 30
    assert kwargs["norm_method"] == "zscore"
    assert kwargs["accept_all_cells"] is True
    assert kwargs["cell_filters"] == [{"name": "test"}]


@patch("lbm_suite2p_python.run_lsp.run_volume")
@patch("mbo_utilities.imread")
def test_path_input_is_opened_with_imread(mock_imread, mock_run_volume, tmp_path):
    mock_imread.return_value = _lazy_array(nz=5)
    pipeline(["p1.tif", "p2.tif"], save_path=tmp_path)
    mock_imread.assert_called_once_with(["p1.tif", "p2.tif"])
    assert mock_run_volume.call_args.kwargs["input_data"] is mock_imread.return_value
