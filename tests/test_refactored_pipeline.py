import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from lbm_suite2p_python import pipeline

# Mock inputs
@pytest.fixture
def mock_3d_array():
    # Time, Y, X
    mock = MagicMock()
    mock.__class__.__name__ = "TiffArray"
    mock.shape = (100, 512, 512)
    mock.ndim = 3
    mock.metadata = {}
    mock.filenames = ["test_file.tif"]
    return mock

@pytest.fixture
def mock_4d_array():
    # Time, Z, Y, X
    mock = MagicMock()
    mock.__class__.__name__ = "TiffArray"
    mock.shape = (100, 5, 512, 512)
    mock.ndim = 4
    mock.metadata = {}
    mock.filenames = ["test_vol.tif"]
    return mock

@patch("lbm_suite2p_python.run_lsp.run_plane")
@patch("lbm_suite2p_python.run_lsp._get_num_planes_from_array")
def test_pipeline_delegates_to_run_plane_for_3d(mock_get_planes, mock_run_plane, mock_3d_array):
    mock_get_planes.return_value = 1
    
    # Act
    pipeline(mock_3d_array, save_path="test_output")
    
    # Assert
    mock_run_plane.assert_called_once()
    # Check args
    call_args = mock_run_plane.call_args
    assert call_args.kwargs['input_data'] == mock_3d_array
    
@patch("lbm_suite2p_python.run_lsp.run_volume")
@patch("lbm_suite2p_python.run_lsp._get_num_planes_from_array")
def test_pipeline_delegates_to_run_volume_for_4d(mock_get_planes, mock_run_volume, mock_4d_array):
    mock_get_planes.return_value = 5
    
    # Act
    pipeline(mock_4d_array, save_path="test_output")
    
    # Assert
    mock_run_volume.assert_called_once()
    assert mock_run_volume.call_args.kwargs['input_data'] == mock_4d_array

@patch("lbm_suite2p_python.run_lsp.run_volume")
@patch("mbo_utilities.lazy_array.imread")
def test_pipeline_delegates_to_run_volume_with_planes_arg(mock_imread, mock_run_volume, mock_4d_array):
    # If passing 4D array directly to pipeline with specific planes
    pipeline(mock_4d_array, save_path="test_output", planes=[1, 3])
    
    mock_run_volume.assert_called_once()
    assert mock_run_volume.call_args.kwargs['planes'] == [1, 3]

@patch("lbm_suite2p_python.run_lsp.run_volume")
def test_pipeline_passes_parameters(mock_run_volume, mock_4d_array):
    pipeline(
        mock_4d_array, 
        save_path="test_output",
        dff_percentile=30,
        accept_all_cells=True,
        cell_filters=[{"name": "test"}]
    )
    
    kwargs = mock_run_volume.call_args.kwargs
    assert kwargs['dff_percentile'] == 30
    assert kwargs['accept_all_cells'] is True
    assert kwargs['cell_filters'] == [{"name": "test"}]

@patch("lbm_suite2p_python.run_lsp.run_volume")
def test_pipeline_list_input(mock_run_volume):
    files = ["p1.tif", "p2.tif"]
    # Provide save_path to avoid imread or file checks failing
    pipeline(files, save_path="test_output")
    
    mock_run_volume.assert_called_once()
    # run_volume receives input_data, which matches the input list
    assert len(mock_run_volume.call_args.kwargs['input_data']) == 2
