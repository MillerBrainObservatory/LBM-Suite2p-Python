# Installation

**With `pip`**:

`pip install lbm_caiman_python`

**With `conda`**:
```bash
conda install -c conda-forge python=3.10
```

On ubuntu, do get `gui` dependencies you will need libxcursor:

``` bash
sudo apt install libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

```

## Troubleshooting

**UV - Fatal: smudge filter lfs failed**

``` bash
Use `git lfs logs last` to view the log.
      error: external filter 'git-lfs filter-process' failed
      fatal: docs/source/_static/guide_hello_world.png: smudge filter lfs failed
```

You can skip the smudge by prepending `GIT_LFS_SKIP_SMIDGE=1 uv <COMMAND>`.

`GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --active`
