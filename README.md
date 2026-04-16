# STM Bulk Process

STM Bulk Process is a desktop GUI application for batch processing `.sm4` scanning tunneling microscopy and related scanning probe microscopy data. It is built in Python on top of **SPyM** and provides an interactive workflow for previewing, cleaning, exporting, and organizing STM image data.

## Features

- Load and preview one or more `.sm4` files
- Browse files and channels directly in the app
- Apply common STM/SPM image-processing operations, including:
  - plane correction
  - zero correction
  - alignment
  - scar / stripe removal
  - row and column flattening
  - Gaussian, median, mean, and sharpen filters
  - contrast percentile adjustment
  - colormap selection and inversion
  - image flipping
- Save and load processing presets with plain-text config files
- Batch export high-resolution PNG images
- Export forward/backward stitched topography composites
- Export metadata for each processed file
- Remove files from the active batch
- Delete unwanted source `.sm4` files directly from the app
- Open the containing folder for the current file

## Input and Output

### Input
- `.sm4` files

### Output
- high-resolution `.png` images for each exported channel
- stitched topography composites
- `metadata.txt` summaries for each processed file
- plain-text configuration files for saved processing preferences

## Requirements

- Python 3.x
- [SPyM](https://spym-docs.readthedocs.io/)
- NumPy
- Matplotlib
- Tkinter

Depending on your Python environment, `tkinter` may already be included.

## Third-Party Software

This application is built on top of SPyM, which is licensed under the MIT License.

## Acknowledgments

This application was developed with assistance from OpenAI's ChatGPT for code generation, refactoring, UI design suggestions, and documentation support. Final testing, review, and use decisions were performed by the author.

## Installation

Clone the repository:

```bash
git clone https://github.com/raburnley/stm-bulk-process.git
cd stm-bulk-process
