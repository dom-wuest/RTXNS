# RTX Neural Shading Change Log

## 1.3.1
- Updated documentation to include Linux instructions.
- Migrate `createGraphicsPipeline` from deprecated function to the current NVRHI API.
- Update to `ResultsWidget` to allow for user configuration of data size and graph ranges. 
- Bug Fixes
	- Corrected `L2Relative` loss function.

## 1.3.0
- Added loss visualization graphs to the `ShaderTraining` and `SimpleTraining` examples.
- Added ImPlot support, enabling real-time plotting for diagnostics and training visualization.
- Introduced helper utilities for tracking and accumulating loss each epoch.
- Bug Fixes
	- Fixed incorrect buffer sizes in `ShaderTraining`.
	- Resolved potential issues when loading and storing networks.	
- Updates
	- Updated Slang to version `2025.23.1`.	 
## 1.2.2
- Fixes bug in matrix alignment
- Updated error messages for DX12

## 1.2.1 
- Added learning rate scheduler with cosine decay for smoother convergence over long training runs.
- Simple Training
	- Improved training stability with learning rate scheduler
	- Added UI display to show the current learning rate during training.

## 1.2.0
- Added Slangpy Inferencing sample.
	- This sample demonstrates how to deploy a neural network prototyped with Python and SlangPy using C++ and Slang.
- Updated samples to use NVRHI for Cooperative Vector format queries and matrix conversions.

## 1.1.0
- Added DX12 cooperative vector support using Preview Agility SDK.
- Moved matrix conversion to GPU.

## 1.0.0
- Initial release.