# Weighted Hazard Scenario Approach

This repo is associated with the following paper:

> **Citation**
>
> Moayyedi, S.A., Yang, D.Y., 2025. Objective risk assessment for bridge management using weighted hazard scenarios. *Structure and Infrastructure Engineering*, 21, 2115–2127. https://doi.org/10.1080/15732479.2025.2529468

## Package: `pyRiskTable`

This folder contains tools and scripts of the developed method. The folder `tutorial` provides examples of using the weighted hazard scenario approach to generate risk table and evaluate risk.

## Input Folder: `bridge-info`

This folder contains the following files:

* `Selected_Bridges_April_22_2024_05_49_28_CSV.csv`: ODOT-owned bridges downloaded from InfoBridge
* `bridges_w_soil.gpkg`: a GIS file created by the following steps:
    * convert bridges coordinate in the previous CSV file to points 
    * overlay with NEHRP-Site-Class tiff (not provided due to file size) to add a field of soil class
* `fragility-summary.csv`: bridge class, modification factors, and medians of DSs obtained based on HAZUS

To analyze other bridges, similar files can be prepared for other regions.

## Input Folder: `USGS-data`

This folder contains USGS hazard curves at each bridge site, considering the NEHRP soil class at the bridge. Similar data need to be prepared for analyses of other regions. The hazard curves should follow the same format as those in this folder.

## Script: `weighted_scenario.py`

Given the package `pyRiskTable` and input folders(`bridge-info` and `USGS-data`), running `weighted_scenario.py` can generate risk tables for all ODOT-owned bridges and a summary file for the risks of all bridges.

## License

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

You are free to use, modify, and distribute this software, but if you do, **you must open-source your entire project** under the same license. This requirement applies even if you run the software over a network (e.g., as a web service) without distributing the executable.

See the [LICENSE](LICENSE) file for details.