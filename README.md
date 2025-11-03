This code belongs to and can be used to reproduce all figures from:

# Visuomotor decision-making through multifeature convergence in the larval zebrafish hindbrain #

Katja Slangewal, Sophie Aimon, Maxim Q. Capelle, Florian Kämpf, Heike Naumann, Herwig Baier, Krasimir Slanchev, Armin Bahl


BioRxiv DOI: https://doi.org/10.1101/2025.08.12.669772

---
The code and corresponding data is split per figure. 

Before starting, download the data here:
- Figure 1 and related supplementary figures (steady-state behavior):
    - DOI:
- Figure 2 and related supplementary figures (temporal dynamics of behavior and modeling): 
  - DOI:
  - DOI:
  - DOI:
  - DOI:
  - DOI:
- Figure 3 and related supplementary figures (functional imaging):
  - DOI:
  - DOI: 
  - DOI:
  - DOI:
  - DOI:
- Figure 4 and related supplementary figures (functional imaging and HCR-FISH):
  - DOI:
  - DOI:
- Figure 5 and related supplementary figures (functional imaging and paGFP photoactivations):
  - DOI:

*Important*: merge the subpackages (_part1, _part2 etc) into one folder per figure, 
thereby getting rid of the (_part1, _part2 etc) hierarchy. The subpackaging was necessary to limit the data-size per package.

---
Each figureX_xx.py file contains the code to reproduce figure X and related supplementary figures 
(as exception figureS1, figure S5 and figure S6, which have their own python file).

To reproduce the figure, Search for '# Provide' (at the start of the main code) and fill in the path to the downloaded (and merged per figure) 
data-folder, as well as the path where to save the figure(s). Then run the code. 

Each figureX_xx.py file contains all the functions needed to plot all panels and subpanels of the figure. Each function that plots any data, 
has 'This is related to Fig. Xx' written in the function description. This makes it easy to find relevant code for specific parts of the figure. 

The main part of the code below all the function definitions is structured as follows: 
- Path declarations
- Variable definitions
- Figure preparation (creation of figure object, including final size in cms of the figure)
- Subpanel outlines (preparation of each plot, setting e.g. the x- and y-limites, gray backgrounds spans, x- and y labels and axes ticks)
- Calling of the plotting functions to add the actual data. 
- Saving of the figures to a PDF file per figure. 

Besides the figureX_xx.py files there are a few util files:
- figure_helper.py contains all the plotting functions. This is a wrapper around matplotlib.pyplot to allow for more a more homogenous style and standard sizes across figures. 
- logic_regression_functions.py contains all the logical statement functions. 
- usefull_small_funcs.py contains a few util functions that are used by multiple figures. 

Any questions and/or comments on this code can be send to katja.slangewal(at)uni-konstanz.de or armin.bahl(at)uni-konstanz.de. 



