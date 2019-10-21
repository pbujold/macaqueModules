# PhD_Analysis
------------------------------------
This is the psychometric analysis code for binary choice data I gathered in the Schultz Lab at Cambridge.  The code incorporates both the required analysis package (the jupyter notebooks containing draft figures and results are in separate repos). 
* Import **macaque** to get access to the following analysis modules:  
  * **f_toolbox**: common custom functions used throughout the package.
  * **f_trials**: module of functions that fetch trial dataframes from sqlite databases, gathers session information, and filter for errors. 
  * **f_choices**:  reorganise trial dataframes into grouped choice dataframes, also fetch choices with specific reward parameters.
  * **f_psychometrics**:  reorganise choice dataframes into psychometrically-arranged softmax dataframes, also fetch choices with specific reward parameters.
  * **f_probabilityDistortion**: functions specific to the analysis of probability distortions exhibited during binary choices. Provides higher order visualizations, statistics, and model fitting on the softmax dataframes
  * **f_sql**: functions required to deal with the sqlite dataframes used to store session data.
  * **f_reference**: functions specific to the analysis of trial history impacts on binary choicess. Provides higher order visualizations, statistics, and model fitting on the softmax dataframes
  * **f_utility**:  
  * **f_Rfunctions**: rpy2-specific functions used to fill in gaps in statsmodels capabilities. Examples include repeated-measures anovas and manovas.
  * **f_uncertainty**: module focused on calculating/visualising uncertainties in the data: confidence intervals, cross-validation...
* The R folder contains the needed rpy2 r-instance, with the addition of the afex and lsmeans libraries.  
* setup_packages.py will make sure the necessary packages are installed on your version of python
* For those interested in running the parameter fitting with Matlab's fminsearch rather than python's scipy.optimize.minimize, I have attached the appropriate .mat file (requires the matlab.engine module for python install)
------------------------------------
### Study:
## Probability distortion depends on choice sequence in rhesus monkeys
*Our lives are peppered with uncertain, probabilistic choices.
Recent studies showed dynamic subjective weightings of probability.
In the present study, we show that probability distortions in macaque monkeys differ significantly between sequences in which single gambles are repeated (S-shaped distortion), as opposed to being pseudorandomly intermixed with other gambles (inverse-S shaped distortion).
Our findings challenge the idea of fixed probability distortions resulting from inflexible computations, and point to a more instantaneous evaluation of probabilistic information.
A win-stay/lose-shift strategy appeared to drive the ‘gap’ between probability distortions in different contexts.
Our data suggest that probability values are slowly but constantly updated from prior experience – like in most adaptive system – driving measures of probability distortion to either side of the S/inverse-S debate.*
