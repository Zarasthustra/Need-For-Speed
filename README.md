# Need-For-Speed
Test sets for Molecular simulation algorithms

Currently only lennard-jones interactions.

Very soon will be EWALD summation and particle-mesh ewald. I have my own python and Fortran code to verify energy calculations for ewald energy, but none to verify force calculations. 

NIST has a set of energy test sets https://www.nist.gov/mml/csd/chemical-informatics-research-group/spce-water-reference-calculations-10%C3%A5-cutoff

my real code can read in configurations, thus avoiding the need for fake coordinates and interaction parameters. I may add it to these test sets.
