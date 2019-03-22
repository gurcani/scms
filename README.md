# scms

 Spiral Chain Models
 ===================
 
 The basic model in this project solves the following equation for E<sub>n</sub>:
 
### &part;<sub>t</sub> E<sub>n</sub> = (g<sup>2m</sup> - g<sup>2l</sup>)t<sup>E</sup><sub>n+l</sub> + (1 - g<sup>2m</sup>)t<sup>E</sup><sub>n</sub> + (g<sup>2l</sup> - 1)t<sup>E</sup><sub>n-m+l</sub>

where:

### t<sup>E</sup><sub>n</sub> = g<sup>-l</sup>k<sub>n</sub>sin[(m-l)&alpha;]E<sub>n</sub><sup>3/2</sup> 

with the two dimensional wavenumber defined as a complex number using k<sub>n</sub> = k0 (g e<sup>i&alpha;</sup>)<sup>n</sup>

in general different values of l and m are possible as described in [X]. And the case that combines l=2,m=3 and l=1, m=5 and considering four such spiral chains (which has the best angular coverage) is included as fcm6E.py

Finally the model for the case l=1,m=3 for the complex amplitude &Phi;<sub>n</sub> is also given as scm3_13.py
