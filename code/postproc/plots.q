\d .aml

//  calculate impact of each feature and save plot of top 20
/* bs   = best model name as a symbol
/* mdl  = best model as a fitted embedPy object or a kdb function
/* data = list containing test features and values
/* cnm  = column names for all columns being shuffled    
/* scf  = scoring function used to determine the best model
/* dt   = dictionary denoting the start time and date of a run
/* fp   = file path dictionaries with the full save path and subsection for printinga
post.featureimpact:{[bs;mdl;data;cnm;scf;dt;fp;p]
  im:post.i.imp[bs;$[b;flip;enlist]mdl;data;cnm;scf;p;]each $[b:2~count bs;01b;bs[0] in i.nlplist;1b;0b];
  post.i.impactplot[$[b;raze;]im;bs:$[b;`$"Comb_","_"sv string bs;bs[0]];dt;fp];
  -1"\nFeature impact calculated for features associated with ",string[bs]," model";
  -1 "Plots saved in ",i.ssrsv[fp[1][`images]],"\n";}
