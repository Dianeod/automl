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
  r:post.i.predshuff[bs;mdl;data;scf;;p`seed]each til count cnm;
  ord:proc.i.ord scf;
  im:post.i.impact[r;cnm;ord];
  post.i.impactplot[im;bs;dt;fp];
  -1"\nFeature impact calculated for features associated with ",string[bs]," model";
  -1 "Plots saved in ",fp[1][`images],"\n";}

post.combfeatureimpact:{[bs;mdl;data;cnm;scf;dt;fp;p]
  nlpdata:normdata:data;
  nlpdata[0]:nlpdata[0][;inlp:where 10h=type each first nlpdata[0]];
  nlpdata[2]:nlpdata[2][;inlp];
  normdata[0]:normdata[0][;inorm:where not 10h=type each first normdata[0]];
  normdata[2]:normdata[2][;inorm];
  r1:post.i.predshuff[bs[0];mdl[0;0],enlist last mdl;normdata;scf;;p`seed]each til count inorm;
  r2:post.i.predshuff[bs[1];mdl[0;1],enlist last mdl;nlpdata;scf;;p`seed]each til count inlp;
  ord:proc.i.ord scf; 
  im:post.i.impact[r1;cnm[inorm];ord],post.i.impact[r2;cnm[inlp];ord];
  post.i.impactplot[im;bs:`$"_"sv string bs;dt;fp];
  -1"\nFeature impact calculated for features associated with ",string[bs]," model";
  -1 "Plots saved in ",fp[1][`images],"\n";}

