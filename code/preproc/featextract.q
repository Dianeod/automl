\d .automl

// For the following code the parameter naming convention holds
// defined here is applied to avoid repetition throughout the file
/* t   = input table
/* p   = parameter dictionary passed as default or modified by user
/* tgt = target data

// Create features using the FRESH algorithm
/. r > table of fresh created features and the time taken to complete extraction as a mixed list
prep.freshcreate:{[t;p]
  agg:p`aggcols;prm:get p`funcs;
  // Feature extraction should be performed on all columns that are non aggregate
  cols2use:k where not (k:cols[t])in agg;
  fe_start:.z.T;
  t:"f"$prep.i.nullencode[value .ml.fresh.createfeatures[t;agg;cols2use;prm];med];
  fe_end:.z.T-fe_start;
  t:.ml.infreplace t;
  (0^.ml.dropconstant t;fe_end)}


// In all cases feature significance currently returns the top 25% of important features
// if no features are deemed important it currently continues with all available features
// this is temporary approptiate action needs to be decided on.
/. r > table with only the significant features available or all features as above
prep.freshsignificance:{[t;tgt]
  $[0<>count k:.ml.fresh.significantfeatures[t;tgt;.ml.fresh.percentile 0.25];
    k;[-1 prep.i.freshsigerr;cols t]]}


// Create features for 'normal problems' -> one target for each row no time dependency
// or fresh like structure
/. r > table with features created in accordance with the normal feature creation procedure 
prep.normalcreate:{[t;p]
  fe_start:.z.T;
  // Time columns are extracted such that constituent parts can be used 
  // but are not transformed according to remaining procedures
  tcols:.ml.i.fndcols[t;"dmntvupz"];
  tb:(cols[t]except tcols)#t;
  tb:prep.i.applyfn/[tb;p`funcs];
  tb:.ml.dropconstant prep.i.nullencode[.ml.infreplace tb;med];
  // Apply the transform of time specific columns as appropriate
  if[0<count tcols;tb^:.ml.timesplit[tcols#t;::]];
  fe_end:.z.T-fe_start;
  (tb;fe_end)}

// Apply word2vec on string data for nlp problems
/. r > table with features created in accordance with the nlp feature creation procedure
prep.nlpcreate:{[t;p]
  fe_start:.z.T;
  r:i.nlp_proc[t;p;0b];
  tb:r 0;strcol:r 1;model:r 2;
  if[0<count cols[t] except strcol;tb:tb,'(prep.normalcreate[(strcol)_t;p])[0]];
  if[p[`saveopt]in 1 2;model[`:save][i.ssrwin[path,"/",p[`spath],"/models/w2v.model"]]];
  fe_end:.z.T-fe_start;
  (tb;fe_end)}

prep.nlppteate:{[t]
  fe_start:.z.T;
  strcol:.ml.i.fndcols[t;"C"];
  mat:$[1<count strcol;raze each flip t[strcol];[]raze t[strcol]];
  strnm:"_" sv string strcol;
  tb:([]strnm:mat);
  fe_end:.z.T-fe_start;
  (tb;fe_end)}
