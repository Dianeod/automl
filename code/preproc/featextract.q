\d .aml

// For the following code the parameter naming convention holds
// defined here is applied to avoid repetition throughout the file
/* t   = input table
/* p   = parameter dictionary passed as default or modified by user
/* tgt = target data

// Create features using the FRESH algorithm
/. r > table of fresh created features and the time taken to complete extraction as a mixed list
prep.freshcreate:{[t;p]
  t:(`$ssr[;"_";""]each string cols t)xcol t;
  agg:p`aggcols;prm:get p`params;
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
  t:(`$ssr[;"_";""]each string cols t)xcol t;
  // Time columns are extracted such that constituent parts can be used 
  // but are not transformed according to remaining procedures
  tcols:.ml.i.fndcols[t;"dmntvupz"];
  tb:(cols[t]except tcols)#t;
  tb:prep.i.truncsvd[tb;::;2];
  tb:prep.i.bulktransform[tb;::;key prep.i.bulkname;1b];
  tb:.ml.dropconstant prep.i.nullencode[.ml.infreplace tb;med];
  // Apply the transform of time specific columns as appropriate
  if[0<count tcols;tb^:.ml.timesplit[tcols#t;::]];
  fe_end:.z.T-fe_start;
  (tb;fe_end)}

// Creat features for nlp problems using TFIDF
/. r > table with features created in accordance with the nlp feature creation procedure
prep.nlpcreate:{[t;p]
 fe_start:.z.T;
 tstr:.ml.i.fndcols[t;"C"];
 tb:prep.i.tfidf[`:fit_transform][raze t[tstr]][`:toarray][]`;
 tb:flip (`$prep.i.tfidf[`:get_feature_names][]`)!m:flip tb;
 system"mkdir -p ",p[`spath],"/nlp";
 joblib:.p.import[`joblib];
 joblib[`:dump][prep.i.tfidf;p[`spath],"/nlp/vectorize"];
 sp:.p.import[`spacy];
 dr:.p.import[`builtins][`:dir];
 pos:dr[sp[`:parts_of_speech]]`;
 unipos:`$pos[til (first where 0<count each pos ss\:"__")];
 myparser:.nlp.newParser[`en;`isStop`uniPOS];
 corpus:myparser raze t[tstr];
 tpos:{((y!(count y)#0f)),`float$(count each x)%count raze x}[;unipos]each group each corpus`uniPOS;
 sentt:.nlp.sentiment each raze t[tstr];
 tb:tb,'tpos,'sentt;
 tb[`isStop]:{sum[x]%count x}each corpus`isStop;
 tb:.ml.dropconstant prep.i.nullencode[.ml.infreplace tb;med];
 if[0<count cols[t] except tstr;tb:tb,'(prep.normalcreate[(tstr)_t;p])[0]];
 fe_end:.z.T-fe_start;
 (tb;fe_end)}

prep.nlppre:{[t;p]
 fe_start:.z.T;
 tstr:.ml.i.fndcols[t;"C"];
 sp:.p.import[`spacy];
 dr:.p.import[`builtins][`:dir];
 pos:dr[sp[`:parts_of_speech]]`;
 unipos:`$pos[til (first where 0<count each pos ss\:"__")];
 myparser:.nlp.newParser[`en;`isStop`uniPOS];
 corpus:myparser raze t[tstr];
 tpos:{((y!(count y)#0f)),`float$(count each x)%count raze x}[;unipos]each group each corpus`uniPOS;
 sentt:.nlp.sentiment each raze t[tstr];
 tb:tpos,'sentt;
 tb[`isStop]:{sum[x]%count x}each corpus`isStop;
 tb:.ml.dropconstant prep.i.nullencode[.ml.infreplace tb;med];
 tb:$[0<count cols[t] except tstr;t,'(prep.normalcreate[tb,'(tstr)_t;p])[0];t,'(prep.normalcreate[tb;p])[0]];
 fe_end:.z.T-fe_start;
 (tb;fe_end)}
