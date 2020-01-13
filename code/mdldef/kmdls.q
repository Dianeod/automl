\d .aml

// The following is a naming convention used in this file
/* d = data as a mixed list containing training and testing data ((xtrn;ytrn);(xtst;ytst))
/* s = seed used for initialising the same model
/* o = one-hot encoding for multi-classification example
/* m = model object being passed through the system
/* dict = dictionary of parameters
/* mdl = dictionary of model being used

fitscore:{[d;dict;mdls;b]
 if[`multi~mtype:mdls`typ;d[;1]:npa@'flip@'./:[;((::;0);(::;1))](0,count d[0]1)_/:value .ml.i.onehot1(,/)d[;1]]
 fncn:$[mtyp:mdls[`model] in i.nlplist;".aml.nlp";".aml."];
 m:get[`$fncn,"mdl"][dict;mdls;d];
 m:get[`$fncn,"fit"][d;m];
 get[`$fncn,string[mdls[`typ]],$[b;"predict";"predictprob"]][d;m]}

actdict:`binary`reg`multi!("sigmoid";"relu";"softmax")
lossdict:`binary`reg`multi!("binary_crossentropy";"mse";"categorical_crossentropy")

nlpdict:(!). flip(
 (`Bert;         "bert-base-uncased");
 (`RoBERTa;      "roberta-base");
 (`XLNet;        "xlnet-base-cased");
 (`XLM ;         "xlm-mlm-en-2048");
 (`DistilBERT;   "distilbert-base-uncased-distilled-squad");
 (`ALBERT  ;     "albert-base-v1");
 (`CamemBERT;    "camembert-base"))

mdl:{[dict;mdls;d]
 nps[s:dict`seed];
 if[not 1~checkimport[];tfs[s]];
 m:seq[];
 m[`:add]dns[32;`activation pykw"relu";`input_dim pykw count first d[0]0];
 m[`:add]dns[$[mtype~`multi;count distinct (d[0]1)`;1];`activation pykw actdict[mtype:mdls`typ]];
 m[`:compile][`loss pykw lossdict[mtype];`optimizer pykw "rmsprop"];m}


nlpmdl:{[dict;mdls;d]
 args:`overwrite_output_dir`use_multiprocessing`output_dir`cache_dir`silent`reprocess_input_data!
  (1b;0b;path,"/",dict[`spath],"/nlpmodel/",string[lower mdln:mdls`model];(cachedr:path,"/",dict[`spath]),"/cache_dir";1b;1b);
 args,:dict`args;
 modeln:$[`best_model in key dict;path,"/",(dict`spath),"/nlpmodel/",string[lower mdln];nlpdict[mdln]];
 pydict:`model_type`model_name`use_cuda`num_labels`args!
        (lower mdln;modeln;0b;dict`tgtnum;args);
 m:get[".aml.nlp",string[dict`ptyp]][pykwargs pydict];m}


fit:{[d;m]m[`:fit][npa d[0]0;d[0]1;`batch_size pykw 32;`verbose pykw 0];m}

nlpfit:{[d;m]m[`:train_model][pdD((d[0]0),'enlist each d[0]1)];m}

// Prediction functions for each of the keras models
/* d = Data from which prediction is to be made 
/*     formatting based on master workflow ((0n;0n);(xtst;0n))
/. r > predicted values
binarypredict  :{[d;m].5<raze m[`:predict][npa d[1]0]`}
multipredict:{[d;m]m[`:predict_classes][npa d[1]0]`}
regpredict  :{[d;m]raze m[`:predict][npa d[1]0]`}

binarypredictprob:{[d;m]flip(poscl;1-poscl:m[`:predict][npa d[1]0]`)}

nlppredict  :{[d;m]first m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`}
nlppredictprob  :{[d;m]sm:.p.import[`scipy.special]`:softmax;
                       sm[last m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`;`axis pykw 1]`}

nlpclasspredict  :{[d;m]first m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`}
nlpclasspredictprob  :{[d;m]sm:.p.import[`scipy.special]`:softmax;
                       sm[last m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`;`axis pykw 1]`}

npa:.p.import[`numpy]`:array;
seq:.p.import[`keras.models]`:Sequential;
dns:.p.import[`keras.layers]`:Dense;
nps:.p.import[`numpy.random][`:seed];
pdD:.p.import[`pandas]`:DataFrame;
nlpclass:.p.import[`simpletransformers.classification]`:ClassificationModel;
nlpmultilabel:.p.import[`simpletransformers.classification]`:MultiLabelClassificationModel;
if[not 1~checkimport[];tf:.p.import[`tensorflow];tfs:tf$[2>"I"$first tf[`:__version__]`;[`:random.set_random_seed];[`:random.set_seed]]];
