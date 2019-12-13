\d .aml

// The following is a naming convention used in this file
/* d = data as a mixed list containing training and testing data ((xtrn;ytrn);(xtst;ytst))
/* s = seed used for initialising the same model
/* o = one-hot encoding for multi-classification example
/* m = model object being passed through the system
/* mtype = type of model being used, i.e multi, binary, reg
/* dict = dictionary of parameters
/* mdl = name of bets model being used


fitscore:{[d;s;mtype]
  if[mtype~`multi;d[;1]:npa@'flip@'./:[;((::;0);(::;1))](0,count d[0]1)_/:value .ml.i.onehot1(,/)d[;1]];
  m:mdl[d;s;mtype];
  m:fit[d;m];
  get[".aml.",string[mtype],"predict"][d;m]}

nlpfitscore:{[d;dict;mdl]
 m:nlpmdl[dict;mdl];
 m:nlpfit[d;m];
 nlppredict[d;m]}

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

mdl:{[d;s;mtype]
 m:seq[];
 nps[s];
 m[`:add]dns[32;`activation pykw"relu";`input_dim pykw count first d[0]0];
 m[`:add]dns[$[mtype~`multi;count distinct (d[0]1)`;1];`activation pykw actdict[mtype]];
 m[`:compile][`loss pykw lossdict[mtype];`optimizer pykw "rmsprop"];m}

nlpmdl:{[dict;mdl]
 args:`overwrite_output_dir`use_multiprocessing`output_dir`cache_dir`silent`reprocess_input_data!(1b;0b;path,"/",(dict`spath),"/nlpmodel";path,"/outputs/cache_dir";1b;1b);
 args,:dict`args;
 if[not `cache_dir in key hsym `$path,"/outputs";system "mkdir -p ",path,"/outputs/cache_dir"];
 pydict:`model_type`model_name`use_cuda`num_labels`args!
        (lower mdl;nlpdict[mdl];0b;dict`tgtnum;args);
 m:get[".aml.nlp",string[dict`ptyp]][pykwargs pydict];m}

fit:{[d;m]m[`:fit][npa d[0]0;d[0]1;`batch_size pykw 32;`verbose pykw 0];m}

nlpfit:{[d;m]m[`:train_model][pdD((d[0]0),'enlist each d[0]1)];m}

// Prediction functions for each of the keras models
/* d = Data from which prediction is to be made
/* m = Fitted Keras model
/. r > predicted value
binarypredict  :{[d;m].5<raze m[`:predict][npa d[1]0]`}
multipredict:{[d;m]m[`:predict_classes][npa d[1]0]`}
regpredict  :{[d;m]raze m[`:predict][npa d[1]0]`}

binarypredictprob:{[d;m]flip(poscl;1-poscl:m[`:predict][npa d[1]0]`)}

nlppredict  :{[d;m]first m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`}
nlppredictprob  :{[d;m]last m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`}

npa:.p.import[`numpy]`:array;
seq:.p.import[`keras.models]`:Sequential;
dns:.p.import[`keras.layers]`:Dense;
nps:.p.import[`numpy.random][`:seed];
pdD:.p.import[`pandas]`:DataFrame;
nlpmulticlass:.p.import[`simpletransformers.classification]`:ClassificationModel;
nlpmultilabel:.p.import[`simpletransformers.classification]`:MultiLabelClassificationModel;

/if[0>system"s";.ml.mproc.init[abs system"s"]("\\l ",path,"/automl.q";"\\l ",path,"/code/")]
