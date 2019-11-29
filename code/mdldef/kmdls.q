\d .aml

binfitscore:{
 npa:.p.import[`numpy]`:array;
 seq:.p.import[`keras.models]`:Sequential;
 dns:.p.import[`keras.layers]`:Dense;
 nps:.p.import[`numpy.random][`:seed];
 m:seq[];
 nps[y];
 m[`:add]dns[32;`activation pykw"relu";`input_dim pykw count first x[0]0];
 m[`:add]dns[1;`activation pykw"sigmoid"];
 m[`:compile][`loss pykw"binary_crossentropy";`optimizer pykw"rmsprop"];
 m[`:fit][npa x[0]0;x[0]1;`batch_size pykw 32;`verbose pykw 0];
 .5<raze m[`:predict][npa x[1]0]`}

multifitscore:{
 npa:.p.import[`numpy]`:array;
 seq:.p.import[`keras.models]`:Sequential;
 dns:.p.import[`keras.layers]`:Dense;
 nps:.p.import[`numpy.random][`:seed];
 l:`ytrn`ytst!flip@'./:[;((::;0);(::;1))](0,count x[0]1)_/:value .ml.i.onehot1(,/)x[;1];
 m:seq[];
 nps[y];
 m[`:add]dns[32;`activation pykw"relu";`input_dim pykw count first x[0]0];
 m[`:add]dns[count distinct x[0]1;`activation pykw"softmax"];
 m[`:compile][`loss pykw"categorical_crossentropy";`optimizer pykw"rmsprop"];
 m[`:fit][npa x[0]0;npa l`ytrn;`batch_size pykw 32;`verbose pykw 0];
 m[`:predict_classes][npa x[1]0]`}

regfitscore:{
 npa:.p.import[`numpy]`:array;
 seq:.p.import[`keras.models]`:Sequential;
 dns:.p.import[`keras.layers]`:Dense;
 nps:.p.import[`numpy.random][`:seed];
 m:seq[];
 nps[y];
 m[`:add]dns[32;`activation pykw"relu";`input_dim pykw count first x[0]0];
 m[`:add]dns[1;`activation pykw"relu"];
 m[`:compile][`optimizer pykw"rmsprop";`loss pykw"mse"];
 m[`:fit][npa x[0]0;x[0]1;`batch_size pykw 32;`verbose pykw 0];
 raze m[`:predict][npa x[1]0]`}
 
nlpmulticlassfitscore:{
 classmod:.p.import[`simpletransformers.classification]`:ClassificationModel;
 pdD:.p.import[`pandas]`:DataFrame;
 args:`overwrite_output_dir`use_multiprocessing!(10b);
 args:`output_dir!p`model_path;
 args,:y`args;
 model:classmod[y`model_type;y`model_name;`use_cuda pykw 0b;`args pykw args];
 model[`:train_model][pdD ((x[0]0),'x[0]1)];
 evals:model[`:eval_model][pdD x[1]0;(y`scf) pykw get[y`scf]]`;
 (first evals)y`scf
 }

nlpmultilabelfitscore:{
 classmod:.p.import[`simpletransformers.classification]`:MultiLabelClassificationModel;
 pdD:.p.import[`pandas]`:DataFrame;
 args:`overwrite_output_dir`use_multiprocessing!(10b);
 args:`output_dir!p`model_path;
 args,:y`args;
 model:classmod[y`model_type; y`model_name;`use_cuda pykw 0b;`args pykw args];
 model[`:train_model][pdD (enlist each x[0]0),'enlist each x[0]1];
 evals:model[`:eval_model][pdD x[1]0;(y`scf) pykw get[y`scf]]`;
 (first evals)y`scf
 }

