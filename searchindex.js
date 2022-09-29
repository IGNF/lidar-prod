Search.setIndex({docnames:["apidoc/lidar_prod","apidoc/lidar_prod.commons","apidoc/lidar_prod.tasks","background/overview","background/production_process","background/thresholds_optimization_process","configs","guides/development","guides/thresholds_optimization","index","introduction","tutorials/install","tutorials/use"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["apidoc/lidar_prod.rst","apidoc/lidar_prod.commons.rst","apidoc/lidar_prod.tasks.rst","background/overview.md","background/production_process.md","background/thresholds_optimization_process.md","configs.rst","guides/development.md","guides/thresholds_optimization.md","index.rst","introduction.md","tutorials/install.md","tutorials/use.md"],objects:{"lidar_prod.commons":[[1,0,0,"-","commons"]],"lidar_prod.commons.commons":[[1,1,1,"","eval_time"],[1,1,1,"","extras"],[1,1,1,"","ignore_warnings"],[1,1,1,"","print_config"]],"lidar_prod.commons.commons.print_config.params":[[1,2,1,"","cfg_print_path"],[1,2,1,"","config"],[1,2,1,"","resolve"]],"lidar_prod.run":[[0,3,1,"","POSSIBLE_TASK"],[0,1,1,"","main"]],"lidar_prod.run.POSSIBLE_TASK":[[0,4,1,"","APPLY_BUILDING"],[0,4,1,"","CLEANING"],[0,4,1,"","ID_VEGETATION_UNCLASSIFIED"],[0,4,1,"","OPT_BUIlDING"],[0,4,1,"","OPT_UNCLASSIFIED"],[0,4,1,"","OPT_VEGETATION"]],"lidar_prod.tasks":[[2,0,0,"-","building_completion"],[2,0,0,"-","building_identification"],[2,0,0,"-","building_validation"],[2,0,0,"-","building_validation_optimization"],[2,0,0,"-","cleaning"],[2,0,0,"-","utils"]],"lidar_prod.tasks.building_completion":[[2,3,1,"","BuildingCompletor"]],"lidar_prod.tasks.building_completion.BuildingCompletor":[[2,5,1,"","prepare_for_building_completion"],[2,5,1,"","run"],[2,5,1,"","update_classification"]],"lidar_prod.tasks.building_completion.BuildingCompletor.run.params":[[2,2,1,"","input_values"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.building_identification":[[2,3,1,"","BuildingIdentifier"]],"lidar_prod.tasks.building_identification.BuildingIdentifier":[[2,5,1,"","prepare"],[2,5,1,"","run"]],"lidar_prod.tasks.building_identification.BuildingIdentifier.prepare.params":[[2,2,1,"","input_values"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.building_identification.BuildingIdentifier.run.params":[[2,2,1,"","input_values"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.building_validation":[[2,3,1,"","BuildingValidationClusterInfo"],[2,3,1,"","BuildingValidator"],[2,1,1,"","request_bd_uni_for_building_shapefile"],[2,3,1,"","thresholds"]],"lidar_prod.tasks.building_validation.BuildingValidationClusterInfo":[[2,4,1,"","entropies"],[2,4,1,"","overlays"],[2,4,1,"","probabilities"],[2,4,1,"","target"]],"lidar_prod.tasks.building_validation.BuildingValidator":[[2,5,1,"","prepare"],[2,5,1,"","run"],[2,5,1,"","setup"],[2,5,1,"","update"]],"lidar_prod.tasks.building_validation.BuildingValidator.run.params":[[2,2,1,"","input_values"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.building_validation.thresholds":[[2,4,1,"","min_confidence_confirmation"],[2,4,1,"","min_confidence_refutation"],[2,4,1,"","min_entropy_uncertainty"],[2,4,1,"","min_frac_confirmation"],[2,4,1,"","min_frac_confirmation_factor_if_bd_uni_overlay"],[2,4,1,"","min_frac_entropy_uncertain"],[2,4,1,"","min_frac_refutation"],[2,4,1,"","min_uni_db_overlay_frac"]],"lidar_prod.tasks.building_validation_optimization":[[2,3,1,"","BuildingValidationOptimizer"],[2,1,1,"","constraints_func"]],"lidar_prod.tasks.building_validation_optimization.BuildingValidationOptimizer":[[2,5,1,"","evaluate"],[2,5,1,"","evaluate_decisions"],[2,5,1,"","optimize"],[2,5,1,"","prepare"],[2,5,1,"","run"],[2,5,1,"","setup"],[2,5,1,"","update"]],"lidar_prod.tasks.building_validation_optimization.BuildingValidationOptimizer.evaluate_decisions.params":[[2,2,1,"","ia_decision"],[2,2,1,"","mts_gt"]],"lidar_prod.tasks.cleaning":[[2,3,1,"","Cleaner"]],"lidar_prod.tasks.cleaning.Cleaner":[[2,5,1,"","add_dimensions"],[2,5,1,"","get_extra_dims_as_str"],[2,5,1,"","remove_dimensions"],[2,5,1,"","run"]],"lidar_prod.tasks.cleaning.Cleaner.run.params":[[2,2,1,"","src_las_path"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.utils":[[2,3,1,"","BDUniConnectionParams"],[2,1,1,"","get_a_las_to_las_pdal_pipeline"],[2,1,1,"","get_integer_bbox"],[2,1,1,"","get_las_data_from_las"],[2,1,1,"","get_las_metadata"],[2,1,1,"","get_pdal_reader"],[2,1,1,"","get_pdal_writer"],[2,1,1,"","get_pipeline"],[2,1,1,"","pdal_read_las_array"],[2,1,1,"","save_las_data_to_las"],[2,1,1,"","split_idx_by_dim"]],"lidar_prod.tasks.utils.BDUniConnectionParams":[[2,4,1,"","bd_name"],[2,4,1,"","host"],[2,4,1,"","pwd"],[2,4,1,"","user"]],"lidar_prod.tasks.utils.get_a_las_to_las_pdal_pipeline.params":[[2,2,1,"","ops"],[2,2,1,"","src_las_path"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.utils.get_pdal_reader.params":[[2,2,1,"","las_path"]],"lidar_prod.tasks.utils.get_pdal_writer.params":[[2,2,1,"","extra_dims"],[2,2,1,"","target_las_path"]],"lidar_prod.tasks.utils.pdal_read_las_array.params":[[2,2,1,"","las_path"]],lidar_prod:[[0,0,0,"-","run"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","parameter","Python parameter"],"3":["py","class","Python class"],"4":["py","attribute","Python attribute"],"5":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:parameter","3":"py:class","4":"py:attribute","5":"py:method"},terms:{"0":[2,6],"05":6,"1":[2,6,8],"10":6,"100":6,"110":6,"111":6,"112":6,"113":6,"114":6,"115":6,"12345":6,"15km\u00b2":[4,8],"19":6,"2":2,"20":6,"200":6,"2002":8,"202":6,"208":6,"21":6,"214":6,"2455":6,"25":6,"28de":6,"3":6,"300":6,"35":6,"37574976186264664":6,"3923":6,"4":[2,6],"5":[2,4,6],"50":6,"5041941489707767":6,"5212204710813311":6,"5m":4,"6":6,"6286345539969987":6,"665mb":7,"6661410185371994":6,"75":2,"8":2,"8820746597006655":6,"9057081151270139":6,"91":4,"93":2,"95":6,"9613067360728214":6,"98":[4,5,6],"break":2,"case":7,"class":[0,2,3,9,10],"default":[4,9,12],"enum":0,"final":[2,5,6],"float":[2,6],"function":1,"int":2,"km\u00b2":2,"new":[2,7],"null":6,"public":[2,9,10,11],"return":2,"true":[1,2,6,8,12],"while":[9,10],A:[3,8,12],AND:4,As:[5,8],At:3,Be:8,By:12,For:[2,4,5,8,12],IF:2,If:2,In:[2,7],Its:[9,10],No:2,OR:[2,4],One:7,That:4,The:[2,3,4,5,7,9,10,11,12],Then:[2,11],There:5,These:5,Theses:8,To:[8,11,12],_args_:6,_target_:6,a_confirm:6,a_refut:6,abl:8,about:2,abov:4,absenc:7,accept:7,access:[2,7],accordingli:2,accur:2,accuraci:2,action:[7,12],activ:[7,8,11,12],actual:2,ad:7,adapt:2,add:[2,8,12],add_dimens:2,addit:12,additionnali:7,address:[9,10],aerial:4,after:5,against:4,ai:[2,5,9,10],ai_building_identifi:6,ai_building_proba:[6,12],ai_unclassified_proba:6,ai_vegetation_proba:6,ai_vegetation_unclassified_group:6,aim:[4,9,10],al:8,algorithm:[2,4,5,8,9,10],all:2,almost:8,alreadi:4,also:[2,5,8,9,10,11,12],alter:3,ambigu:[2,4],among:[2,4,5],an:[0,2,3,7,8,12],anaconda:11,ani:2,anoth:[3,5,8,12],anyth:2,anywher:11,app:[7,8,9],appli:0,applic:[2,6,9,10,11,12],apply_build:0,apply_on_build:[0,12],approach:[5,9,10],apt:11,ar:[2,3,4,5,6,7,8,9,10,12],area:[2,4,9,10],arrai:2,arrow:3,ascend:2,assign:2,associ:4,assum:2,attribut:2,augment:[9,10],auto_precision_recal:6,autom:[2,4,5,8,9,10],avail:7,avoid:2,balanc:[5,9,10],base:[0,2,3,4,5,8,9,10],basenam:8,bash:12,basi:2,basic_identif:6,bbox:2,bd:2,bd_name:[2,6],bd_param:2,bd_uni_connection_param:[2,6],bd_uni_request:[2,6],bdtopooverlai:6,bduni:[2,9,10],bduni_france_consult:6,bduniconnectionparam:[2,6],been:4,befor:8,being:[2,4,5,8,9,10],belong:4,best:[4,12],better:8,between:[5,9,10],bool:[1,2],both:2,both_confirm:6,both_unsur:6,branch:[7,11],buffer:[2,6],build:[2,3,6,7,9,10,12],building_complet:6,building_identif:6,building_valid:[6,8,12],building_validation_optim:6,building_validation_thresholds_pickl:[6,8],buildingcompletor:[2,6],buildingidentifi:[2,6],buildings_correction_label:[2,6,8],buildingvalid:[2,6],buildingvalidationclusterinfo:2,buildingvalidationoptim:[2,6],built:12,bulding_valid:8,c1:[4,5],c2:[4,5],c:2,calcul:4,callabl:1,can:[4,7,8,9,10,11,12],candid:[2,4,5,6],candidate_buildings_flag:6,capabl:8,captur:4,care:2,cd:11,cfg_print_path:1,channel:[2,8,12],check:0,choic:5,chosen:[4,5],cid_candidateb:6,cid_isolatedorconfirm:6,clasif:8,classif:[2,5,6,8,9,10],clean:[0,6,12],cleaner:[2,6],clone:11,cloud:[2,8,9,10,12],cluser:2,cluster:[2,4,5,6],cluster_id:6,clusterid:[2,6],clusterid_candidate_build:6,clusterid_isolated_plus_confirm:6,code:[2,6,8],cohes:2,color:12,column:2,com:11,come:3,command:8,common:9,compar:4,complet:2,compliant:5,compon:4,compos:1,comput:2,conda:[7,8,11,12],confid:[4,5],config:[0,1,6,12],config_tre:1,configur:[0,1,2,8,9,12],confirm:[2,4,5],confirmation_accuraci:6,confus:2,confusion_matrix_no_norm:6,confusion_matrix_norm:6,connect:[2,4],consid:[2,4],consist:8,constrain:5,constraint:[5,6],constraints_func:[2,6],consum:[2,3],content:[1,3,7,12],control:11,convent:7,copi:12,correct:[8,9,10,12],could:5,coupl:5,cr:4,creat:[2,11,12],create_studi:6,credenti:2,criteria:5,criterion:5,crossover_prob:6,current:[4,9,10,12],cwd:6,data:[2,5,8],data_format:[2,6,8,12],databas:[2,4,7,9,10,11],databs:2,dataclass:2,dataformat:2,dataset:[2,4,8,9,10],date:[7,12],db:12,db_overlayed_onli:6,deb:8,debug:[2,8,12],decid:3,decis:[2,4,9,10],decor:1,dedic:2,deep:[4,8,9,10],defin:[2,4,5,7],depend:[5,7,9,10,11],deriv:5,describ:[2,5],deseri:2,design:[2,6],destruct:2,detail:[2,6],detailed_to_fin:6,detect:[8,12],dev:7,develop:[8,9,11,12],dict:2,dictconfig:[0,1],dictionnari:2,differ:[2,5,8,9,10],dim:2,dim_arrai:2,dimens:[2,3,4,9,10,12],direct:[6,7],directli:11,directori:[2,8],distinguish:8,divers:8,docker:7,dockerfil:12,document:[7,12],doe:2,done:4,down:2,due:4,durat:1,dure:[8,11],e1:4,e2:4,e:[2,4,5,7,8],each:[2,4],edit:[4,11],effici:5,either:[0,2,7],element:[2,9,10],elitist:8,els:3,elsewis:4,empir:5,empti:2,encapsul:12,end:4,enough:[2,4],ensur:[9,10],entri:[0,9,10],entropi:[2,4,6,8,12],entry_valu:2,enumer:0,env:[11,12],environ:[7,8,12],equal:8,error:[5,9,10],establish:4,et:8,etc:4,eval_tim:1,evalu:[2,6,12],evaluate_decis:2,event:7,everi:2,exactli:4,exclud:2,exist:2,expect:2,explain:8,express:4,extens:[9,10],extern:3,extra:[1,2,4],extra_dim:[2,6],extra_dimens:2,extract:2,f_candidateb:6,factor:4,fall:2,fals:[2,6,8],false_neg:6,false_posit:6,fashion:4,fast:[7,8],faster:11,featur:7,few:2,field:1,file:[0,2,3,4,6,7,8,12],fill:2,filter:2,find:9,first:[3,4],flag:[2,5,8,12],fly:12,focus:5,folder:[6,8,12],follow:[2,7,8],forc:12,format:[2,8,12],former:2,forward:[2,7],found:[2,4,5,8,9,10],fr:6,from:[2,3,4,5,7,8,11],front:5,full:8,functionn:7,functool:6,furthermor:8,fuse:[9,10],g:[2,4,7],gener:[2,5,7,8],genet:[2,5,8,9,10],geograph:[9,10],geometr:4,get:[2,11],get_a_las_to_las_pdal_pipelin:2,get_extra_dims_as_str:2,get_integer_bbox:2,get_las_data_from_la:2,get_las_metadata:2,get_method:6,get_pdal_read:2,get_pdal_writ:2,get_pipelin:2,git:11,github:[7,11,12],give:8,glanc:6,goal:4,goe:3,good:2,ground:[2,4],group:[2,4,5,6,8],group_build:6,group_info:6,group_info_pickle_path:6,group_no_build:6,group_unsur:6,groups_count:6,guid:[8,12],h:12,ha:4,have:[2,3,4,8],help:8,here:6,high:[2,4,7,9,10],higher:[2,4,5],highli:5,highlight:[4,9,10],hoc:7,horizont:2,host:[2,6,7],how:[2,5,7,9],http:11,human:[4,9,10],hydra:[1,6,12],hyperparamet:[4,5],i:[5,8],ia_confirmed_onli:6,ia_decis:2,ia_refut:6,ia_refuted_but_under_db_uni:6,id:2,id_vegetation_unclassifi:0,identif:2,identifi:[2,4,12],identify_vegetation_unclassifi:[0,12],ign:6,ignf:11,ignor:2,ignore_warn:[1,6],ii:[5,8],imag:[7,12],implement:[2,7],impos:2,includ:[2,8],increas:5,index:[4,9],indic:[2,8],individu:11,infer:3,inform:[2,6,9,10],input:[2,4,6,12],input_build:6,input_las_dir:[6,8],input_valu:2,input_vegetation_unclassifi:6,inspect:4,instal:[8,9,12],instanc:[2,5],instead:12,integr:12,interest:2,intermediari:2,intern:2,introduc:5,invit:6,is3d:6,isol:[2,4,11],iter:2,its:[1,3,9,10],jointli:5,keep:[2,8],keyword:8,know:2,known:2,la:[2,3,4,6,8,12],label:[2,8],lamber:2,larg:[7,8],larger:8,las_data:2,las_dimens:[6,8,12],las_path:2,lasdata:2,laspi:2,later:2,learn:[4,8,9,10],level:[2,4,5,7,8],leverag:[4,7],librari:[1,5,7,9,10],lidar:[2,4,10,11],lidar_prod:[6,7,8,9,11,12],like:[2,12],line:8,lint:7,list:[2,12],live:8,load:2,local:7,local_output_dir:12,local_src_las_dir:12,log:1,logic:2,lower:5,m:[7,12],made:[5,8],mai:[2,4,5,8],main:[0,7,9,10],make:[2,4,7],mamba:11,manag:[6,11],mark:4,match:[7,8,12],matric:2,matrix:2,maxim:[4,5,6],md:[2,7],mean:[2,7,9,10],meet:[4,5],merg:[2,7],method:1,metric:[2,6],metric_nam:2,metric_valu:2,might:[5,8,9,10],min_automation_constraint:6,min_building_proba:[2,6],min_building_proba_relaxation_if_bd_uni_overlai:[2,6],min_confidence_confirm:[2,6],min_confidence_refut:[2,6],min_entropy_uncertainti:[2,6],min_frac:6,min_frac_confirm:[2,6],min_frac_confirmation_factor_if_bd_uni_overlai:[2,6],min_frac_entropy_uncertain:[2,6],min_frac_refut:[2,6],min_point:6,min_precision_constraint:6,min_recall_constraint:6,min_uni_db_overlay_frac:[2,6],minim:[9,10],minut:2,miss:[4,9,10],mode:[8,11],model:[4,5,8,9,10],modul:[2,3,8,9],moment:7,more:[6,8],most:2,mostli:12,mts_gt:2,much:[4,5],multi:[4,5,9,10],multiclass:[9,10],multiobject:8,must:[5,8],mutation_prob:6,myria3d:3,n:2,n_trial:6,name:[2,4,7,12],nc:2,ndarrai:2,necessari:2,need:[2,7,8,9,10,11,12],neg:8,network:[9,10],neural:[9,10],nevertheless:4,newli:4,non:[2,7],none:[1,2],not_build:6,note:8,now:[9,10],np:2,nr:2,nsga:[5,8],nsgaiisampl:6,nu:2,number:[2,5],numpi:2,o1:4,object:[2,4,5,9,10],occur:2,omegaconf:[0,1],onc:[2,8],one:[2,5,9,10,12],ones:2,onli:[2,9,10],op:2,oper:2,opt_build:0,opt_unclassifi:0,opt_veget:0,optim:[0,2,4,6,9,10],optimize_build:[0,8,12],optimize_unc_id:[0,12],optimize_veg_id:[0,12],optimized_threshold:[6,8],option:[1,2,12],optuna:[2,5,6],order:2,origin:8,other:[2,4,9,10,12],otpim:8,our:[9,10],out:2,output:[2,5,6,12],output_build:6,output_dir:[6,12],output_vegetation_unclassifi:6,overlai:[2,4],overrid:[2,12],overriden:12,overview:9,p:[2,4],p_auto:6,p_confirm:6,p_refut:6,p_unsur:6,packag:[4,7,11,12],page:[7,11],param:8,paramet:[1,2,8,12],pareto:5,part:[2,12],partial:6,particular:8,pass:7,path:[2,6,8,12],pdal:2,pdal_read_las_arrai:2,per:2,percentag:[4,5],perfect:2,perform:[2,4,8],pickl:[2,6,8],pip:11,pipe:2,pipelin:2,plan:4,plane:4,point:[0,2,3,5,8,9,10,12],pool:8,population_s:6,posit:8,possibl:4,possible_task:0,posterior:2,postgi:11,potenti:[2,4],precis:[2,4,5,6],predict:[2,8,9,10,12],prepar:[2,6,8,12],prepare_for_building_complet:2,prepared_las_dir:6,prepared_las_path:2,presenc:2,preserv:2,pretti:12,previou:[2,4,8],previous:[2,4],print:[1,12],print_config:[1,6,12],probabl:[2,3,4,5,8,9,10,12],process:[2,5,7,8,9],prod:[2,7,10,11],produc:[4,5,9,10],product:[2,5,7,8,9,10,11],project:11,proport:[2,4,5],proportion_of_automated_decis:6,proportion_of_confirm:6,proportion_of_refut:6,proportion_of_uncertainti:6,provid:[8,11,12],publicli:7,pull:7,push:7,pwd:[2,6],py:[7,8,9,10,12],pytest:7,python:[2,7,8],qualiti:[2,9,10,11],quantiti:5,queri:[7,12],r1:4,r2:4,r:2,ran:2,raw:3,reach:8,read:2,reader:2,readi:7,readm:2,recal:[2,4,5,6],reduc:[4,5],refactor:7,refer:[1,2,6,8,12],refert:11,refut:[2,4],refutation_accuraci:6,relax:4,releas:7,remain:4,remov:2,remove_dimens:2,repositori:12,repres:[3,5],request:[2,7,11],request_bd_uni_for_building_shapefil:2,requir:[4,5,7,9,10,12],rerun:2,reset:2,resolv:1,respect:[4,8],result:[2,7,8,12],results_output_dir:[2,6,8],resum:2,rich:1,right:[9,10],robust:8,role:7,rule:[2,3,4,5,8,9,10],run:[2,7,9,11],runner:7,runtim:6,s:[9,12],safeguard:4,said:8,same:[2,4,8],sampler:6,save:[1,2,8,12],save_las_data_to_la:2,save_result:2,schema:2,script:11,search:[2,5],second:3,section:5,see:[2,7,11,12],seed:6,segment:[3,4,9,10],select:2,self:[2,7],semant:[4,7,9,10],sens:8,sent:3,sequenc:2,serial:2,serveurbdudiff:6,set:[2,4,5],setup:[2,7,11],setup_env:[11,12],sever:[2,3,5],sh:[11,12],shape:2,shapefil:[2,12],shapefile_path:2,share:2,should:12,show:6,shp_path:[2,6,12],simpl:[9,10],simpli:12,singl:8,so:[2,7,8,9,10],solut:[5,8],some:[2,4],someth:3,sourc:[0,1,2,5,6,9,10,11],specif:[2,12],specifi:[2,8,12],split_idx_by_dim:2,spot:[9,10],sr:2,src_la:[6,12],src_las_basenam:12,src_las_path:2,standalon:12,standard:2,step:[2,3,4,7,8],store:2,str:[1,2],strategi:[8,9,10],string:8,stringifi:2,structur:1,studi:[2,6],study_nam:6,subfold:8,sudo:11,suppos:4,sure:[2,8],surfac:4,surround:2,swapping_prob:6,syntax:12,tag:7,take:[9,10],taken:2,tarbal:11,target:2,target_las_path:2,task:[6,8,9],task_nam:12,termin:2,test:[2,4],thei:[3,5,8,12],them:[2,4],therefor:[4,5],thes:12,thi:[2,3,4,7,8,12],those:[2,4,5,12],three:5,threshod:2,threshold:[0,2,4,6,9,10],through:[3,4],time:[2,3],todo:[2,6,8],togeth:[2,4,8,9,10],toler:[4,6],too:[2,4],tool:4,top:11,total:2,track:8,train:[8,9,10],transform:[2,9],tree:1,trial:2,true_posit:6,truth:2,tutori:[7,8],two:[2,5,8],txt:1,type:2,typic:2,u:2,uc:2,uint32:6,uint:6,uncertain:[2,4],uncertainti:[4,9,10],unclassifi:[3,6,12],unclassified_nb_tri:6,unclassified_threshold:6,unclust:6,under:[2,8],uni:[2,12],uni_db_overlai:6,union:2,unseen:8,unsur:[2,4,5,6],unsure_by_entropi:6,unwant:2,up:[2,4,7,8,12],updat:[2,4,6,8,9,10],update_classif:2,updated_las_dir:6,upgrad:11,ur:2,url:2,us:[1,2,3,5,7,8,9,10,11],usag:[0,11],use_final_classification_cod:[2,6],user:[2,6],util:6,uu:2,v:12,val:8,valid:[2,9],valu:[0,2],variabl:2,variou:3,vector:[2,4,9,10],veget:[3,6,12],vegetation_high:6,vegetation_low:6,vegetation_medium:6,vegetation_nb_tri:6,vegetation_target:6,vegetation_threshold:6,vertic:[2,4],via:[2,4,9,10,12],virtual:12,volum:8,vuildingvalid:2,wa:[2,5,7,8],wall:4,want:[8,12],we:[2,3,4,5,6,8,9,10,11],well:[2,4,5],went:4,were:[2,4,5,8,9,10],when:[4,7],where:[1,3,4],whether:1,which:[2,3,4,5,7,8,9,10,11],whose:12,without:5,workflow:7,write:2,writer:2,www:11,xfail:7,xy:[2,4],y:2,yaml:12,yc:2,ye:2,yn:2,you:[4,8,11,12],your:[8,11],yr:2,yu:2},titles:["lidar_prod.run","lidar_prod.commons","lidar_prod.tasks","Overview of the process","Production process used to transform point clouds classification","Strategy to find optimal decision thresholds for Building validation","Default configuration","Developer\u2019s guide","How to optimize building validation decision thresholds?","Lidar-Prod &gt; Documentation","&lt;no title&gt;","Installation","Using the app"],titleterms:{"1":4,"2":4,"3":4,"default":6,A:4,app:[11,12],b:4,background:9,build:[4,5,8],building_complet:2,building_identif:2,building_valid:2,building_validation_optim:2,cd:7,ci:7,classif:4,clean:2,cloud:4,code:7,common:1,complet:4,configur:6,contain:12,continu:7,decis:[5,8],deliveri:7,detect:4,develop:7,differ:12,directli:12,docker:12,document:9,environ:11,evalu:8,find:[5,8],from:12,get:9,guid:[7,9],how:8,identif:4,indic:9,instal:11,integr:7,lidar:9,lidar_prod:[0,1,2],modul:[1,11,12],motiv:5,optim:[5,8],overal:3,overview:3,packag:9,point:4,process:[3,4],prod:9,product:4,python:[11,12],refer:9,requir:8,run:[0,8,12],s:7,schema:3,set:[8,11],sourc:12,start:9,strategi:5,tabl:9,task:[2,12],test:[7,8],threshold:[5,8],transform:4,unclassifi:4,up:11,us:[4,12],util:[2,8],valid:[4,5,8],veget:4,version:7,virtual:11,within:12}})