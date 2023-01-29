#include "brif.h"
#include <Rcpp.h>
#include <R_ext/Print.h>


using namespace Rcpp;

//' Train a random forest
//' 
//' This function is not intended for end users. Users should use the brif.formula or brif.default function. 
//' 
//' @param rdf a data frame. The first column is treated as the target variable. 
//' @param par a list containing all parameters. 
//' @return a list, of class "brif", containing the trained random forest model.
// [[Rcpp::export]]
List rftrain(DataFrame rdf, List par){
  //List res = List::create();
  int nthreads = par["nthreads"];
  int verbose = par["verbose"];
  int ps = par["ps"];
  int max_depth = par["max_depth"];
  int min_node_size = par["min_node_size"];
  int ntrees = par["ntrees"];
  int bagging_method = par["bagging_method"];
  double bagging_proportion = par["bagging_proportion"];
  int split_search = par["split_search"];
  int search_radius = par["search_radius"];
  //int seed = par["seed"];
  int p = rdf.length() - 1;
  int n = rdf.nrow();
  char *var_types = (char*)malloc((p+1)*sizeof(char));
  char **var_labels = (char**)malloc((p+1)*sizeof(char*));
  CharacterVector labels = rdf.names();
  Function cl("class");
  for(int j = 0; j < rdf.length(); j++){
    //res.push_back(cl(rdf[i]));
    //std::string str1 = cl(rdf[j]);
    String str1 = cl(rdf[j]);
    String this_label = labels[j];
    var_labels[j] = (char*)this_label.get_cstring();
    if(str1 == "numeric"){
      var_types[j] = 'n';
    } else if(str1 == "factor"){
      var_types[j] = 'f';
    } else if(str1 == "integer"){
      var_types[j] = 'i';
    }
    //printf("%s | %c \n", var_labels[j], var_types[j]);
  }
  
  void **data = (void**)malloc((p+1)*sizeof(void*));
  
  for(int j = 0; j <= p; j++){
    data[j] = NULL;  // initialize
  }
  for(int j = 0; j <= p; j++){
    if(var_types[j] == 'f'){
      // construct the fnode tree that encodes levels to values in the same order as in R
      // then directly use fac_vec as the index vector
      // after done, must delete the factor content appropriately
      IntegerVector fac_vec = rdf[j];
      CharacterVector fac_levels = fac_vec.attr("levels");
      data[j] = create_factor(0);
      // R integer index starts from 1 not 0
      ((factor_t*)data[j])->start_index = 1;
      for(int i = 0; i < fac_levels.length(); i++){
        String this_level = fac_levels[i];
        char *temp_name = (char*) this_level.get_cstring();
        //add_element((factor_t*)tmp_factor, i, (char*)temp_name);
        //printf("%s \n", temp_name);
        insert_node(&(((factor_t*)data[j])->levels), temp_name, i + ((factor_t*)data[j])->start_index);  
      }
      // re-assign its elements
      ((factor_t*)data[j])->nlevels = fac_levels.length();
      ((factor_t*)data[j])->n = n;
      ((factor_t*)data[j])->index = fac_vec.begin();
       
    } else if(var_types[j] == 'n'){
      NumericVector num_vec = rdf[j];
      data[j] = (numeric_t *) num_vec.begin();
    } else if(var_types[j] == 'i'){
      IntegerVector int_vec = rdf[j];
      data[j] = (integer_t *) int_vec.begin();
    }
  
  }
  
  data_frame_t * train_df = (data_frame_t*) malloc(sizeof(data_frame_t));
  train_df->p = p;
  train_df->n = n;
  train_df->var_labels = var_labels;
  train_df->var_types = var_types;
  train_df->data = data;
  
  rf_model_t *model = create_empty_model();
  model->p = p;
  model->var_types = (char*)malloc((p+1)*sizeof(char));
  memcpy(model->var_types, var_types, (p+1)*sizeof(char));
  
  model->var_labels = (char**)malloc((p+1)*sizeof(char*));
  for(int j = 0; j <= p; j++){
    model->var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
    strncpy(model->var_labels[j], var_labels[j], MAX_VAR_NAME_LEN-1);
  }
  
  int n_numeric_cuts = par["n_numeric_cuts"];
  int n_integer_cuts = par["n_integer_cuts"];
  int max_integer_classes = par["max_integer_classes"];

  
  make_cuts(train_df, &model, n_numeric_cuts, n_integer_cuts); 
  
  bx_info_t *bx_train = make_bx(train_df, &model, nthreads);
  ycode_t *yc_train = make_yc(train_df, &model, max_integer_classes, nthreads);
  
  
  if(ps == 0) ps = (int)(round(sqrt(model->p)));
  build_forest(bx_train, yc_train, &model, ps, max_depth, min_node_size, ntrees, nthreads, bagging_method, bagging_proportion, split_search, search_radius);
  if(verbose){
    Rprintf("Tree 0 printout:\n");
    printTree(model->trees[0], 0, model->yc->nlevels);

    Rprintf("Target coding:\n");
    if(model->yc->ycuts_int != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %d\n", model->yc->level_names[c], model->yc->ycuts_int[c]);
        } else {
          Rprintf("%d\n", model->yc->ycuts_int[c]);
        }
      }
    } else if(model->yc->ycuts_num != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %0.4f\n", model->yc->level_names[c], model->yc->ycuts_num[c]);
        } else {
          Rprintf("%0.4f\n", model->yc->ycuts_num[c]);
        }
      }
    } else if(model->yc->yvalues_int != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %d\n", model->yc->level_names[c], model->yc->yvalues_int[c]);
        } else {
          Rprintf("%d\n", model->yc->yvalues_int[c]);
        }
      }
    } else if(model->yc->yvalues_num != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %0.4f\n", model->yc->level_names[c], model->yc->yvalues_num[c]);
        } else {
          Rprintf("%0.4f\n", model->yc->yvalues_num[c]);
        }
      }
    }

    Rprintf("Split points:\n");
    int n_num_vars = 0;
    int n_int_vars = 0;
    int n_fac_vars = 0;
    for(int j = 1; j <= p; j++){
      Rprintf("%d | %s | %c | %d split points: \n", j, model->var_labels[j], model->var_types[j], model->n_bcols[j]);
      if(model->var_types[j] == 'n'){
        for(int c = 0; c < model->n_bcols[j]; c++){
          Rprintf("%0.4f ", model->numeric_cuts[n_num_vars][c]);
        }
        Rprintf("\n");
        n_num_vars++;
      } else if(model->var_types[j] == 'i'){
        for(int c = 0; c < model->n_bcols[j]; c++){
          Rprintf("%d ", model->integer_cuts[n_int_vars][c]);
        }
        Rprintf("\n");
        n_int_vars++;        
      } else if(model->var_types[j] == 'f'){
        Rprintf("%d levels.\n", model->factor_cuts[n_fac_vars]->nlevels);
        n_fac_vars++;          
      }
    }
  }
   
  flatten_model(&model, nthreads);

  if(verbose >= 2){
    Rprintf("Tree 0 rules:\n");
    printRules(model, 0);
  }
  
  delete_bx(bx_train, model);
  delete_yc(yc_train);

  // save model information
  
  CharacterVector model_var_types((p+1));
  IntegerVector model_n_bcols((p+1));
  IntegerVector model_index_in_group((p+1));
  for(int j = 0; j <= p; j++){
    if(var_types[j] == 'n') model_var_types[j] = "numeric";
    if(var_types[j] == 'f') model_var_types[j] = "factor";
    if(var_types[j] == 'i') model_var_types[j] = "integer";
    model_n_bcols[j] = model->n_bcols[j];
    model_index_in_group[j] = model->index_in_group[j];
  }
  
  List num_cuts = List::create();
  List int_cuts = List::create();
  List fac_cuts = List::create();
  int this_num_var = 0;
  int this_int_var = 0;
  int this_fac_var = 0;
  
  
  for(int j = 1; j <= p; j++){
    if(model->var_types[j] == 'n'){
      if(model->numeric_cuts[this_num_var] != NULL){
        NumericVector num_vec(model->n_bcols[j]);
        for(int c = 0; c < model->n_bcols[j]; c++){
          num_vec[c] = model->numeric_cuts[this_num_var][c];
        }      
        num_cuts.push_back(num_vec);   
      } else {
        NumericVector num_vec(0);
        num_cuts.push_back(num_vec);
      }
      this_num_var += 1;
    } else if(model->var_types[j] == 'i'){
      
      if(model->integer_cuts[this_int_var] != NULL){
        IntegerVector int_vec(model->n_bcols[j]);
        for(int c = 0; c < model->n_bcols[j]; c++){
          int_vec[c] = model->integer_cuts[this_int_var][c];
        }      
        int_cuts.push_back(int_vec);   
      } else {
        IntegerVector int_vec(0);
        int_cuts.push_back(int_vec);
      }
      this_int_var += 1;
    } 
    else if(model->var_types[j] == 'f'){
      if(model->factor_cuts[this_fac_var] != NULL){
        factor_t * f = model->factor_cuts[this_fac_var];
        char** level_array = (char**)malloc(model->n_bcols[j]*sizeof(char*));
        for(int c = 0; c < model->n_bcols[j]; c++){
          level_array[c] = (char*)malloc(MAX_LEVEL_NAME_LEN*sizeof(char));
        }
        fill_name_array(f->levels, level_array, 1);  // start index in R is 1
        
        CharacterVector fac_levels_vec(model->n_bcols[j]);
        for(int c = 0; c < model->n_bcols[j]; c++){
          fac_levels_vec[c] = level_array[c];
        }      
        fac_cuts.push_back(fac_levels_vec);   
        for(int c = 0; c < model->n_bcols[j]; c++){
          free(level_array[c]);
        }
        free(level_array);
      } else {
        CharacterVector fac_levels_vec(0);
        fac_cuts.push_back(fac_levels_vec);
      }
      this_fac_var += 1;
    }
  }
   
  List model_leaves = List::create();
  int J = model->yc->nlevels;

  for(int t = 0; t < model->ntrees; t++){
    dt_leaf_t * this_leaves = model->tree_leaves[t];
    List this_flat_tree = List::create();
    //copy_leaves(this_leaves, &this_flat_tree, model->yc->nlevels);
    //printf("model->yc->nlevels = %d\n", model->yc->nlevels);
    while(this_leaves != NULL){
      //copy_leaves_no_recursive(this_leaves, this_flat_tree, model->yc->nlevels);
      
      IntegerVector count(J);
      for(int i = 0; i < J; i++){
        count[i] = this_leaves->count[i];
      }
      //printf("this_leaves->depth = %d\n", this_leaves->depth);
      IntegerVector rulepath_var(this_leaves->depth);
      IntegerVector rulepath_bx(this_leaves->depth);
      for(int i = 0; i < this_leaves->depth; i++){
        rulepath_var[i] = this_leaves->rulepath_var[i];
        rulepath_bx[i] = this_leaves->rulepath_bx[i];
      }
      List this_leaf = List::create(Named("ct") = count,
                                    Named("dp") = this_leaves->depth,
                                    Named("va") = rulepath_var,
                                    Named("bx") = rulepath_bx);
      this_flat_tree.push_back(this_leaf);
      this_leaves = this_leaves->next;
    }
    
    model_leaves.push_back(this_flat_tree);
  }


  NumericVector yavg;
  NumericVector yvalues_num;
  IntegerVector yvalues_int;
  NumericVector ycuts_num;
  IntegerVector ycuts_int;
  CharacterVector level_names;
  for(int i = 0; i < model->yc->nlevels; i++){
    if(model->yc->yavg != NULL){
      yavg.push_back(model->yc->yavg[i]);
    }
    if(model->yc->ycuts_int != NULL){
      ycuts_int.push_back(model->yc->ycuts_int[i]);
    }
    if(model->yc->ycuts_num != NULL){
      ycuts_num.push_back(model->yc->ycuts_num[i]);
    }
    if(model->yc->yvalues_int != NULL){
      yvalues_int.push_back(model->yc->yvalues_int[i]);
    }
    if(model->yc->yvalues_num != NULL){
      yvalues_num.push_back(model->yc->yvalues_num[i]);
    }
    if(model->yc->level_names != NULL){
      level_names.push_back((model->yc->level_names[i]));
    }
  }

  
  List model_yc = List::create(Named("nlevels") = model->yc->nlevels,
                               Named("type") = model->yc->type,
                               Named("start_index") = model->yc->start_index,
                               Named("yavg") = yavg,
                               Named("yvalues_int") = yvalues_int,
                               Named("yvalues_num") = yvalues_num,
                               Named("ycuts_int") = ycuts_int,
                               Named("ycuts_num") = ycuts_num,
                               Named("level_names") = level_names);
  
  
  List res = List::create(Named("p") = p,
                          Named("var_types") = model_var_types,
                          Named("var_labels") = labels,
                          Named("n_bcols") = model_n_bcols,
                          Named("ntrees") = ntrees,
                          Named("index_in_group") = model_index_in_group,
                          Named("numeric_cuts") = num_cuts,
                          Named("integer_cuts") = int_cuts,
                          Named("factor_cuts") = fac_cuts,
                          Named("n_num_vars") = model->n_num_vars,
                          Named("n_int_vars") = model->n_int_vars,
                          Named("n_fac_vars") = model->n_fac_vars,
                          Named("tree_leaves") = model_leaves,
                          Named("yc") = model_yc
                          );
  
  
  delete_model(model);
  free(train_df);
   
   
  
  for(int j = 0; j <= p; j++){
    if(var_types[j] == 'f'){
      // manually delete 
      factor_t * f = (factor_t *)data[j];
      delete_tree(f->levels);
      free(data[j]);
    }
  }
   
  free(data);

  free(var_types);
  free(var_labels);
  String model_class = "brif";
  res.attr("class") = model_class;
  return(res);

}

// do not free count because it points to R
void deleteLeaves2(dt_leaf_t* root){
  if(root != NULL){
    dt_leaf_t * this_leaf = root;
    root = root->next;
    free(this_leaf);
    deleteLeaves2(root);        
  }
}

//' Predict new cases
//' 
//' This function is not intended for end users. Users should use the predict.brif function instead.
//' 
//' @param rf an object of class 'brif', as returned by rftrain.
//' @param rdf a data frame containing the new cases to be predicted.
//' @param vote_method an integer (0 or 1) indicating the voting mechanism among leaf predictions.
//' @param nthreads an integer specifying the number of threads to be used in prediction.
//' @return a data frame containing the predicted values.
// [[Rcpp::export]]
DataFrame rfpredict(List rf, DataFrame rdf, int vote_method, int nthreads){
  DataFrame result;
  if(strcmp(rf.attr("class"),"brif")){
    Rprintf("Model is not a brif object.\n");
    return(result);
  }
  // unpack rf
  int p = rf["p"];
  CharacterVector rf_var_types = rf["var_types"];
  CharacterVector rf_var_labels = rf["var_labels"];
  IntegerVector rf_n_bcols = rf["n_bcols"];
  IntegerVector rf_index_in_group = rf["index_in_group"];
  int rf_ntrees = rf["ntrees"];
  List rf_numeric_cuts = rf["numeric_cuts"];
  List rf_integer_cuts = rf["integer_cuts"];
  List rf_factor_cuts = rf["factor_cuts"];
  int rf_n_num_vars = rf["n_num_vars"];
  int rf_n_int_vars = rf["n_int_vars"];
  int rf_n_fac_vars = rf["n_fac_vars"];
  List rf_tree_leaves = rf["tree_leaves"];
  List rf_yc = rf["yc"];
  
  
  // construct model
  rf_model_t * model = (rf_model_t*)malloc(sizeof(rf_model_t));
  char* var_types = (char*)malloc((p+1)*sizeof(char));
  char** var_labels = (char**)malloc((p+1)*sizeof(char*));
  for(int j = 0; j <= p; j++){
    var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
    strncpy(var_labels[j], rf_var_labels[j], MAX_VAR_NAME_LEN-1);
    if(rf_var_types[j] == "numeric"){
      var_types[j] = 'n';
    } else if (rf_var_types[j] == "integer"){
      var_types[j] = 'i';
    } else if (rf_var_types[j] == "factor"){
      var_types[j] = 'f';
    } 
    //printf("%s | %c \n", var_labels[j], var_types[j]);
  }
  model->p = p;
  model->ntrees = rf_ntrees;
  model->n_num_vars = rf_n_num_vars;
  model->n_int_vars = rf_n_int_vars;
  model->n_fac_vars = rf_n_fac_vars;
  model->var_types = var_types;
  model->var_labels = var_labels;
  model->n_bcols = rf_n_bcols.begin();
  model->index_in_group = rf_index_in_group.begin();
  model->integer_cuts = (integer_t**)malloc(rf_n_int_vars*sizeof(integer_t*));
  model->factor_cuts = (factor_t **)malloc(rf_n_fac_vars*sizeof(factor_t*));
  model->numeric_cuts = (numeric_t**)malloc(rf_n_num_vars*sizeof(numeric_t*));
  model->trees = NULL;
  model->tree_leaves = NULL;
  model->yc = NULL;
  
  //printf("%d %d %d\n", rf_n_int_vars, rf_n_num_vars, rf_n_fac_vars);
  
  for(int j = 0; j < rf_n_int_vars; j++){
    IntegerVector this_j_int_cuts = rf_integer_cuts[j];
    if(this_j_int_cuts.length() > 0){
      model->integer_cuts[j] = (integer_t *)this_j_int_cuts.begin();
    } else {
      model->integer_cuts[j] = NULL;
    }
  }
   
  for(int j = 0; j < rf_n_num_vars; j++){
    NumericVector this_j_num_cuts = rf_numeric_cuts[j];
    if(this_j_num_cuts.length() > 0){
      model->numeric_cuts[j] = (numeric_t *)this_j_num_cuts.begin();
    } else {
      model->numeric_cuts[j] = NULL;
    }
  }
  
  for(int j = 0; j < rf_n_fac_vars; j++){
    CharacterVector this_j_fac_levels = rf_factor_cuts[j];
    //CharacterVector this_j_fac_levels = this_j_fac_cut_list[0];
    //printf("length of rf_factor_cuts[%d] = %ld\n", j, this_j_fac_levels.length());
    if(this_j_fac_levels.length() > 0){
      factor_t * this_factor = create_factor(0);
      this_factor->nlevels = this_j_fac_levels.length();
      fnode_t * level_tree = NULL;  // starting point for insert_node
      for(int c = 0; c < this_j_fac_levels.length(); c++){
        insert_node(&level_tree, this_j_fac_levels[c], c + this_factor->start_index);
      }
      this_factor->levels = level_tree;
      model->factor_cuts[j] = this_factor;
    } else {
      model->factor_cuts[j] = NULL;
    }
  }
  
  // construct model->tree_leaves
  model->tree_leaves = (dt_leaf_t **)malloc(rf_ntrees*sizeof(dt_leaf_t*));
  for(int t = 0; t < rf_ntrees; t++){
    model->tree_leaves[t] = NULL;
    List rf_this_leaves = rf_tree_leaves[t];
    int this_n_leaves = rf_this_leaves.length();
    
    for(int i = 0; i < this_n_leaves; i++){
      List rf_this_leaf = rf_this_leaves[i];
      IntegerVector rf_this_leaf_count = rf_this_leaf["ct"];
      IntegerVector rf_this_leaf_rulepath_var = rf_this_leaf["va"];
      IntegerVector rf_this_leaf_rulepath_bx = rf_this_leaf["bx"];

      dt_leaf_t * new_leaf = (dt_leaf_t *)malloc(sizeof(dt_leaf_t));
      new_leaf->count = rf_this_leaf_count.begin();
      new_leaf->depth = rf_this_leaf["dp"];
      for(int d = 0; d < new_leaf->depth; d++){
        new_leaf->rulepath_var[d] = rf_this_leaf_rulepath_var[d];
        new_leaf->rulepath_bx[d] = rf_this_leaf_rulepath_bx[d];
      }
      new_leaf->next = model->tree_leaves[t];
      model->tree_leaves[t] = new_leaf;
    }
  }
  
  
   
  // construct model->yc
  NumericVector rf_yavg = rf_yc["yavg"];
  NumericVector rf_yvalues_num = rf_yc["yvalues_num"];
  IntegerVector rf_yvalues_int = rf_yc["yvalues_int"];
  NumericVector rf_ycuts_num = rf_yc["ycuts_num"];
  IntegerVector rf_ycuts_int = rf_yc["ycuts_int"];
  CharacterVector rf_level_names = rf_yc["level_names"];
  int rf_yc_start_index = rf_yc["start_index"];
  int rf_yc_nlevels = rf_yc["nlevels"];
  int rf_yc_type = rf_yc["type"];
  
  model->yc = (ycode_t *)malloc(sizeof(ycode_t));
  model->yc->nlevels = rf_yc_nlevels;
  model->yc->type = rf_yc_type;
  model->yc->start_index = rf_yc_start_index;
  model->yc->ymat = NULL;
  
  if(rf_yavg.length() > 0){
    model->yc->yavg = (numeric_t*)malloc(model->yc->nlevels*sizeof(numeric_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->yavg[i] = rf_yavg[i];
    }
  } else {
    model->yc->yavg = NULL;
  }
  if(rf_yvalues_int.length() > 0){
    model->yc->yvalues_int = (integer_t*)malloc(model->yc->nlevels*sizeof(integer_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->yvalues_int[i] = rf_yvalues_int[i];
    }
  } else {
    model->yc->yvalues_int = NULL;
  }
  if(rf_yvalues_num.length() > 0){
    model->yc->yvalues_num = (numeric_t*)malloc(model->yc->nlevels*sizeof(numeric_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->yvalues_num[i] = rf_yvalues_num[i];
    }
  } else {
    model->yc->yvalues_num = NULL;
  }
  if(rf_ycuts_int.length() > 0){
    model->yc->ycuts_int = (integer_t*)malloc(model->yc->nlevels*sizeof(integer_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->ycuts_int[i] = rf_ycuts_int[i];
    }
  } else {
    model->yc->ycuts_int = NULL;
  }
  if(rf_ycuts_num.length() > 0){
    model->yc->ycuts_num = (numeric_t*)malloc(model->yc->nlevels*sizeof(numeric_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->ycuts_num[i] = rf_ycuts_num[i];
    }
  } else {
    model->yc->ycuts_num = NULL;
  }
  if(rf_level_names.length() > 0){
    model->yc->level_names = (char**)malloc(model->yc->nlevels*sizeof(char*));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->level_names[i] = (char*)malloc(MAX_LEVEL_NAME_LEN*sizeof(char));
      strncpy(model->yc->level_names[i], rf_level_names[i], MAX_LEVEL_NAME_LEN-1);
    }
  } else {
    model->yc->level_names = NULL;
  }
  
  int n = rdf.nrow();
  

  
  // construct test data frame
  CharacterVector labels = rdf.names();
  Function cl("class");
  IntegerVector col_index(p+1);  // contains the column index of each needed variable in rdf
  int match_ok = 1;
  for(int j = 1; j <= p; j++){
    int found = 0;
    for(int k = 0; k < rdf.length(); k++){
      // both match in name and type
      CharacterVector str1 = cl(rdf[k]);
      //printf("labels[k] = %s, rf_var_labels[j] = %s\n", (char*)labels[k], (char*)rf_var_labels[j]);
      //printf("cl(rdf[k]) = %s, rf_var_tyes[j] = %s\n", (char*)str1[0], (char*)rf_var_types[j]);
      if(labels[k] == rf_var_labels[j] && str1[0] == rf_var_types[j]){
        found = 1;
        col_index[j] = k;
        break;
      }
    }
    if(found == 0){
      //printf("Cannot find %s variable %s in newdata. \n", (char*)rf_var_types[j], (char*)rf_var_labels[j]);
      match_ok = 0;
      break;
    }
  }
  //printf("%d \n", match_ok);
  
  if(!match_ok){
    //printf("Abort.\n");
    // Clean up
    for(int j = 0; j <= p; j++){
      free(model->var_labels[j]);
    }
    free(model->var_labels);
    
    for(int j = 0; j < rf_n_fac_vars; j++){
      if(model->factor_cuts[j] != NULL){
        delete_factor(model->factor_cuts[j]);
      }
    }
    
    for(int t = 0; t < model->ntrees; t++){
      deleteLeaves2(model->tree_leaves[t]);
    }
    free(model->tree_leaves);
    delete_yc(model->yc);
    free(model->var_types);
    free(model->integer_cuts);
    free(model->factor_cuts);
    free(model->numeric_cuts);
    free(model);
    return(result);
  }
 
  void **data = (void**)malloc((p+1)*sizeof(void*));
  for(int j = 0; j <= p; j++){
    data[j] = NULL;  // initialize
  }
  
  for(int j = 1; j <= p; j++){
    if(var_types[j] == 'f'){
      IntegerVector fac_vec = rdf[col_index[j]];
      CharacterVector fac_levels = fac_vec.attr("levels");
      int which_factor = model->index_in_group[j];
      //printf("which_factor = %d\n", which_factor);
      data[j] = copy_factor(0, model->factor_cuts[which_factor]);
      if(data[j] != NULL){
        ((factor_t*) data[j])->n = n;
        ((factor_t*) data[j])->index = (integer_t *) fac_vec.begin();
      }
    } else if(var_types[j] == 'n'){
      NumericVector num_vec = rdf[col_index[j]];
      data[j] = (numeric_t *) num_vec.begin();
    } else if(var_types[j] == 'i'){
      IntegerVector int_vec = rdf[col_index[j]];
      data[j] = (integer_t *) int_vec.begin();
    }
  }
  
  
  
  data_frame_t * test_df = (data_frame_t*) malloc(sizeof(data_frame_t));
  test_df->p = p;
  test_df->n = n;
  test_df->var_labels = var_labels;
  test_df->var_types = var_types;
  test_df->data = data;
  
  // prepare score matrix
  double **score = (double**)malloc(model->yc->nlevels*sizeof(double*));
  if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num == NULL){
    for(int k = 0; k < model->yc->nlevels; k++){
      NumericVector this_k(n);
      char var_name[MAX_VAR_NAME_LEN];
      if(model->yc->yvalues_int != NULL){
        if(model->var_types[0] == 'f'){  // if the target variable is a factor
          int this_level_index = model->yc->yvalues_int[k] - model->yc->start_index;
          snprintf(var_name,MAX_VAR_NAME_LEN, "%s", model->yc->level_names[this_level_index]);
        } else {
          snprintf(var_name,MAX_VAR_NAME_LEN, "%d", model->yc->yvalues_int[k]);
        }
      } else if(model->yc->yvalues_num != NULL){
        //snprintf(var_name,MAX_VAR_NAME_LEN, "%f", model->yc->yvalues_num[k]);
      }
      score[k] = (double *)this_k.begin();
      result.push_back(this_k, var_name);
    }
  } else {
    for(int k = 0; k < model->yc->nlevels; k++){
      NumericVector this_k(n);
      score[k] = (double *)this_k.begin();
    }    
  }
  
  bx_info_t *bx_test = make_bx(test_df, &model, nthreads);
  predict(model, bx_test, score, vote_method, nthreads);
  delete_bx(bx_test, model);

  if(model->yc->type == REGRESSION){
    NumericVector pred(n);
    for(int i = 0; i < n; i++){
      pred[i] = 0;
      //double normalizing_factor = 0;
      for(int k = 0; k < model->yc->nlevels; k++){
        pred[i] += model->yc->yavg[k]*score[k][i];
        //pred[i] += model->yc->yavg[k]*(score[k][i])*(score[k][i]);
        //normalizing_factor += (score[k][i])*(score[k][i]);
      }
      //pred[i] = pred[i] / normalizing_factor;
    }
    result.push_back(pred,"pred");
  } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num != NULL){
    NumericVector pred(n);
    for(int i = 0; i < n; i++){
      pred[i] = 0;
      //double normalizing_factor = 0;
      for(int k = 0; k < model->yc->nlevels; k++){
        pred[i] += model->yc->yvalues_num[k]*score[k][i];
        //pred[i] += model->yc->yvalues_num[k]*(score[k][i])*(score[k][i]);
        //normalizing_factor += (score[k][i])*(score[k][i]);
      }
      //pred[i] = pred[i] / normalizing_factor;
    }
    result.push_back(pred,"pred");    
  }
  
  free(score);
  
  
  for(int j = 0; j <= p; j++){
    if(model->var_types[j] == 'f'){
      if(data[j] != NULL){
        delete_tree(((factor_t *)data[j])->levels);
        free(data[j]);
      }
    }
  }
  free(data);
  free(test_df);
  
  // manually delete model (some components belong to R, not to be freed here)
  for(int j = 0; j <= p; j++){
    free(model->var_labels[j]);
  }
  free(model->var_labels);

  for(int j = 0; j < rf_n_fac_vars; j++){
      if(model->factor_cuts[j] != NULL){
        delete_factor(model->factor_cuts[j]);
      }
  }

  for(int t = 0; t < model->ntrees; t++){
    deleteLeaves2(model->tree_leaves[t]);
  }
  free(model->tree_leaves);
  delete_yc(model->yc);
  free(model->var_types);
  free(model->integer_cuts);
  free(model->factor_cuts);
  free(model->numeric_cuts);
  free(model);
    
  return(result);
}


//' Print the decision rules of a Brif tree
//' 
//' @param rf an object of class 'brif', as returned by rftrain.
//' @param which_tree an integer indicating the tree number 
//' @return No return value. The function is intended for producing a side effect, which prints the decision rules to the standard output.  
// [[Rcpp::export]]
void printBrifTree(List rf, int which_tree){
  if(strcmp(rf.attr("class"),"brif")){
    Rprintf("Model is not a brif object.\n");
    return;
  }
  // unpack rf
  int p = rf["p"];
  CharacterVector rf_var_types = rf["var_types"];
  CharacterVector rf_var_labels = rf["var_labels"];
  IntegerVector rf_n_bcols = rf["n_bcols"];
  IntegerVector rf_index_in_group = rf["index_in_group"];
  int rf_ntrees = rf["ntrees"];
  List rf_numeric_cuts = rf["numeric_cuts"];
  List rf_integer_cuts = rf["integer_cuts"];
  List rf_factor_cuts = rf["factor_cuts"];
  int rf_n_num_vars = rf["n_num_vars"];
  int rf_n_int_vars = rf["n_int_vars"];
  int rf_n_fac_vars = rf["n_fac_vars"];
  List rf_tree_leaves = rf["tree_leaves"];
  List rf_yc = rf["yc"];
  
  // construct model
  rf_model_t * model = (rf_model_t*)malloc(sizeof(rf_model_t));
  char* var_types = (char*)malloc((p+1)*sizeof(char));
  char** var_labels = (char**)malloc((p+1)*sizeof(char*));
  for(int j = 0; j <= p; j++){
    var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
    strncpy(var_labels[j], rf_var_labels[j], MAX_VAR_NAME_LEN-1);
    if(rf_var_types[j] == "numeric"){
      var_types[j] = 'n';
    } else if (rf_var_types[j] == "integer"){
      var_types[j] = 'i';
    } else if (rf_var_types[j] == "factor"){
      var_types[j] = 'f';
    } 
    //printf("%s | %c \n", var_labels[j], var_types[j]);
  }
  model->p = p;
  model->ntrees = rf_ntrees;
  model->n_num_vars = rf_n_num_vars;
  model->n_int_vars = rf_n_int_vars;
  model->n_fac_vars = rf_n_fac_vars;
  model->var_types = var_types;
  model->var_labels = var_labels;
  model->n_bcols = rf_n_bcols.begin();
  model->index_in_group = rf_index_in_group.begin();
  model->integer_cuts = (integer_t**)malloc(rf_n_int_vars*sizeof(integer_t*));
  model->factor_cuts = (factor_t **)malloc(rf_n_fac_vars*sizeof(factor_t*));
  model->numeric_cuts = (numeric_t**)malloc(rf_n_num_vars*sizeof(numeric_t*));
  model->trees = NULL;
  model->tree_leaves = NULL;
  model->yc = NULL;
  
  for(int j = 0; j < rf_n_int_vars; j++){
    IntegerVector this_j_int_cuts = rf_integer_cuts[j];
    if(this_j_int_cuts.length() > 0){
      model->integer_cuts[j] = (integer_t *)this_j_int_cuts.begin();
    } else {
      model->integer_cuts[j] = NULL;
    }
  }
   
  for(int j = 0; j < rf_n_num_vars; j++){
    NumericVector this_j_num_cuts = rf_numeric_cuts[j];
    if(this_j_num_cuts.length() > 0){
      model->numeric_cuts[j] = (numeric_t *)this_j_num_cuts.begin();
    } else {
      model->numeric_cuts[j] = NULL;
    }
  }
  
  for(int j = 0; j < rf_n_fac_vars; j++){
    CharacterVector this_j_fac_levels = rf_factor_cuts[j];
    //CharacterVector this_j_fac_levels = this_j_fac_cut_list[0];
    //printf("length of rf_factor_cuts[%d] = %ld\n", j, this_j_fac_levels.length());
    if(this_j_fac_levels.length() > 0){
      factor_t * this_factor = create_factor(0);
      this_factor->nlevels = this_j_fac_levels.length();
      fnode_t * level_tree = NULL;  // starting point for insert_node
      for(int c = 0; c < this_j_fac_levels.length(); c++){
        insert_node(&level_tree, this_j_fac_levels[c], c + this_factor->start_index);
      }
      this_factor->levels = level_tree;
      model->factor_cuts[j] = this_factor;
    } else {
      model->factor_cuts[j] = NULL;
    }
  }
  
  // construct model->tree_leaves
  model->tree_leaves = (dt_leaf_t **)malloc(rf_ntrees*sizeof(dt_leaf_t*));
  for(int t = 0; t < rf_ntrees; t++){
    model->tree_leaves[t] = NULL;
    List rf_this_leaves = rf_tree_leaves[t];
    int this_n_leaves = rf_this_leaves.length();
    
    for(int i = 0; i < this_n_leaves; i++){
      List rf_this_leaf = rf_this_leaves[i];
      IntegerVector rf_this_leaf_count = rf_this_leaf["ct"];
      IntegerVector rf_this_leaf_rulepath_var = rf_this_leaf["va"];
      IntegerVector rf_this_leaf_rulepath_bx = rf_this_leaf["bx"];

      dt_leaf_t * new_leaf = (dt_leaf_t *)malloc(sizeof(dt_leaf_t));
      new_leaf->count = rf_this_leaf_count.begin();
      new_leaf->depth = rf_this_leaf["dp"];
      for(int d = 0; d < new_leaf->depth; d++){
        new_leaf->rulepath_var[d] = rf_this_leaf_rulepath_var[d];
        new_leaf->rulepath_bx[d] = rf_this_leaf_rulepath_bx[d];
      }
      new_leaf->next = model->tree_leaves[t];
      model->tree_leaves[t] = new_leaf;
    }
  }
  
  
   
  // construct model->yc
  NumericVector rf_yavg = rf_yc["yavg"];
  NumericVector rf_yvalues_num = rf_yc["yvalues_num"];
  IntegerVector rf_yvalues_int = rf_yc["yvalues_int"];
  NumericVector rf_ycuts_num = rf_yc["ycuts_num"];
  IntegerVector rf_ycuts_int = rf_yc["ycuts_int"];
  CharacterVector rf_level_names = rf_yc["level_names"];
  int rf_yc_start_index = rf_yc["start_index"];
  int rf_yc_nlevels = rf_yc["nlevels"];
  int rf_yc_type = rf_yc["type"];
  
  model->yc = (ycode_t *)malloc(sizeof(ycode_t));
  model->yc->nlevels = rf_yc_nlevels;
  model->yc->type = rf_yc_type;
  model->yc->start_index = rf_yc_start_index;
  model->yc->ymat = NULL;
  
  if(rf_yavg.length() > 0){
    model->yc->yavg = (numeric_t*)malloc(model->yc->nlevels*sizeof(numeric_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->yavg[i] = rf_yavg[i];
    }
  } else {
    model->yc->yavg = NULL;
  }
  if(rf_yvalues_int.length() > 0){
    model->yc->yvalues_int = (integer_t*)malloc(model->yc->nlevels*sizeof(integer_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->yvalues_int[i] = rf_yvalues_int[i];
    }
  } else {
    model->yc->yvalues_int = NULL;
  }
  if(rf_yvalues_num.length() > 0){
    model->yc->yvalues_num = (numeric_t*)malloc(model->yc->nlevels*sizeof(numeric_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->yvalues_num[i] = rf_yvalues_num[i];
    }
  } else {
    model->yc->yvalues_num = NULL;
  }
  if(rf_ycuts_int.length() > 0){
    model->yc->ycuts_int = (integer_t*)malloc(model->yc->nlevels*sizeof(integer_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->ycuts_int[i] = rf_ycuts_int[i];
    }
  } else {
    model->yc->ycuts_int = NULL;
  }
  if(rf_ycuts_num.length() > 0){
    model->yc->ycuts_num = (numeric_t*)malloc(model->yc->nlevels*sizeof(numeric_t));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->ycuts_num[i] = rf_ycuts_num[i];
    }
  } else {
    model->yc->ycuts_num = NULL;
  }
  if(rf_level_names.length() > 0){
    model->yc->level_names = (char**)malloc(model->yc->nlevels*sizeof(char*));
    for(int i = 0; i < model->yc->nlevels; i++){
      model->yc->level_names[i] = (char*)malloc(MAX_LEVEL_NAME_LEN*sizeof(char));
      strncpy(model->yc->level_names[i], rf_level_names[i], MAX_LEVEL_NAME_LEN-1);
    }
  } else {
    model->yc->level_names = NULL;
  }
  
  printRules(model, which_tree);
  
  // manually delete model (some components belong to R, not to be freed here)
  for(int j = 0; j <= p; j++){
    free(model->var_labels[j]);
  }
  free(model->var_labels);

  for(int j = 0; j < rf_n_fac_vars; j++){
      if(model->factor_cuts[j] != NULL){
        delete_factor(model->factor_cuts[j]);
      }
  }

  for(int t = 0; t < model->ntrees; t++){
    deleteLeaves2(model->tree_leaves[t]);
  }
  free(model->tree_leaves);
  delete_yc(model->yc);
  free(model->var_types);
  free(model->integer_cuts);
  free(model->factor_cuts);
  free(model->numeric_cuts);
  free(model);
  return;
}


//' Train a model and predict for newdata in one go
//' 
//' This function is not intended for end users. Users should use the function brif or brif.trainpredict and supply the newdata argument thereof. 
//' @param rdf a data frame containing the training data.
//' @param rdf_new a data frame containing new cases to be predicted.
//' @param par a list containing all parameters.
//' @return a data frame containing the predicted values.
// [[Rcpp::export]]
DataFrame rftrainpredict(DataFrame rdf, DataFrame rdf_new, List par){
  DataFrame result;
  int nthreads = par["nthreads"];
  int verbose = par["verbose"];
  int ps = par["ps"];
  int max_depth = par["max_depth"];
  int min_node_size = par["min_node_size"];
  //int seed = par["seed"];
  int ntrees = par["ntrees"];
  int bagging_method = par["bagging_method"];
  double bagging_proportion = par["bagging_proportion"];
  int vote_method = par["vote_method"];
  int split_search = par["split_search"];
  int search_radius = par["search_radius"];
  int p = rdf.length() - 1;
  int n = rdf.nrow();

  CharacterVector labels = rdf.names();
  Function cl("class");
  // check if rdf_new contains the needed variables and matching types
  CharacterVector labels_new = rdf_new.names();
  IntegerVector col_index(p+1);  // to contain the column index of each needed variable in rdf_new
  int match_ok = 1;
  for(int j = 1; j <= p; j++){
    int found = 0;
    CharacterVector olddata_type = cl(rdf[j]);
    for(int k = 0; k < rdf_new.length(); k++){
      // both match in name and type
      CharacterVector newdata_type = cl(rdf_new[k]);
      //printf("labels[k] = %s, rf_var_labels[j] = %s\n", (char*)labels[k], (char*)rf_var_labels[j]);
      //printf("cl(rdf[k]) = %s, rf_var_tyes[j] = %s\n", (char*)str1[0], (char*)rf_var_types[j]);
      if(labels_new[k] == labels[j] && newdata_type[0] == olddata_type[0]){
        found = 1;
        col_index[j] = k;
        break;
      }
    }
    if(found == 0){
      Rprintf("Cannot find %s variable %s in newdata. \n", (char*)olddata_type[0], (char*)labels[j]);
      match_ok = 0;
      break;
    }
  }
  
  if(!match_ok){
    return(result);
  }


  char *var_types = (char*)malloc((p+1)*sizeof(char));
  char **var_labels = (char**)malloc((p+1)*sizeof(char*));
  for(int j = 0; j < rdf.length(); j++){
    //res.push_back(cl(rdf[i]));
    //std::string str1 = cl(rdf[j]);
    String str1 = cl(rdf[j]);
    String this_label = labels[j];
    var_labels[j] = (char*)this_label.get_cstring();
    if(str1 == "numeric"){
      var_types[j] = 'n';
    } else if(str1 == "factor"){
      var_types[j] = 'f';
    } else if(str1 == "integer"){
      var_types[j] = 'i';
    }
    //Rprintf("%s | %c \n", var_labels[j], var_types[j]);
  }

  // construct training data frame
  void **data = (void**)malloc((p+1)*sizeof(void*));
  
  for(int j = 0; j <= p; j++){
    data[j] = NULL;  // initialize
  }
  for(int j = 0; j <= p; j++){
    if(var_types[j] == 'f'){
      // construct the fnode tree that encodes levels to values in the same order as in R
      // then directly use fac_vec as the index vector
      // after done, must delete the factor content appropriately
      IntegerVector fac_vec = rdf[j];
      CharacterVector fac_levels = fac_vec.attr("levels");
      data[j] = create_factor(0);
      // R integer index starts from 1 not 0
      ((factor_t*)data[j])->start_index = 1;
      for(int i = 0; i < fac_levels.length(); i++){
        String this_level = fac_levels[i];
        char *temp_name = (char*) this_level.get_cstring();
        //add_element((factor_t*)tmp_factor, i, (char*)temp_name);
        //printf("%s \n", temp_name);
        insert_node(&(((factor_t*)data[j])->levels), temp_name, i + ((factor_t*)data[j])->start_index);  
      }
      // re-assign its elements
      ((factor_t*)data[j])->nlevels = fac_levels.length();
      ((factor_t*)data[j])->n = n;
      ((factor_t*)data[j])->index = fac_vec.begin();
       
    } else if(var_types[j] == 'n'){
      NumericVector num_vec = rdf[j];
      data[j] = (numeric_t *) num_vec.begin();
    } else if(var_types[j] == 'i'){
      IntegerVector int_vec = rdf[j];
      data[j] = (integer_t *) int_vec.begin();
    }
  
  }
  
  data_frame_t * train_df = (data_frame_t*) malloc(sizeof(data_frame_t));
  train_df->p = p;
  train_df->n = n;
  train_df->var_labels = var_labels;
  train_df->var_types = var_types;
  train_df->data = data;
  
  rf_model_t *model = create_empty_model();
  model->p = p;
  model->var_types = (char*)malloc((p+1)*sizeof(char));
  memcpy(model->var_types, var_types, (p+1)*sizeof(char));
  
  model->var_labels = (char**)malloc((p+1)*sizeof(char*));
  for(int j = 0; j <= p; j++){
    model->var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
    strncpy(model->var_labels[j], var_labels[j], MAX_VAR_NAME_LEN-1);
  }
  
  int n_numeric_cuts = par["n_numeric_cuts"];
  int n_integer_cuts = par["n_integer_cuts"];
  int max_integer_classes = par["max_integer_classes"];
  
  
  make_cuts(train_df, &model, n_numeric_cuts, n_integer_cuts); 
  bx_info_t *bx_train = make_bx(train_df, &model, nthreads);
  ycode_t *yc_train = make_yc(train_df, &model, max_integer_classes, nthreads);
  
  free(train_df);
  
  if(ps == 0) ps = (int)(round(sqrt(model->p)));
  build_forest(bx_train, yc_train, &model, ps, max_depth, min_node_size, ntrees, nthreads, bagging_method, bagging_proportion, split_search, search_radius);
  delete_bx(bx_train, model);
  delete_yc(yc_train);

  if(verbose){
    Rprintf("Tree 0 printout:\n");
    printTree(model->trees[0], 0, model->yc->nlevels);

    Rprintf("Target coding:\n");
    if(model->yc->ycuts_int != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %d\n", model->yc->level_names[c], model->yc->ycuts_int[c]);
        } else {
          Rprintf("%d\n", model->yc->ycuts_int[c]);
        }
      }
    } else if(model->yc->ycuts_num != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %0.4f\n", model->yc->level_names[c], model->yc->ycuts_num[c]);
        } else {
          Rprintf("%0.4f\n", model->yc->ycuts_num[c]);
        }
      }
    } else if(model->yc->yvalues_int != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %d\n", model->yc->level_names[c], model->yc->yvalues_int[c]);
        } else {
          Rprintf("%d\n", model->yc->yvalues_int[c]);
        }
      }
    } else if(model->yc->yvalues_num != NULL){
      for(int c = 0; c < model->yc->nlevels; c++){
        if(model->yc->level_names != NULL){
          Rprintf("%s %0.4f\n", model->yc->level_names[c], model->yc->yvalues_num[c]);
        } else {
          Rprintf("%0.4f\n", model->yc->yvalues_num[c]);
        }
      }
    }

    Rprintf("Split points:\n");
    int n_num_vars = 0;
    int n_int_vars = 0;
    int n_fac_vars = 0;
    for(int j = 1; j <= p; j++){
      Rprintf("%d | %s | %c | %d split points: \n", j, model->var_labels[j], model->var_types[j], model->n_bcols[j]);
      if(model->var_types[j] == 'n'){
        for(int c = 0; c < model->n_bcols[j]; c++){
          Rprintf("%0.4f ", model->numeric_cuts[n_num_vars][c]);
        }
        Rprintf("\n");
        n_num_vars++;
      } else if(model->var_types[j] == 'i'){
        for(int c = 0; c < model->n_bcols[j]; c++){
          Rprintf("%d ", model->integer_cuts[n_int_vars][c]);
        }
        Rprintf("\n");
        n_int_vars++;        
      } else if(model->var_types[j] == 'f'){
        Rprintf("%d levels.\n", model->factor_cuts[n_fac_vars]->nlevels);
        n_fac_vars++;          
      }
    }
  }

  flatten_model(&model, nthreads);

  if(verbose >= 2){
    Rprintf("Tree 0 rules:\n");
    printRules(model, 0);
  }
  
  for(int j = 0; j <= p; j++){
    if(var_types[j] == 'f'){
      // manually delete 
      delete_tree(((factor_t *)data[j])->levels);
      free(data[j]);
    }
  }
  free(data);


  // Make predictions
  
  // construct test data frame
  n = rdf_new.nrow();
  data = (void**)malloc((p+1)*sizeof(void*));
  for(int j = 0; j <= p; j++){
    data[j] = NULL;  // initialize
  }
  
  for(int j = 1; j <= p; j++){
    if(var_types[j] == 'f'){
      IntegerVector fac_vec = rdf_new[col_index[j]];
      CharacterVector fac_levels = fac_vec.attr("levels");
      int which_factor = model->index_in_group[j];
      //printf("which_factor = %d\n", which_factor);
      data[j] = copy_factor(0, model->factor_cuts[which_factor]);
      if(data[j] != NULL){
        ((factor_t*) data[j])->n = n;
        ((factor_t*) data[j])->index = (integer_t *) fac_vec.begin();
      }
    } else if(var_types[j] == 'n'){
      NumericVector num_vec = rdf_new[col_index[j]];
      data[j] = (numeric_t *) num_vec.begin();
    } else if(var_types[j] == 'i'){
      IntegerVector int_vec = rdf_new[col_index[j]];
      data[j] = (integer_t *) int_vec.begin();
    }
  }
  
  data_frame_t * test_df = (data_frame_t*) malloc(sizeof(data_frame_t));
  test_df->p = p;
  test_df->n = n;
  test_df->var_labels = var_labels;
  test_df->var_types = var_types;
  test_df->data = data;
  
  // prepare score matrix
  double **score = (double**)malloc(model->yc->nlevels*sizeof(double*));
  if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num == NULL){
    for(int k = 0; k < model->yc->nlevels; k++){
      NumericVector this_k(n);
      char var_name[MAX_VAR_NAME_LEN];
      if(model->yc->yvalues_int != NULL){
        if(model->var_types[0] == 'f'){  // if the target variable is a factor
          int this_level_index = model->yc->yvalues_int[k] - model->yc->start_index;
          snprintf(var_name,MAX_VAR_NAME_LEN, "%s", model->yc->level_names[this_level_index]);
        } else {
          snprintf(var_name,MAX_VAR_NAME_LEN, "%d", model->yc->yvalues_int[k]);
        }
      } else if(model->yc->yvalues_num != NULL){
        //snprintf(var_name,MAX_VAR_NAME_LEN, "%f", model->yc->yvalues_num[k]);
      }
      score[k] = (double *)this_k.begin();
      result.push_back(this_k, var_name);
    }
  } else {
    for(int k = 0; k < model->yc->nlevels; k++){
      NumericVector this_k(n);
      score[k] = (double *)this_k.begin();
    }    
  }
  
  bx_info_t *bx_test = make_bx(test_df, &model, nthreads);
  predict(model, bx_test, score, vote_method, nthreads);
  delete_bx(bx_test, model);

  if(model->yc->type == REGRESSION){
    NumericVector pred(n);
    for(int i = 0; i < n; i++){
      pred[i] = 0;
      //double normalizing_factor = 0;
      for(int k = 0; k < model->yc->nlevels; k++){
        pred[i] += model->yc->yavg[k]*score[k][i];
        //pred[i] += model->yc->yavg[k]*(score[k][i])*(score[k][i]);
        //normalizing_factor += (score[k][i])*(score[k][i]);
      }
      //pred[i] = pred[i] / normalizing_factor;
    }
    result.push_back(pred,"pred");
  } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num != NULL){
    NumericVector pred(n);
    for(int i = 0; i < n; i++){
      pred[i] = 0;
      //double normalizing_factor = 0;
      for(int k = 0; k < model->yc->nlevels; k++){
        pred[i] += model->yc->yvalues_num[k]*score[k][i];
        //pred[i] += model->yc->yvalues_num[k]*(score[k][i])*(score[k][i]);
        //normalizing_factor += (score[k][i])*(score[k][i]);
      }
      //pred[i] = pred[i] / normalizing_factor;
    }
    result.push_back(pred,"pred");    
  }
  
  free(score);
  
  for(int j = 0; j <= p; j++){
    if(model->var_types[j] == 'f'){
      //delete_factor((factor_t *)data[j]);
      if(data[j] != NULL){
        delete_tree(((factor_t *)data[j])->levels);
        free(data[j]);
      }
    }
  }
  free(data);
  free(test_df);

  delete_model(model);
  free(var_types);
  free(var_labels);
  
  return(result);
}

