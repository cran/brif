#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif


#include <R.h>  
#ifdef ENABLE_NLS
#include <libintl.h>
#define _(String) dgettext ("brif", String)
#else
#define _(String) (String)
#endif

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MAX_LEVEL_NAME_LEN 30
#define MAX_VAR_NAME_LEN 50
#define MAX_FILE_NAME_LEN 200
#define REGRESSION 0
#define CLASSIFICATION 1
#define LEFT 0
#define RIGHT 1
#define MAXDEPTH 40
#define MAXNODES 1000
#define MAXBITBLOCK_VALUE UINT_MAX
#define INTEGER_T_MIN INT_MIN
#define INTEGER_T_MAX INT_MAX
#define NUMERIC_TYPE 0
#define INTEGER_TYPE 1
#define FACTOR_TYPE 2

    
// Declarations of this file

typedef double numeric_t;
typedef int integer_t;
typedef unsigned int bitblock_t;

typedef struct ycode {
    bitblock_t ** ymat;
    integer_t *yvalues_int;  // representative value of the bucket's truth value
    integer_t *ycuts_int;  // cut points low to high
    numeric_t *yvalues_num;  // representative value of the bucket's truth value
    numeric_t *ycuts_num;  // cut points low to high
    numeric_t *yavg;  // average y value in the interval
    int start_index;  // start index of factors
    char** level_names;  // if y is factor, this points to level names in order
    int nlevels;
    int type;  // 0 regression; 1 classification
} ycode_t;

typedef struct integer_linked_list {
    integer_t val;
    struct integer_linked_list * next;
} integer_linked_list_t;

typedef struct numeric_linked_list {
    numeric_t val;
    struct numeric_linked_list * next;
} numeric_linked_list_t;

typedef struct param {
    int n;
    int p;
    int n_blocks;
    int n_discard_bits;
    int J;  // number of response levels (0...(J-1)), J=0 means regression problem
    int min_node_size; 
    int max_depth;
    int n_numeric_cuts;
    int n_integer_cuts;
    int max_integer_classes;
    int ntrees;
    int nthreads;
    int ps; // number of features to sample each time for a split
} param_t;

typedef struct fnode {
    char name[MAX_LEVEL_NAME_LEN];
    int val;  
    int count;  
    struct fnode * left;
    struct fnode * right;
} fnode_t;

typedef struct {
    int n;
    int start_index; // 0 for C, 1 for R
    int nlevels;
    fnode_t *levels;  // name tree
    int * index;  // index array of size n
} factor_t;

typedef struct {
    bitblock_t ***bx;  
    int n;
    int n_blocks;
    int n_discard_bits;
} bx_info_t;

typedef struct {
    char *var_types;
    integer_t **integer_cuts;
    numeric_t **numeric_cuts;
    int *n_bcols;
} x_info_t;

typedef struct node {
    //int J; // number of classes
    int *count;  // number of cases in different classes indexed by class
    int rulepath_var[MAXDEPTH];  // how to get here, positive left, negative right
    int rulepath_bx[MAXDEPTH];
    int depth;  // 0 means the root node
    int split_var;  // split variable index
    int split_bx;  // the index of the binary feature created from the split_var
    struct node* left;
    struct node* right;
} dt_node_t;

typedef struct leave {
    //int J;
    int *count;
    int rulepath_var[MAXDEPTH];
    int rulepath_bx[MAXDEPTH];
    int depth;
    struct leave * next;
} dt_leaf_t;

typedef struct rf_model {
    int p;  // number of predictors
    char *var_types;  // type designation character (p+1)
    char **var_labels;  // variable names (p+1)
    int *n_bcols;  // number of binary features of each variable (p+1)
    int ntrees;  // number of trees
    int *index_in_group;  // index of each variable in its type group (p+1)
    numeric_t **numeric_cuts;
    integer_t **integer_cuts;
    factor_t **factor_cuts; 
    int n_num_vars;
    int n_int_vars;
    int n_fac_vars;
    dt_node_t **trees;
    dt_leaf_t **tree_leaves;
    ycode_t *yc;  // y code
} rf_model_t;

typedef struct {
    int n;
    int p;
    char *var_types;  // type designation character (p+1)
    char **var_labels;  // variable names (p+1)
    void **data;
} data_frame_t;

void fillSetBitTable(unsigned char table[], int n);
dt_node_t* newNode(dt_node_t* parent, int J, unsigned branch);
void deleteTree(dt_node_t* root);
void deleteLeaves(dt_leaf_t* root);
void set_default_params(param_t *param);
integer_linked_list_t * create_int_list(integer_t val);
void add_int_next(integer_linked_list_t **list, integer_t val);
void delete_int_list(integer_linked_list_t **list);
numeric_linked_list_t * create_num_list(numeric_t val);
void add_num_next(numeric_linked_list_t **list, numeric_t val);
void delete_num_list(numeric_linked_list_t **list);
int insert_node(fnode_t **tree, char *name, int n);
int check_value(fnode_t *tree, char *name, int val);
int find_value(fnode_t *tree, char *name);
void fill_name_array(fnode_t *tree, char **name, int start_index);
factor_t * create_factor(int n);
void copy_tree(fnode_t **newtree, fnode_t *tree);
factor_t * copy_factor(int n, factor_t * fm);
void add_element(factor_t *f, int index, char *name);
void find_add_element(factor_t *f, int index, char *name);
void delete_tree(fnode_t *tree);
void delete_factor(factor_t * f);

int cmpfunc_ptr_integer_t(const void *v1, const void *v2);
int cmpfunc_ptr_numeric_t(const void *v1, const void *v2);
numeric_t * numeric_cutpoints(numeric_t *x, int n, int *n_cuts);
integer_t * integer_cutpoints(integer_t *x, int n, int *n_cuts);
numeric_t * numeric_cutpoints_2(numeric_t *x, int n, int *n_cuts, int *yindex, int J, int start_index);
integer_t * integer_cutpoints_2(integer_t *x, int n, int *n_cuts, int *yindex, int J, int start_index);

// set the k-th bit from left of x 
void set_bit(bitblock_t *x, int k);

bitblock_t ** binarize_numeric(numeric_t *x, numeric_t *cuts, int n, int n_blocks, int n_cuts, int nthreads);
bitblock_t ** binarize_factor_index(int *index, int n, int n_blocks, int n_levels, int start_index, int nthreads);
factor_t * factor_cutpoints(factor_t *f, int n, int *n_cuts);
bitblock_t ** binarize_factor(factor_t *f, int n, int n_blocks, int *n_cuts, int nthreads);
bitblock_t ** binarize_integer(integer_t *x, integer_t *cuts, int n, int n_blocks, int n_cuts, int nthreads);

void delete_bmat(bitblock_t **bmat, int ncol);

ycode_t * codify_integer_target(integer_t *y, int n, int n_blocks, int max_integer_classes, int nthreads);
ycode_t * codify_factor_target(factor_t *y, int n, int n_blocks, int max_integer_classes, int nthreads);
ycode_t * codify_numeric_target(numeric_t *y, int n, int n_blocks, int max_integer_classes, int nthreads);

// copy yc ignoring ymat
ycode_t * copy_ycode(ycode_t * yc);

int countSetBits(bitblock_t n);
int count1s(bitblock_t *x, int n_blocks);
void shuffle_array_first_ps(int *arr, int n, int ps);
void find_best_split(bx_info_t *bxall, ycode_t *yc, rf_model_t *model, int min_node_size, int split_search, dt_node_t * cur_node, bitblock_t *useful_cur, int *uindex, int n_useful_blocks, int *var_index, int actual_ps, bitblock_t *z3, bitblock_t *z4, int *count, int *child_count, int *train_count, int *valid_count, int search_radius, int *candidate_index, int *split_var, int* split_bx);
void bootstrap_index_array(int n, int *array);
dt_node_t* build_tree(bx_info_t *bxall, ycode_t *yc, rf_model_t *model, int ps, int max_depth, int min_node_size, int bagging_method, double bagging_proportion, int split_search, int search_radius);
void predict_tree(dt_node_t *tree, bitblock_t ***bx, int **pred_tree, int J, int n_blocks);
void predict_leaves(dt_leaf_t *leaves, bitblock_t ***bx, int **pred_tree, int J, int n_blocks);
void predict(rf_model_t *model, bx_info_t * bx_new, double **pred, int vote_method, int nthreads);
rf_model_t *create_empty_model(void);
void get_numeric_summary(numeric_t *vector, int n, numeric_t *min_val, numeric_t *max_val, numeric_t *avg_val);
void get_integer_summary(integer_t *vector, int n, integer_t *min_val, integer_t *max_val, numeric_t *avg_val);
void delete_data(data_frame_t *df);
void delete_yc(ycode_t * yc);
void delete_model(rf_model_t *model);
void make_cuts(data_frame_t *train, rf_model_t **model, int n_numeric_cuts, int n_integer_cuts);
bx_info_t * make_bx(data_frame_t * train, rf_model_t ** model, int nthreads);
void delete_bx(bx_info_t *bxall, rf_model_t *model);
ycode_t * make_yc(data_frame_t *train, rf_model_t **model, int max_integer_classes, int nthreads);
void build_forest(bx_info_t *bxall, ycode_t *yc, rf_model_t **model, int ps, int max_depth, int min_node_size, int ntrees, int nthreads, int bagging_method, double bagging_proportion, int split_search, int search_radius);
void flatten_tree(dt_node_t *tree, dt_leaf_t **leaves, int J);
void flatten_model(rf_model_t **model, int nthreads);
void printTree(dt_node_t * tree, unsigned indent, int J);
void fill_name_addr_array(fnode_t *tree, char **name, int start_index);
void printRules(rf_model_t *model, int which_tree);
