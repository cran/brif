#include "brif.h"

// branch means which branch this node is of its parent 
dt_node_t* newNode(dt_node_t* parent, int J, unsigned branch){
    dt_node_t* node = (dt_node_t*)malloc(sizeof(dt_node_t));
    node->count = (int*)malloc(J*sizeof(int));
    if(parent == NULL){
        node->depth = 0;
    } else {
        node->depth = parent->depth + 1;
        memcpy(node->rulepath_var, parent->rulepath_var, (parent->depth)*sizeof(int));
        memcpy(node->rulepath_bx, parent->rulepath_bx, (parent->depth)*sizeof(int));
        node->rulepath_var[parent->depth] = (branch == LEFT ? parent->split_var : -(parent->split_var));
        node->rulepath_bx[parent->depth] = parent->split_bx;
    }
    node->split_var = 0;  // initialize to 0
    node->split_bx = 0; // initialize to 0
    node->left = NULL;
    node->right = NULL;
    return (node);
}

void deleteTree(dt_node_t* root){
    if(root != NULL){
        if(root->left != NULL){
            deleteTree(root->left);
        }
        if(root->right != NULL){
            deleteTree(root->right);
        }
        free(root->count);
        free(root);
    }
}

void deleteLeaves(dt_leaf_t* root){
    if(root != NULL){
        dt_leaf_t * this_leaf = root;
        root = root->next;
        free(this_leaf->count);
        free(this_leaf);
        deleteLeaves(root);        
    }
}

void set_default_params(param_t *param){
    param->min_node_size = 5;
    param->max_depth = 5;
    param->n_numeric_cuts = 15;
    param->n_integer_cuts = 15;
    param->max_integer_classes = 20;
    param->ntrees = 1;
    param->nthreads = 1;
}

integer_linked_list_t * create_int_list(integer_t val){
    integer_linked_list_t * list = (integer_linked_list_t *)malloc(sizeof(integer_linked_list_t));
    list->val = val;
    list->next = NULL;
    return(list);
}

void add_int_next(integer_linked_list_t **list, integer_t val){
    integer_linked_list_t *a = (integer_linked_list_t *)malloc(sizeof(integer_linked_list_t));
    a->val = val;
    a->next = *list;
    *list = a;
}

void delete_int_list(integer_linked_list_t **list){
    if(*list == NULL) return;
    integer_linked_list_t *a = *list;
    *list = (*list)->next;
    free(a);
    delete_int_list(list);
}

numeric_linked_list_t * create_num_list(numeric_t val){
    numeric_linked_list_t * list = (numeric_linked_list_t *)malloc(sizeof(numeric_linked_list_t));
    list->val = val;
    list->next = NULL;
    return(list);
}

void add_num_next(numeric_linked_list_t **list, numeric_t val){
    numeric_linked_list_t *a = (numeric_linked_list_t *)malloc(sizeof(numeric_linked_list_t));
    a->val = val;
    a->next = *list;
    *list = a;
}

void delete_num_list(numeric_linked_list_t **list){
    if(*list == NULL) return;
    numeric_linked_list_t *a = *list;
    *list = (*list)->next;
    free(a);
    delete_num_list(list);
}

int insert_node(fnode_t **tree, char *name, int n){
    if(*tree !=NULL){
        int where = strcmp(name, (*tree)->name);
        if(where < 0){
            return insert_node(&((*tree)->left),name,n);
        } else if(where > 0){
            return insert_node(&((*tree)->right), name,n);
        } else {
            (*tree)->count += 1;
            return (*tree)->val;
        }
    } else {
        *tree = (fnode_t*)malloc(sizeof(fnode_t));
        memcpy((*tree)->name, name, MIN(MAX_LEVEL_NAME_LEN, strlen(name)+1));
        (*tree)->val = n;
        (*tree)->count = 1;
        (*tree)->left = NULL;
        (*tree)->right = NULL;
        return (*tree)->val;
    }
}

// given a name and an index value, check if they match in the binary search tree; return 1 yes, 0 no
int check_value(fnode_t *tree, char *name, int val){
    if(tree != NULL){
        int where = strcmp(name, tree->name);
        if(where < 0){
            return check_value(tree->left, name, val);
        } else if(where > 0){
            return check_value(tree->right, name, val);
        } else {
            return tree->val == val ? 1 : 0;
        }
    } else {
        return 0;
    }
}

// given a name, return its value in the binary search tree; if not found, return -1
int find_value(fnode_t *tree, char *name){
    if(tree != NULL){
        int where = strcmp(name, tree->name);
        if(where < 0){
            return find_value(tree->left, name);
        } else if(where > 0){
            return find_value(tree->right, name);
        } else {
            return tree->val;
        }
    } else {
        return -1;
    }
}

// traverse the tree to copy names in the name array in order of value
// this function is to be used by Rcpp
void fill_name_array(fnode_t *tree, char **name, int start_index){
    if(tree != NULL){
        memcpy(name[tree->val - start_index], tree->name, MAX_LEVEL_NAME_LEN);
        fill_name_array(tree->left, name, start_index);
        fill_name_array(tree->right, name, start_index);
    }
}

factor_t * create_factor(int n){
    factor_t *f = (factor_t*) malloc(sizeof(factor_t));
    f->n = n;
    f->start_index = 1;  // default is 1, unless changed manually
    f->nlevels = 0;
    f->levels = NULL;
    if(n > 0){
        f->index = (int*)malloc(n*sizeof(int));
    } else {
        f->index = NULL;
    }
    return f;
}

// make a copy of the tree
void copy_tree(fnode_t **newtree, fnode_t *tree){
    if(tree != NULL){
        *newtree = (fnode_t*)malloc(sizeof(fnode_t));
        memcpy((*newtree)->name, tree->name, MAX_LEVEL_NAME_LEN);
        (*newtree)->val = tree->val;
        (*newtree)->count = 0;
        (*newtree)->left = NULL;
        (*newtree)->right = NULL;
        copy_tree(&((*newtree)->left), tree->left);
        copy_tree(&((*newtree)->right), tree->right);
    }
}

factor_t * copy_factor(int n, factor_t * fm){
    if(fm == NULL) return NULL;
    factor_t *f = (factor_t*)malloc(sizeof(factor_t));
    f->n = n;
    f->start_index = fm->start_index;
    f->nlevels = fm->nlevels;
    f->levels = NULL;
    copy_tree(&(f->levels),fm->levels);
    if(n > 0){
        f->index = (int*)malloc(n*sizeof(int));
    } else {
        f->index = NULL;
    }
    return f;
}

void add_element(factor_t *f, int index, char *name){
    (f->index)[index] = insert_node(&(f->levels), name, f->start_index + f->nlevels);
    if((f->index)[index] == f->start_index + f->nlevels){
        f->nlevels += 1;
    }
}

void find_add_element(factor_t *f, int index, char *name){
    (f->index)[index] = find_value(f->levels, name);
}

void delete_tree(fnode_t *tree){
    if(tree != NULL){
        delete_tree(tree->left);
        delete_tree(tree->right);
        free(tree);
    }
}

void delete_factor(factor_t * f){
    if(f == NULL) return;
    delete_tree(f->levels);
    if(f->index != NULL) free(f->index);
    free(f);
}


int cmpfunc_ptr_integer_t(const void *v1, const void *v2){
    const integer_t i1 = **(const integer_t **)v1;
    const integer_t i2 = **(const integer_t **)v2;
    return i1<i2?-1:(i1>i2);
}

int cmpfunc_ptr_numeric_t(const void *v1, const void *v2){
    const numeric_t i1 = **(const numeric_t **)v1;
    const numeric_t i2 = **(const numeric_t **)v2;
    return i1<i2?-1:(i1>i2);
}

numeric_t * numeric_cutpoints(numeric_t *x, int n, int *n_cuts){
    numeric_t **x_ptr = (numeric_t **)malloc(n*sizeof(numeric_t*));
    for(int i = 0; i < n; i++){
        x_ptr[i] = &x[i];
    }
    qsort(x_ptr, n, sizeof(numeric_t*), cmpfunc_ptr_numeric_t);
    numeric_linked_list_t *unique_values = create_num_list(x[x_ptr[n-1] - x]);
    int n_unique = 1;

    for(int ii = n-2; ii >= 0; ii--){
        int this_one = x_ptr[ii] - x;
        int prev = x_ptr[ii+1] - x;
        if(x[this_one] != x[prev]){
            n_unique += 1;
            if(n_unique <= *n_cuts){  // do not need to finish constructing the linked list if unique values are too many
                add_num_next(&unique_values, x[this_one]);
            }
        }
    }

    if(n_unique == 1){
        *n_cuts = 0;
        free(x_ptr);      
        delete_num_list(&unique_values);
        return NULL;        
    }

    numeric_t *cuts = NULL;
    if(n_unique <= *n_cuts){
        numeric_linked_list_t *iter = unique_values;
        *n_cuts = n_unique;
        //printf("%d unique value cuts:\n", *n_cuts);
        cuts = (numeric_t*)malloc((*n_cuts)*sizeof(numeric_t));
        for(int i = 0; i < *n_cuts; i++){
            cuts[i] = iter->val;
            iter = iter->next;
            //printf("%0.3f ", cuts[i]);
        }
        //printf("\n");
    } else {
        cuts = (numeric_t*)malloc((*n_cuts)*sizeof(numeric_t));
        int actual_n_cuts = 0;
        for(int c = 0; c < *n_cuts; c++){
            int qdex = n*(c+1)/(*n_cuts+1);
            int i = x_ptr[qdex] - x;
            if(c > 0){
                if(x[i] == cuts[actual_n_cuts-1]){
                    int found = 0;
                    // search up until meeting the first different value
                    for(int ii = 1; ii < n / (*n_cuts + 1); ii++){
                        if(x[x_ptr[qdex+ii] - x] != cuts[actual_n_cuts-1]){
                            found = 1;
                            i = x_ptr[qdex+ii] - x;
                            break;
                        }
                    }
                    if(found == 0){
                        continue;
                    }
                }
            }
            cuts[actual_n_cuts++] = x[i];
        }  
        if(actual_n_cuts < *n_cuts){
            cuts = (numeric_t*)realloc(cuts, actual_n_cuts*sizeof(numeric_t));
            *n_cuts = actual_n_cuts;
        }
    }
    free(x_ptr);      
    delete_num_list(&unique_values);
    return(cuts);
}

// use factor y to locally refine cutpoint placement
numeric_t * numeric_cutpoints_2(numeric_t *x, int n, int *n_cuts, int *yindex, int J, int start_index){
    
    numeric_t **x_ptr = (numeric_t **)malloc(n*sizeof(numeric_t*));
    for(int i = 0; i < n; i++){
        x_ptr[i] = &x[i];
    }
    qsort(x_ptr, n, sizeof(numeric_t*), cmpfunc_ptr_numeric_t);
    numeric_linked_list_t *unique_values = create_num_list(x[x_ptr[n-1] - x]);
    int n_unique = 1;

    for(int ii = n-2; ii >= 0; ii--){
        int this_one = x_ptr[ii] - x;
        int prev = x_ptr[ii+1] - x;
        if(x[this_one] != x[prev]){
            n_unique += 1;
            if(n_unique <= *n_cuts){  // do not need to finish constructing the linked list if unique values are too many
                add_num_next(&unique_values, x[this_one]);
            }
        }
    }

    if(n_unique == 1){
        *n_cuts = 0;
        free(x_ptr);      
        delete_num_list(&unique_values);
        return NULL;        
    }

    numeric_t *cuts = NULL;
    if(n_unique <= *n_cuts){
        numeric_linked_list_t *iter = unique_values;
        *n_cuts = n_unique;
        //printf("%d unique value cuts:\n", *n_cuts);
        cuts = (numeric_t*)malloc((*n_cuts)*sizeof(numeric_t));
        for(int i = 0; i < *n_cuts; i++){
            cuts[i] = iter->val;
            iter = iter->next;
            //printf("%0.3f ", cuts[i]);
        }
        //printf("\n");
    } else {
        int* count = (int*)malloc(J*sizeof(int));
        int last_qdex = -1;
        cuts = (numeric_t*)malloc((*n_cuts)*sizeof(numeric_t));
        int actual_n_cuts = 0;
        int c = 0;
        while(c < *n_cuts){
            int qdex = n*(c+1)/(*n_cuts+1);
            int shift = 0;
            // get y count by class in the interval
            memset(count, 0, J*sizeof(int));
            for(int q = last_qdex+1; q <= qdex; q++){
                count[yindex[x_ptr[q] - x] - start_index] += 1;
            }
            int majority_class = -1; 
            int minority_class = -1;
            int max_count = 0;
            int min_count = n / (*n_cuts + 1);
            for(int k = 0; k < J; k++){
                if(count[k] > max_count){
                    max_count = count[k];
                    majority_class = k;
                }
                if(count[k] > 0 && count[k] < min_count){
                    min_count = count[k];
                    minority_class = k;
                }
            }
            //Rprintf("majority class = %d, minority class = %d\n", majority_class, minority_class);
            
            if(majority_class != minority_class){
                if(majority_class != -1){
                    while(qdex + shift < n && yindex[x_ptr[qdex+shift] - x] - start_index == majority_class){
                        shift++;
                    }
                } 
                if(shift == 0 && minority_class != -1){
                    while(qdex+shift > last_qdex && yindex[x_ptr[qdex+shift] - x] - start_index == minority_class){
                        shift--;
                    }
                }
            }
            if(qdex + shift == n){
                break;  // done adding cuts
            }
            
            int i = x_ptr[qdex+shift] - x;
            last_qdex = qdex + shift;
            int ii = 0;
            if(actual_n_cuts > 0){
                if(x[i] == cuts[actual_n_cuts-1]){
                    // search up until meeting the first different value
                    ii = 1;
                    while(ii + qdex + shift < n && x[x_ptr[qdex+shift+ii] - x] == cuts[actual_n_cuts-1]){
                        ii++;
                    }
                }
            }
            //Rprintf("shift = %d, ii = %d\n", shift, ii);

            i = x_ptr[qdex+shift+ii] - x;
            last_qdex = qdex+shift+ii;    
            if(last_qdex < n-1){  // if it happens to be the last data point, no need to place a cutpoint
                cuts[actual_n_cuts++] = x[i];
            }
            // set c accordingly for the next iteration
            c += 1 + (shift+ii) / ((n / (*n_cuts + 1)));
        }  
        if(actual_n_cuts < *n_cuts){
            cuts = (numeric_t*)realloc(cuts, actual_n_cuts*sizeof(numeric_t));
            *n_cuts = actual_n_cuts;
        }

        free(count);
    }
    free(x_ptr);      
    delete_num_list(&unique_values);
    return(cuts);
}

integer_t * integer_cutpoints(integer_t *x, int n, int *n_cuts){
    integer_t **x_ptr = (integer_t**)malloc(n*sizeof(integer_t*));
    for(int i = 0; i < n; i++){
        x_ptr[i] = &x[i];
    }
    qsort(x_ptr, n, sizeof(integer_t*), cmpfunc_ptr_integer_t);
    integer_linked_list_t *unique_values = create_int_list(x[x_ptr[n-1] - x]);
    int n_unique = 1;
    
    for(int ii = n-2; ii >= 0; ii--){
        int this_one = x_ptr[ii] - x;
        int prev = x_ptr[ii+1] - x;
        if(x[this_one] != x[prev]){
            n_unique += 1;
            if(n_unique <= *n_cuts){  // do not need to finish constructing the linked list if unique values are too many
                add_int_next(&unique_values, x[this_one]);
            }
        }
    }

    if(n_unique == 1){
        *n_cuts = 0;
        free(x_ptr);      
        delete_int_list(&unique_values);
        return NULL;        
    }
    
    //print_int_list(unique_values);
    integer_t *cuts = NULL;
    if(n_unique <= *n_cuts){
        integer_linked_list_t *iter = unique_values;
        *n_cuts = n_unique;
        cuts = (integer_t*)malloc((*n_cuts)*sizeof(integer_t));
        for(int i = 0; i < *n_cuts; i++){
            cuts[i] = iter->val;
            iter = iter->next;
        }
        // the cuts are already sorted low to high coming out of the linked list
        //qsort(cuts, *n_cuts, sizeof(integer_t), cmpfunc_element_integer_t);
        /*
        printf("Sorted Integer cuts:");
        for(int i = 0; i < *n_cuts; i++){
            printf("%d ", cuts[i]);
        }
        printf("\n");
        */
    } else {
        cuts = (integer_t*)malloc((*n_cuts)*sizeof(integer_t));
        int actual_n_cuts = 0;
        for(int c = 0; c < *n_cuts; c++){
            int qdex = n*(c+1)/(*n_cuts+1);
            int i = x_ptr[qdex] - x;
            if(c > 0){
                if(x[i] == cuts[actual_n_cuts-1]){
                    int found = 0;
                    // search up until meeting the first different value
                    for(int ii = 1; ii < n / (*n_cuts + 1); ii++){
                        if(x[x_ptr[qdex+ii] - x] != cuts[actual_n_cuts-1]){
                            found = 1;
                            i = x_ptr[qdex+ii] - x;
                            break;
                        }
                    }
                    if(found == 0){
                        continue;
                    }
                }
            }
            cuts[actual_n_cuts++] = x[i];
        }  
        if(actual_n_cuts < *n_cuts){
            cuts = (integer_t*)realloc(cuts, actual_n_cuts*sizeof(integer_t));
            *n_cuts = actual_n_cuts;
        }
    }
    free(x_ptr);      
    delete_int_list(&unique_values);
    return(cuts);
}

integer_t * integer_cutpoints_2(integer_t *x, int n, int *n_cuts, int *yindex, int J, int start_index){
    integer_t **x_ptr = (integer_t**)malloc(n*sizeof(integer_t*));
    for(int i = 0; i < n; i++){
        x_ptr[i] = &x[i];
    }
    qsort(x_ptr, n, sizeof(integer_t*), cmpfunc_ptr_integer_t);
    integer_linked_list_t *unique_values = create_int_list(x[x_ptr[n-1] - x]);
    int n_unique = 1;
    
    for(int ii = n-2; ii >= 0; ii--){
        int this_one = x_ptr[ii] - x;
        int prev = x_ptr[ii+1] - x;
        if(x[this_one] != x[prev]){
            n_unique += 1;
            if(n_unique <= *n_cuts){  // do not need to finish constructing the linked list if unique values are too many
                add_int_next(&unique_values, x[this_one]);
            }
        }
    }

    if(n_unique == 1){
        *n_cuts = 0;
        free(x_ptr);      
        delete_int_list(&unique_values);
        return NULL;        
    }
    
    //print_int_list(unique_values);
    integer_t *cuts = NULL;
    if(n_unique <= *n_cuts){
        integer_linked_list_t *iter = unique_values;
        *n_cuts = n_unique;
        cuts = (integer_t*)malloc((*n_cuts)*sizeof(integer_t));
        for(int i = 0; i < *n_cuts; i++){
            cuts[i] = iter->val;
            iter = iter->next;
        }
    } else {
        int* count = (int*)malloc(J*sizeof(int));
        int last_qdex = -1;
        cuts = (integer_t*)malloc((*n_cuts)*sizeof(integer_t));
        int actual_n_cuts = 0;
        int c = 0;
        while(c < *n_cuts){
            int qdex = n*(c+1)/(*n_cuts+1);
            int shift = 0;
            // get y count by class in the interval
            memset(count, 0, J*sizeof(int));
            for(int q = last_qdex+1; q <= qdex; q++){
                count[yindex[x_ptr[q] - x] - start_index] += 1;
            }
            int majority_class = -1; 
            int minority_class = -1;
            int max_count = 0;
            int min_count = n / (*n_cuts + 1);
            for(int k = 0; k < J; k++){
                if(count[k] > max_count){
                    max_count = count[k];
                    majority_class = k;
                }
                if(count[k] > 0 && count[k] < min_count){
                    min_count = count[k];
                    minority_class = k;
                }
            }
            //Rprintf("majority class = %d, minority class = %d\n", majority_class, minority_class);
            
            if(majority_class != minority_class){
                if(majority_class != -1){
                    while(qdex + shift < n && yindex[x_ptr[qdex+shift] - x] - start_index == majority_class){
                        shift++;
                    }
                } 
                if(shift == 0 && minority_class != -1){
                    while(qdex+shift > last_qdex && yindex[x_ptr[qdex+shift] - x] - start_index == minority_class){
                        shift--;
                    }
                }
            }
            if(qdex + shift == n){
                break;  // done adding cuts
            }
            
            int i = x_ptr[qdex+shift] - x;
            last_qdex = qdex + shift;
            int ii = 0;
            if(actual_n_cuts > 0){
                if(x[i] == cuts[actual_n_cuts-1]){
                    // search up until meeting the first different value
                    ii = 1;
                    while(ii + qdex + shift < n && x[x_ptr[qdex+shift+ii] - x] == cuts[actual_n_cuts-1]){
                        ii++;
                    }
                }
            }
            //Rprintf("shift = %d, ii = %d\n", shift, ii);

            i = x_ptr[qdex+shift+ii] - x;
            last_qdex = qdex+shift+ii;    
            if(last_qdex < n-1){  // if it happens to be the last data point, no need to place a cutpoint
                cuts[actual_n_cuts++] = x[i];
            }
            // set c accordingly for the next iteration
            c += 1 + (shift+ii) / ((n / (*n_cuts + 1)));
        }  
        if(actual_n_cuts < *n_cuts){
            cuts = (integer_t*)realloc(cuts, actual_n_cuts*sizeof(integer_t));
            *n_cuts = actual_n_cuts;
        }
        free(count);
    }
    free(x_ptr);      
    delete_int_list(&unique_values);
    return(cuts);
}

// set the k-th bit from left of x 
void set_bit(bitblock_t *x, int k){
    *x |= (1 << (8*sizeof(bitblock_t) - 1 - k));
}

bitblock_t ** binarize_numeric(numeric_t *x, numeric_t *cuts, int n, int n_blocks, int n_cuts){
    if(n_cuts == 0) return NULL;
    bitblock_t ** bmat = (bitblock_t**)malloc(n_cuts*sizeof(bitblock_t*));
    for(int c = 0; c < n_cuts; c++){
        bmat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
        memset(bmat[c], 0, n_blocks*sizeof(bitblock_t));
    }
    int block_num = 0;
    int bit_num = 0;
    for(int i = 0; i < n; i++){
        block_num = i / (8*sizeof(bitblock_t));
        bit_num = i % (8*sizeof(bitblock_t));
        for(int c = 0; c < n_cuts; c++){
            if(x[i] <= cuts[c]){  // split rule is always in the form of value <= cutpoint
                for(int cc = c; cc < n_cuts; cc++){
                    set_bit(&bmat[cc][block_num], bit_num);
                }
                break;
            }
        }
    }
    return(bmat);
}

bitblock_t ** binarize_factor_index(int *index, int n, int n_blocks, int n_levels, int start_index){
    if(n_levels == 0) return NULL;
    bitblock_t ** bmat = (bitblock_t**)malloc(n_levels*sizeof(bitblock_t*));
    for(int c = 0; c < n_levels; c++){
        bmat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
        memset(bmat[c], 0, n_blocks*sizeof(bitblock_t));
    }
    int block_num = 0;
    int bit_num = 0;
    for(int i = 0; i < n; i++){
        block_num = i / (8*sizeof(bitblock_t));
        bit_num = i % (8*sizeof(bitblock_t));
        for(int c = 0; c < n_levels; c++){
            if(index[i] == c+start_index){  // split rule is always in the form of value <= cutpoint
                set_bit(&bmat[c][block_num], bit_num);
                break;
            }
        }
    }
    return(bmat);
}

factor_t * factor_cutpoints(factor_t *f, int n, int *n_cuts){
    if(f->nlevels <= 1){
        *n_cuts = 0;
        return NULL;
    } else {
        return copy_factor(0, f);
    }
}

bitblock_t ** binarize_factor(factor_t *f, int n, int n_blocks, int *n_cuts){    
    if(f->nlevels <= 1){
        *n_cuts = 0;
        return NULL;
    } else {
        return binarize_factor_index(f->index, n, n_blocks, f->nlevels, f->start_index);
    }
}

bitblock_t ** binarize_integer(integer_t *x, integer_t *cuts, int n, int n_blocks, int n_cuts){
    if(n_cuts == 0) return NULL;
    bitblock_t ** bmat = (bitblock_t**)malloc(n_cuts*sizeof(bitblock_t*));
    for(int c = 0; c < n_cuts; c++){
        bmat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
        memset(bmat[c], 0, n_blocks*sizeof(bitblock_t));
    }
    int block_num = 0;
    int bit_num = 0;
    for(int i = 0; i < n; i++){
        block_num = i / (8*sizeof(bitblock_t));
        bit_num = i % (8*sizeof(bitblock_t));
        for(int c = 0; c < n_cuts; c++){
            if(x[i] <= cuts[c]){  // split rule is always in the form of value <= cutpoint
                for(int cc = c; cc < n_cuts; cc++){
                    set_bit(&bmat[cc][block_num], bit_num);
                }
                break;
            }
        }
    }
    return(bmat);
}

void delete_bmat(bitblock_t **bmat, int ncol){
    for(int i = 0; i < ncol; i++){
        free(bmat[i]);
    }
    free(bmat);
}

ycode_t * codify_integer_target(integer_t *y, int n, int n_blocks, int max_integer_classes){
    ycode_t *yc = (ycode_t*)malloc(sizeof(ycode_t));
    yc->yvalues_num = NULL;
    yc->ycuts_num = NULL;
    yc->level_names = NULL;
    yc->start_index = 0; // placeholder, meaningless here
    int maxJ = max_integer_classes;
    integer_t **y_ptr = (integer_t**)malloc(n*sizeof(integer_t*));
    for(int i = 0; i < n; i++){
        y_ptr[i] = &y[i];
    }
    qsort(y_ptr, n, sizeof(integer_t*), cmpfunc_ptr_integer_t);
    integer_linked_list_t *unique_values = create_int_list(y[y_ptr[n-1] - y]);
    int n_unique = 1;
    for(int ii = n-2; ii >= 0; ii--){
        int this_one = y_ptr[ii] - y;
        int prev = y_ptr[ii+1] - y;
        if(y[this_one] != y[prev]){
            n_unique += 1;
            if(n_unique <= maxJ){  // do not need to finish constructing the linked list if unique values are too many
                add_int_next(&unique_values, y[this_one]);
            }
        }
    }
    //print_int_list(unique_values); 
    //assert(n_unique >= 2);
    if(n_unique <= maxJ){
        yc->nlevels = n_unique;
        yc->yvalues_int = (integer_t *)malloc(n_unique*sizeof(integer_t));
        yc->yavg = NULL;
        yc->ycuts_int = NULL;
        yc->type = CLASSIFICATION;
        integer_linked_list_t *iter = unique_values;
        //printf("yvalues: ");
        for(int i = 0; i < n_unique; i++){
            yc->yvalues_int[i] = (integer_t) iter->val;
            iter = iter->next;
            //printf("%d ", yc->yvalues_int[i]);
        }
        //printf("\n");
        yc->ymat = (bitblock_t **) malloc(n_unique*sizeof(bitblock_t *));
        for(int c = 0; c < n_unique; c++){
            yc->ymat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
            memset(yc->ymat[c], 0, n_blocks*sizeof(bitblock_t));
        }
        int block_num = 0;
        int bit_num = 0;
        for(int i = 0; i < n; i++){
            block_num = i / (8*sizeof(bitblock_t));
            bit_num = i % (8*sizeof(bitblock_t));
            for(int c = 0; c < n_unique; c++){
                if(y[i] == yc->yvalues_int[c]){  // split rule is always in the form of value <= cutpoint
                    set_bit(&(yc->ymat[c][block_num]), bit_num);
                    break;
                }
            }
        }        
    } else {
        yc->nlevels = maxJ;
        yc->yvalues_int = NULL;
        yc->yavg = (numeric_t *)malloc(maxJ*sizeof(numeric_t));  // yvalues represents the bucket mean
        yc->ycuts_int = (integer_t *)malloc(maxJ*sizeof(integer_t)); 
        yc->type = REGRESSION;
        yc->ymat = (bitblock_t **) malloc(maxJ*sizeof(bitblock_t *));
        for(int c = 0; c < maxJ; c++){
            int this_qdex = (n-1)*(c+1)/maxJ;
            int last_qdex = (n-1)*c/maxJ;
            //integer_t this_cut = y[y_ptr[this_qdex] - y];
            integer_t last_cut = y[y_ptr[last_qdex] - y];
            //printf("last_qdex = %d, this_qdex = %d ", last_qdex, this_qdex);
            //printf("last_cut = %d, this_cut = %d ", last_cut, this_cut);
            yc->ycuts_int[c] = last_cut;
            numeric_t avg_value = 0;
            for(int d = last_qdex; d < this_qdex; d++){
                avg_value += y[y_ptr[d] - y];
            }
            yc->yavg[c] = 1.0*avg_value / (this_qdex - last_qdex);
            //printf("yc->avg[%d] = %f \n", c, yc->yavg[c]);
            yc->ymat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
            memset(yc->ymat[c], 0, n_blocks*sizeof(bitblock_t));
        }
        
        int block_num = 0;
        int bit_num = 0;
        for(int i = 0; i < n; i++){
            block_num = i / (8*sizeof(bitblock_t));
            bit_num = i % (8*sizeof(bitblock_t));
            for(int c = 0; c < maxJ-1; c++){
                if(y[i] >= yc->ycuts_int[c] && y[i] < yc->ycuts_int[c+1]){  
                    set_bit(&(yc->ymat[c][block_num]), bit_num);
                    break;
                }
            }
            if(y[i] >= yc->ycuts_int[maxJ-1]){
                set_bit(&(yc->ymat[maxJ-1][block_num]), bit_num);
            }
        }           
    }
    
    free(y_ptr);
    delete_int_list(&unique_values);
    return(yc);
}

ycode_t * codify_factor_target(factor_t *y, int n, int n_blocks, int max_integer_classes){
    ycode_t *yc = (ycode_t*)malloc(sizeof(ycode_t)); 
    yc->start_index = y->start_index;
    yc->yvalues_num = NULL;
    yc->ycuts_num = NULL;
    yc->nlevels = y->nlevels;
    yc->yvalues_int = (integer_t *)malloc(yc->nlevels*sizeof(integer_t));
    yc->yavg = NULL;
    yc->ycuts_int = NULL;
    yc->type = CLASSIFICATION;
    yc->level_names = (char**)malloc(yc->nlevels*sizeof(char*));
    for(int c = 0; c < yc->nlevels; c++){
        yc->level_names[c] = (char*)malloc(MAX_LEVEL_NAME_LEN*sizeof(char));
    }
    fill_name_array(y->levels, yc->level_names, y->start_index);
    yc->ymat = (bitblock_t **)malloc(yc->nlevels*sizeof(bitblock_t*));  
    for(int c = 0; c < yc->nlevels; c++){
        yc->yvalues_int[c] = c + y->start_index;
        yc->ymat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
        memset(yc->ymat[c], 0, n_blocks*sizeof(bitblock_t));
    }
    int block_num = 0;
    int bit_num = 0;
    for(int i = 0; i < n; i++){
        block_num = i / (8*sizeof(bitblock_t));
        bit_num = i % (8*sizeof(bitblock_t));
        for(int c = 0; c < yc->nlevels; c++){
            if(y->index[i] == c+y->start_index){  // split rule is always in the form of value <= cutpoint
                set_bit(&yc->ymat[c][block_num], bit_num);
                break;
            }
        }
    }
    return(yc);
}

ycode_t * codify_numeric_target(numeric_t *y, int n, int n_blocks, int max_integer_classes){
    ycode_t *yc = (ycode_t*)malloc(sizeof(ycode_t));
    yc->yvalues_int = NULL;
    yc->ycuts_int = NULL;
    yc->level_names = NULL;
    yc->start_index = 0; // placeholder, meaningless here
    int maxJ = max_integer_classes;
    numeric_t **y_ptr = (numeric_t**)malloc(n*sizeof(numeric_t*));
    for(int i = 0; i < n; i++){
        y_ptr[i] = &y[i];
    }
    qsort(y_ptr, n, sizeof(numeric_t*), cmpfunc_ptr_numeric_t);
    numeric_linked_list_t *unique_values = create_num_list(y[y_ptr[n-1] - y]);
    int n_unique = 1;
    
    for(int ii = n-2; ii >= 0; ii--){
        int this_one = y_ptr[ii] - y;
        int prev = y_ptr[ii+1] - y;
        if(y[this_one] != y[prev]){
            n_unique += 1;
            if(n_unique <= maxJ){  // do not need to finish constructing the linked list if unique values are too many
                add_num_next(&unique_values, y[this_one]);
            }
        }
    }
    //print_num_list(unique_values); 
    //assert(n_unique >= 2);
    if(n_unique <= maxJ){
        yc->nlevels = n_unique;
        yc->yvalues_num = (numeric_t *)malloc(n_unique*sizeof(numeric_t));
        yc->yavg = NULL;
        yc->ycuts_num = NULL;
        yc->type = CLASSIFICATION;
        numeric_linked_list_t *iter = unique_values;
        //printf("yvalues: ");
        for(int i = 0; i < n_unique; i++){
            yc->yvalues_num[i] = (numeric_t) iter->val;
            iter = iter->next;
            //printf("%f ", yc->yvalues_num[i]);
        }
        //printf("\n");
        yc->ymat = (bitblock_t **) malloc(n_unique*sizeof(bitblock_t *));
        for(int c = 0; c < n_unique; c++){
            yc->ymat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
            memset(yc->ymat[c], 0, n_blocks*sizeof(bitblock_t));
        }
        int block_num = 0;
        int bit_num = 0;
        for(int i = 0; i < n; i++){
            block_num = i / (8*sizeof(bitblock_t));
            bit_num = i % (8*sizeof(bitblock_t));
            for(int c = 0; c < n_unique; c++){
                if(y[i] == yc->yvalues_num[c]){  // split rule is always in the form of value <= cutpoint
                    set_bit(&(yc->ymat[c][block_num]), bit_num);
                    break;
                }
            }
        }        
    } else {
        yc->nlevels = maxJ;
        yc->yvalues_num = NULL;
        yc->yavg = (numeric_t *)malloc(maxJ*sizeof(numeric_t));  // yvalues represents the bucket mean
        yc->ycuts_num = (numeric_t *)malloc(maxJ*sizeof(numeric_t)); 
        yc->type = REGRESSION;
        yc->ymat = (bitblock_t **) malloc(maxJ*sizeof(bitblock_t *));
        for(int c = 0; c < maxJ; c++){
            int this_qdex = (n-1)*(c+1)/maxJ;
            int last_qdex = (n-1)*c/maxJ;
            //numeric_t this_cut = y[y_ptr[this_qdex] - y];
            numeric_t last_cut = y[y_ptr[last_qdex] - y];
            //printf("last_qdex = %d, this_qdex = %d ", last_qdex, this_qdex);
            //printf("last_cut = %f, this_cut = %f ", last_cut, this_cut);
            yc->ycuts_num[c] = last_cut;
            numeric_t avg_value = 0;
            for(int d = last_qdex; d < this_qdex; d++){
                avg_value += y[y_ptr[d] - y];
            }
            yc->yavg[c] = 1.0*avg_value / (this_qdex - last_qdex);
            //printf("yc->avg[%d] = %f \n", c, yc->yavg[c]);
            yc->ymat[c] = (bitblock_t*)malloc(n_blocks*sizeof(bitblock_t));
            memset(yc->ymat[c], 0, n_blocks*sizeof(bitblock_t));
        }
        
        int block_num = 0;
        int bit_num = 0;
        for(int i = 0; i < n; i++){
            block_num = i / (8*sizeof(bitblock_t));
            bit_num = i % (8*sizeof(bitblock_t));
            for(int c = 0; c < maxJ-1; c++){
                if(y[i] >= yc->ycuts_num[c] && y[i] < yc->ycuts_num[c+1]){  
                    set_bit(&(yc->ymat[c][block_num]), bit_num);
                    break;
                }
            }
            if(y[i] >= yc->ycuts_num[maxJ-1]){
                set_bit(&(yc->ymat[maxJ-1][block_num]), bit_num);
            }
        }           
    }
    
    free(y_ptr);
    delete_num_list(&unique_values);
    return(yc);
}

// copy yc ignoring ymat
ycode_t * copy_ycode(ycode_t * yc){
    if(yc == NULL) return NULL;
    ycode_t * newyc = (ycode_t*)malloc(sizeof(ycode_t));
    newyc->nlevels = yc->nlevels;
    newyc->type = yc->type;
    newyc->start_index = yc->start_index;
    newyc->ymat = NULL;  // if forget, get a bug
    if(yc->yavg != NULL){
        newyc->yavg = (numeric_t*)malloc(yc->nlevels*sizeof(numeric_t));
        memcpy(newyc->yavg, yc->yavg, yc->nlevels*sizeof(numeric_t));
    } else {
        newyc->yavg = NULL;
    }

    if(yc->yvalues_int != NULL){
        newyc->yvalues_int = (integer_t*)malloc(yc->nlevels*sizeof(integer_t));
        memcpy(newyc->yvalues_int, yc->yvalues_int, yc->nlevels*sizeof(integer_t));
    } else {
        newyc->yvalues_int = NULL;
    }

    if(yc->yvalues_num != NULL){
        newyc->yvalues_num = (numeric_t*)malloc(yc->nlevels*sizeof(numeric_t));
        memcpy(newyc->yvalues_num, yc->yvalues_num, yc->nlevels*sizeof(numeric_t));
    } else {
        newyc->yvalues_num = NULL;
    }

    if(yc->ycuts_num != NULL){
        newyc->ycuts_num = (numeric_t*)malloc(yc->nlevels*sizeof(numeric_t));
        memcpy(newyc->ycuts_num, yc->ycuts_num, yc->nlevels*sizeof(numeric_t));
    } else {
        newyc->ycuts_num = NULL;
    }

    if(yc->ycuts_int != NULL){
        newyc->ycuts_int = (integer_t*)malloc(yc->nlevels*sizeof(integer_t));
        memcpy(newyc->ycuts_int, yc->ycuts_int, yc->nlevels*sizeof(integer_t));
    } else {
        newyc->ycuts_int = NULL;
    }
    if(yc->level_names != NULL){
        newyc->level_names = (char**)malloc(yc->nlevels*sizeof(char*));
        for(int c = 0; c < yc->nlevels; c++){
            newyc->level_names[c] = (char*)malloc(MAX_LEVEL_NAME_LEN*sizeof(char));
            memcpy(newyc->level_names[c], yc->level_names[c], MAX_LEVEL_NAME_LEN);
        }
    } else {
        newyc->level_names = NULL;  // if forget, get a bug
    }
    return(newyc);
}

int countSetBits(bitblock_t n){
    int cnt = 0;
    while (n){
        n &= (n - 1);
        cnt++;
    }
    return cnt;
}

int count1s(bitblock_t *x, int n_blocks, int n_discard_bits){
    int cnt = 0;
    if(n_discard_bits){
        for(int i = 0; i < n_blocks-1; i++){
            cnt += countSetBits(x[i]);
        }
        cnt += countSetBits(x[n_blocks-1] >> n_discard_bits);
    } else {
        for(int i = 0; i < n_blocks; i++){
            cnt += countSetBits(x[i]);
        }
    }
    return(cnt);
}

// shuffle the index array of size n in place, only care about the first ps elements
void shuffle_array_first_ps(int *arr, int n, int ps){
    int temp;
    for(int k = 0; k < MIN(n, ps); k++){
        int index = k + (int)floor(unif_rand()*(n - k));
        temp = arr[k];
        arr[k] = arr[index];
        arr[index] = temp;
    }
}

// find the best split_var and split_bx index; z4 is scratch pad; count, split_var, split_bx contain return values
void find_best_split(bx_info_t *bxall, ycode_t *yc, rf_model_t *model, int min_node_size, int split_search, dt_node_t * cur_node, bitblock_t *cur, int *bindex, int n_sampled_blocks, int *var_index, int actual_ps, bitblock_t *z3, bitblock_t *z4, int *count, int *child_count, int *train_count, int *valid_count, int search_radius, int *candidate_index, int *split_var, int* split_bx){
    bitblock_t **ymat = yc->ymat;
    bitblock_t ***bx = bxall->bx;
    int J = yc->nlevels;
    int ps = actual_ps;
    int n_discard_bits = bxall->n_discard_bits;
    int depth = cur_node->depth;
    int *path_var = cur_node->rulepath_var;
    int *path_bx = cur_node->rulepath_bx;
    int *n_bcols = model->n_bcols;
    char *var_types = model->var_types;
    int nobs = 0;  // number of cases in the current node

    // get the count of classes
    for(int k = 0; k < J; k++){
        for(int i=0; i < n_sampled_blocks; i++){
            z4[i] = ymat[k][bindex[i]] & cur[i];
        }
        count[k] = count1s(z4, n_sampled_blocks, n_discard_bits);
        nobs += count[k]; 
    }
    // if this node is pure, stop splitting
    int node_pure = 0;
    for(int k = 0; k < J; k++){
        if(count[k] == nobs){
            node_pure = 1;
            break;
        }
    }
    if(node_pure == 1){
        *split_var = 0;
        *split_bx = 0;
        return;     
    }

    int best_split_var = 0;
    int best_split_bx = 0;

    if(split_search == 0){
        double best_gini = 1e10;  // avg gini after the split, the smaller the better
        for(int var = 0; var < ps; var++){
            int j = var_index[var];  // j is now the variable index as in data
            int nb = n_bcols[j];  // how many binary columns this variable has
            // determine the available binary columns to serve as potential split rule for the current node
            // for a numeric or an integer variable, knock out unavailable ranges
            // for a factor variable, if the current node is in the left branch of an ancestor that used this factor variable to split, this variable is not eligible any more
            // if the current node is in the right branch of such an ancestor, the used levels in the path are not eligible for selection
            int split_index;
            double total_gini = 1e10;
            if(var_types[j] == 'n' || var_types[j] == 'i'){
                int lower_index = 0;  // inclusive
                int upper_index = nb;  // noninclusive
                // search the split path
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        upper_index = path_bx[d];
                    } else if(-path_var[d] == j){
                        lower_index = path_bx[d] + 1;
                    }
                }
                if(lower_index >= upper_index){
                    continue;  // this var cannot be used
                }
                split_index = lower_index + (int) floor(unif_rand() * (upper_index - lower_index));  // take a random point
                double left_gini;
                double right_gini;
                for(int i = 0; i < n_sampled_blocks; i++){
                    z4[i] = cur[i] & bx[j][split_index][bindex[i]];
                }
                int left_size = count1s(z4, n_sampled_blocks, n_discard_bits);
                int right_size = nobs - left_size;
                if(left_size < min_node_size || right_size < min_node_size){
                    continue;  
                }
                left_gini = 1;
                right_gini = 1;
                // calculate would-be counts for each class in the left node
                for(int k = 0; k < J; k++){
                    for(int i=0; i < n_sampled_blocks; i++){
                        z3[i] = ymat[k][bindex[i]] & z4[i];
                    }
                    child_count[k] = count1s(z3, n_sampled_blocks, n_discard_bits);
                    left_gini -= (1.0*child_count[k]/left_size)*(1.0*child_count[k]/left_size);
                    right_gini -= (1.0*(count[k] - child_count[k])/right_size)*(1.0*(count[k] - child_count[k])/right_size);
                } 
                total_gini = (1.0*left_size/nobs)*left_gini + (1.0*right_size/nobs)*right_gini;  
                if(total_gini < best_gini){
                    best_split_var = j;
                    best_split_bx = split_index;
                    best_gini = total_gini;
                }    
            } else if(var_types[j] == 'f'){
                int n_tabu_levels = 0;
                int pass = 0;
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        pass = 1; 
                        break;
                    } else if(-path_var[d] == j){
                        candidate_index[n_tabu_levels] = path_bx[d];
                        n_tabu_levels += 1;
                    }
                }
                if(pass == 1 || n_tabu_levels == nb){
                    continue;  // this var cannot be used
                }
                int b_pos = n_tabu_levels;
                for(int b = 0; b < nb; b++){
                    int b_good = 1;
                    for(int t = 0; t < n_tabu_levels; t++){
                        if(b == candidate_index[t]){
                            b_good = 0;
                            break;
                        }
                    }
                    if(b_good){
                        candidate_index[b_pos] = b;
                        b_pos += 1;
                    }
                }
                // randomly sample an available index
                split_index = candidate_index[n_tabu_levels + (int)floor(unif_rand() * (nb - n_tabu_levels))];
                
                for(int i = 0; i < n_sampled_blocks; i++){
                    z4[i] = cur[i] & bx[j][split_index][bindex[i]];
                }
                int left_size = count1s(z4, n_sampled_blocks, n_discard_bits);
                int right_size = nobs - left_size;
                if(left_size < min_node_size || right_size < min_node_size){
                    continue;
                }
                double left_gini = 1;
                double right_gini = 1;
                // calculate would-be counts for each class in the left node
                for(int k = 0; k < J; k++){
                    for(int i=0; i < n_sampled_blocks; i++){
                        z3[i] = ymat[k][bindex[i]] & z4[i];
                    }
                    child_count[k] = count1s(z3, n_sampled_blocks, n_discard_bits);
                    left_gini -= (1.0*child_count[k]/left_size)*(1.0*child_count[k]/left_size);
                    right_gini -= (1.0*(count[k] - child_count[k])/right_size)*(1.0*(count[k] - child_count[k])/right_size);
                } 
                total_gini = (1.0*left_size/nobs)*left_gini + (1.0*right_size/nobs)*right_gini;
                if(total_gini < best_gini){
                    best_gini = total_gini;
                    best_split_bx = split_index;
                    best_split_var = j;
                }
            }
        }
    } 
    else if(split_search == 1){
        double best_gini = 1e10;  // avg gini after the split, the smaller the better
        for(int var = 0; var < ps; var++){
            int j = var_index[var];  // j is now the variable index as in data
            int nb = n_bcols[j];  // how many binary columns this variable has
            // determine the available binary columns to serve as potential split rule for the current node
            // for a numeric or an integer variable, knock out unavailable ranges
            // for a factor variable, if the current node is in the left branch of an ancestor that used this factor variable to split, this variable is not eligible any more
            // if the current node is in the right branch of such an ancestor, the used levels in the path are not eligible for selection
            int split_index;
            double total_gini = 1e10;
            if(var_types[j] == 'n' || var_types[j] == 'i'){
                int lower_index = 0;  // inclusive
                int upper_index = nb;  // noninclusive
                // search the split path
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        upper_index = path_bx[d];
                    } else if(-path_var[d] == j){
                        lower_index = path_bx[d] + 1;
                    }
                }
                if(lower_index >= upper_index){
                    continue;  // this var cannot be used
                }
                // neighborhood search: try each candidate in a neighborhood centered around the mid-point
                int mid_index = lower_index + (upper_index - lower_index - 1)/2;  // take the mid point (-1 for binary features)
                int neighbor_lo = MAX(mid_index - search_radius, lower_index);
                int neighbor_up = MIN(mid_index + search_radius, upper_index);
                for(int trial_index = neighbor_lo; trial_index < neighbor_up; trial_index++){
                    double left_gini;
                    double right_gini;
                    for(int i = 0; i < n_sampled_blocks; i++){
                        z4[i] = cur[i] & bx[j][trial_index][bindex[i]];
                    }
                    int left_size = count1s(z4, n_sampled_blocks, n_discard_bits);
                    int right_size = nobs - left_size;
                    if(left_size < min_node_size || right_size < min_node_size){
                        continue;  
                    }
                    left_gini = 1;
                    right_gini = 1;
                    // calculate would-be counts for each class in the left node
                    for(int k = 0; k < J; k++){
                        for(int i=0; i < n_sampled_blocks; i++){
                            z3[i] = ymat[k][bindex[i]] & z4[i];
                        }
                        child_count[k] = count1s(z3, n_sampled_blocks, n_discard_bits);
                        left_gini -= (1.0*child_count[k]/left_size)*(1.0*child_count[k]/left_size);
                        right_gini -= (1.0*(count[k] - child_count[k])/right_size)*(1.0*(count[k] - child_count[k])/right_size);
                    } 
                    total_gini = (1.0*left_size/nobs)*left_gini + (1.0*right_size/nobs)*right_gini;  
                    if(total_gini < best_gini){
                        best_split_var = j;
                        best_split_bx = trial_index;
                        best_gini = total_gini;
                    }    
                }
            } else if(var_types[j] == 'f'){
                int n_tabu_levels = 0;
                int pass = 0;
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        pass = 1; 
                        break;
                    } else if(-path_var[d] == j){
                        candidate_index[n_tabu_levels] = path_bx[d];
                        n_tabu_levels += 1;
                    }
                }
                if(pass == 1 || n_tabu_levels == nb){
                    continue;  // this var cannot be used
                }
                int b_pos = n_tabu_levels;
                for(int b = 0; b < nb; b++){
                    int b_good = 1;
                    for(int t = 0; t < n_tabu_levels; t++){
                        if(b == candidate_index[t]){
                            b_good = 0;
                            break;
                        }
                    }
                    if(b_good){
                        candidate_index[b_pos] = b;
                        b_pos += 1;
                    }
                }
                // randomly sample a few and select the best one
                int a_few = (int)round(sqrt(1.0*(nb - n_tabu_levels)));
                //int a_few = nb - n_tabu_levels;
                shuffle_array_first_ps(candidate_index+n_tabu_levels, nb - n_tabu_levels, a_few);
                for(int t = 0; t < a_few; t++){
                    split_index = candidate_index[n_tabu_levels + t];
                    for(int i = 0; i < n_sampled_blocks; i++){
                        z4[i] = cur[i] & bx[j][split_index][bindex[i]];
                    }
                    int left_size = count1s(z4, n_sampled_blocks, n_discard_bits);
                    int right_size = nobs - left_size;
                    if(left_size < min_node_size || right_size < min_node_size){
                        continue;
                    }
                    double left_gini = 1;
                    double right_gini = 1;
                    // calculate would-be counts for each class in the left node
                    for(int k = 0; k < J; k++){
                        for(int i=0; i < n_sampled_blocks; i++){
                            z3[i] = ymat[k][bindex[i]] & z4[i];
                        }
                        child_count[k] = count1s(z3, n_sampled_blocks, n_discard_bits);
                        left_gini -= (1.0*child_count[k]/left_size)*(1.0*child_count[k]/left_size);
                        right_gini -= (1.0*(count[k] - child_count[k])/right_size)*(1.0*(count[k] - child_count[k])/right_size);
                    } 
                    total_gini = (1.0*left_size/nobs)*left_gini + (1.0*right_size/nobs)*right_gini;
                    if(total_gini < best_gini){
                        best_gini = total_gini;
                        best_split_var = j;
                        best_split_bx = split_index;
                    }
                }
            }
        }
    } 
    else if(split_search == 2){
        // Use half data to evaluate gini and use the other half to test label prediction
        // If the majority class on the validation set does not agree with that on the training set, abandon the candidate split
        double best_gini = 1e10;  // avg gini after the split, the smaller the better
        
        int n_train_blocks = n_sampled_blocks / 2;
        int n_valid_blocks = n_sampled_blocks - n_train_blocks;
        // get the count of classes for the training half and validation half separately    

        memset(train_count, 0, J*sizeof(int));
        memset(valid_count, 0, J*sizeof(int));
        int nobs_train = 0;
        int nobs_valid = 0;

        for(int k = 0; k < J; k++){
            for(int i=0; i < n_sampled_blocks; i++){
                z4[i] = ymat[k][bindex[i]] & cur[i];
            }
            train_count[k] = count1s(z4, n_train_blocks, n_discard_bits);
            nobs_train += train_count[k]; 
            valid_count[k] = count1s(z4+n_train_blocks, n_valid_blocks, n_discard_bits);
            nobs_valid += valid_count[k];
        }

        for(int var = 0; var < ps; var++){
            int j = var_index[var];  // j is now the variable index as in data
            int nb = n_bcols[j];  // how many binary columns this variable has
            // determine the available binary columns to serve as potential split rule for the current node
            // for a numeric or an integer variable, knock out unavailable ranges
            // for a factor variable, if the current node is in the left branch of an ancestor that used this factor variable to split, this variable is not eligible any more
            // if the current node is in the right branch of such an ancestor, the used levels in the path are not eligible for selection
            int split_index;
            double total_gini = 1e11;
            if(var_types[j] == 'n' || var_types[j] == 'i'){
                int lower_index = 0;  // inclusive
                int upper_index = nb;  // noninclusive
                // search the split path
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        upper_index = path_bx[d];
                    } else if(-path_var[d] == j){
                        lower_index = path_bx[d] + 1;
                    }
                }
                if(lower_index >= upper_index){
                    continue;  // this var cannot be used
                }

                split_index = lower_index + (int) floor(unif_rand() * (upper_index - lower_index));  // take a random point
                double train_left_gini;
                double train_right_gini;
                double valid_left_gini;
                double valid_right_gini;
                for(int i = 0; i < n_sampled_blocks; i++){
                    z4[i] = cur[i] & bx[j][split_index][bindex[i]];
                }
                // use first half to train, second half to validate
                int train_left_size = count1s(z4, n_train_blocks, n_discard_bits);
                int train_right_size = nobs_train - train_left_size;
                int valid_left_size = count1s(z4+n_train_blocks, n_valid_blocks, n_discard_bits);
                int valid_right_size = nobs_valid - valid_left_size;

                if(train_left_size < min_node_size || train_right_size < min_node_size
                    || valid_left_size < min_node_size || valid_right_size < min_node_size){
                    continue;  
                }
                train_left_gini = 1;
                train_right_gini = 1;
                valid_left_gini = 1;
                valid_right_gini = 1;
                int train_left_label = 0;
                int train_right_label = 0;
                int valid_left_label = 0;
                int valid_right_label = 0;
                int train_left_max = 0;
                int train_right_max = 0;
                int valid_left_max = 0;
                int valid_right_max = 0;
                // calculate would-be counts for each class in the left node
                for(int k = 0; k < J; k++){
                    for(int i=0; i < n_sampled_blocks; i++){
                        z3[i] = ymat[k][bindex[i]] & z4[i];
                    }
                    // get train part
                    child_count[k] = count1s(z3, n_train_blocks, n_discard_bits);
                    train_left_gini -= (1.0*child_count[k]/train_left_size)*(1.0*child_count[k]/train_left_size);
                    train_right_gini -= (1.0*(train_count[k] - child_count[k])/train_right_size)*(1.0*(train_count[k] - child_count[k])/train_right_size);
                    if(child_count[k] > train_left_max){
                        train_left_max = child_count[k];
                        train_left_label = k;
                    }
                    if(train_count[k] - child_count[k] > train_right_max){
                        train_right_max = train_count[k] - child_count[k];
                        train_right_label = k;
                    }
                    // get valid part
                    child_count[k] = count1s(z3+n_train_blocks, n_valid_blocks, n_discard_bits);
                    valid_left_gini -= (1.0*child_count[k]/valid_left_size)*(1.0*child_count[k]/valid_left_size);
                    valid_right_gini -= (1.0*(valid_count[k] - child_count[k])/valid_right_size)*(1.0*(valid_count[k] - child_count[k])/valid_right_size);
                    if(child_count[k] > valid_left_max){
                        valid_left_max = child_count[k];
                        valid_left_label = k;
                    }
                    if(valid_count[k] - child_count[k] > valid_right_max){
                        valid_right_max = valid_count[k] - child_count[k];
                        valid_right_label = k;
                    }
                } 
                if(train_left_label != valid_left_label || train_right_label != valid_right_label){
                    total_gini = 1e11;
                } else {
                    total_gini = (1.0*train_left_size/nobs_train)*train_left_gini + (1.0*train_right_size/nobs_train)*train_right_gini
                                +(1.0*valid_left_size/nobs_valid)*valid_left_gini + (1.0*valid_right_size/nobs_valid)*valid_right_gini; 
                }

                if(total_gini < best_gini){
                    best_split_var = j;
                    best_split_bx = split_index;
                    best_gini = total_gini;
                }
                
            } else if(var_types[j] == 'f'){
                int n_tabu_levels = 0;
                int pass = 0;
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        pass = 1; 
                        break;
                    } else if(-path_var[d] == j){
                        candidate_index[n_tabu_levels] = path_bx[d];
                        n_tabu_levels += 1;
                    }
                }
                if(pass == 1 || n_tabu_levels == nb){
                    continue;  // this var cannot be used
                }
                int b_pos = n_tabu_levels;
                for(int b = 0; b < nb; b++){
                    int b_good = 1;
                    for(int t = 0; t < n_tabu_levels; t++){
                        if(b == candidate_index[t]){
                            b_good = 0;
                            break;
                        }
                    }
                    if(b_good){
                        candidate_index[b_pos] = b;
                        b_pos += 1;
                    }
                }
                // randomly sample an available index
                split_index = candidate_index[n_tabu_levels + (int)floor(unif_rand() * (nb - n_tabu_levels))];
                double train_left_gini;
                double train_right_gini;
                double valid_left_gini;
                double valid_right_gini;
                for(int i = 0; i < n_sampled_blocks; i++){
                    z4[i] = cur[i] & bx[j][split_index][bindex[i]];
                }
                // use first half to train, second half to validate
                int train_left_size = count1s(z4, n_train_blocks, n_discard_bits);
                int train_right_size = nobs_train - train_left_size;
                int valid_left_size = count1s(z4+n_train_blocks, n_valid_blocks, n_discard_bits);
                int valid_right_size = nobs_valid - valid_left_size;

                if(train_left_size < min_node_size || train_right_size < min_node_size
                    || valid_left_size < min_node_size || valid_right_size < min_node_size){
                    continue;  
                }
                train_left_gini = 1;
                train_right_gini = 1;
                valid_left_gini = 1;
                valid_right_gini = 1;
                int train_left_label = 0;
                int train_right_label = 0;
                int valid_left_label = 0;
                int valid_right_label = 0;
                int train_left_max = 0;
                int train_right_max = 0;
                int valid_left_max = 0;
                int valid_right_max = 0;
                // calculate would-be counts for each class in the left node
                for(int k = 0; k < J; k++){
                    for(int i=0; i < n_sampled_blocks; i++){
                        z3[i] = ymat[k][bindex[i]] & z4[i];
                    }
                    // get train part
                    child_count[k] = count1s(z3, n_train_blocks, n_discard_bits);
                    train_left_gini -= (1.0*child_count[k]/train_left_size)*(1.0*child_count[k]/train_left_size);
                    train_right_gini -= (1.0*(train_count[k] - child_count[k])/train_right_size)*(1.0*(train_count[k] - child_count[k])/train_right_size);
                    if(child_count[k] > train_left_max){
                        train_left_max = child_count[k];
                        train_left_label = k;
                    }
                    if(train_count[k] - child_count[k] > train_right_max){
                        train_right_max = train_count[k] - child_count[k];
                        train_right_label = k;
                    }
                    // get valid part
                    child_count[k] = count1s(z3+n_train_blocks, n_valid_blocks, n_discard_bits);
                    valid_left_gini -= (1.0*child_count[k]/valid_left_size)*(1.0*child_count[k]/valid_left_size);
                    valid_right_gini -= (1.0*(valid_count[k] - child_count[k])/valid_right_size)*(1.0*(valid_count[k] - child_count[k])/valid_right_size);
                    if(child_count[k] > valid_left_max){
                        valid_left_max = child_count[k];
                        valid_left_label = k;
                    }
                    if(valid_count[k] - child_count[k] > valid_right_max){
                        valid_right_max = valid_count[k] - child_count[k];
                        valid_right_label = k;
                    }
                } 
                if(train_left_label != valid_left_label || train_right_label != valid_right_label){
                    total_gini = 1e11;
                } else {
                    total_gini = (1.0*train_left_size/nobs_train)*train_left_gini + (1.0*train_right_size/nobs_train)*train_right_gini
                                +(1.0*valid_left_size/nobs_valid)*valid_left_gini + (1.0*valid_right_size/nobs_valid)*valid_right_gini;  
                }

                if(total_gini < best_gini){
                    best_gini = total_gini;
                    best_split_var = j;
                    best_split_bx = split_index;
                }
            }
        }        
    }
    else if(split_search == 3){
        // Use half data to evaluate gini and use the other half to test label prediction
        // If the majority class on the validation set does not agree with that on the training set, abandon the candidate split
        double best_gini = 1e10;  // avg gini after the split, the smaller the better
        
        int n_train_blocks = n_sampled_blocks / 2;
        int n_valid_blocks = n_sampled_blocks - n_train_blocks;
        // get the count of classes for the training half and validation half separately    

        memset(train_count, 0, J*sizeof(int));
        memset(valid_count, 0, J*sizeof(int));
        int nobs_train = 0;
        int nobs_valid = 0;

        for(int k = 0; k < J; k++){
            for(int i=0; i < n_sampled_blocks; i++){
                z4[i] = ymat[k][bindex[i]] & cur[i];
            }
            train_count[k] = count1s(z4, n_train_blocks, n_discard_bits);
            nobs_train += train_count[k]; 
            valid_count[k] = count1s(z4+n_train_blocks, n_valid_blocks, n_discard_bits);
            nobs_valid += valid_count[k];
        }

        for(int var = 0; var < ps; var++){
            int j = var_index[var];  // j is now the variable index as in data
            int nb = n_bcols[j];  // how many binary columns this variable has
            // determine the available binary columns to serve as potential split rule for the current node
            // for a numeric or an integer variable, knock out unavailable ranges
            // for a factor variable, if the current node is in the left branch of an ancestor that used this factor variable to split, this variable is not eligible any more
            // if the current node is in the right branch of such an ancestor, the used levels in the path are not eligible for selection
            int split_index;
            double total_gini = 1e11;
            if(var_types[j] == 'n' || var_types[j] == 'i'){
                int lower_index = 0;  // inclusive
                int upper_index = nb;  // noninclusive
                // search the split path
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        upper_index = path_bx[d];
                    } else if(-path_var[d] == j){
                        lower_index = path_bx[d] + 1;
                    }
                }
                if(lower_index >= upper_index){
                    continue;  // this var cannot be used
                }

                // neighborhood search: try each candidate in a neighborhood centered around the mid-point
                int mid_index = lower_index + (upper_index - lower_index - 1)/2;  // take the mid point (-1 for binary features)
                int neighbor_lo = MAX(mid_index - search_radius, lower_index);
                int neighbor_up = MIN(mid_index + search_radius, upper_index);
                for(int trial_index = neighbor_lo; trial_index < neighbor_up; trial_index++){
                    double train_left_gini;
                    double train_right_gini;
                    double valid_left_gini;
                    double valid_right_gini;
                    for(int i = 0; i < n_sampled_blocks; i++){
                        z4[i] = cur[i] & bx[j][trial_index][bindex[i]];
                    }
                    // use first half to train, second half to validate
                    int train_left_size = count1s(z4, n_train_blocks, n_discard_bits);
                    int train_right_size = nobs_train - train_left_size;
                    int valid_left_size = count1s(z4+n_train_blocks, n_valid_blocks, n_discard_bits);
                    int valid_right_size = nobs_valid - valid_left_size;

                    if(train_left_size < min_node_size || train_right_size < min_node_size
                        || valid_left_size < min_node_size || valid_right_size < min_node_size){
                        continue;  
                    }
                    train_left_gini = 1;
                    train_right_gini = 1;
                    valid_left_gini = 1;
                    valid_right_gini = 1;
                    int train_left_label = 0;
                    int train_right_label = 0;
                    int valid_left_label = 0;
                    int valid_right_label = 0;
                    int train_left_max = 0;
                    int train_right_max = 0;
                    int valid_left_max = 0;
                    int valid_right_max = 0;
                    // calculate would-be counts for each class in the left node
                    for(int k = 0; k < J; k++){
                        for(int i=0; i < n_sampled_blocks; i++){
                            z3[i] = ymat[k][bindex[i]] & z4[i];
                        }
                        // get train part
                        child_count[k] = count1s(z3, n_train_blocks, n_discard_bits);
                        train_left_gini -= (1.0*child_count[k]/train_left_size)*(1.0*child_count[k]/train_left_size);
                        train_right_gini -= (1.0*(train_count[k] - child_count[k])/train_right_size)*(1.0*(train_count[k] - child_count[k])/train_right_size);
                        if(child_count[k] > train_left_max){
                            train_left_max = child_count[k];
                            train_left_label = k;
                        }
                        if(train_count[k] - child_count[k] > train_right_max){
                            train_right_max = train_count[k] - child_count[k];
                            train_right_label = k;
                        }
                        // get valid part
                        child_count[k] = count1s(z3+n_train_blocks, n_valid_blocks, n_discard_bits);
                        valid_left_gini -= (1.0*child_count[k]/valid_left_size)*(1.0*child_count[k]/valid_left_size);
                        valid_right_gini -= (1.0*(valid_count[k] - child_count[k])/valid_right_size)*(1.0*(valid_count[k] - child_count[k])/valid_right_size);
                        if(child_count[k] > valid_left_max){
                            valid_left_max = child_count[k];
                            valid_left_label = k;
                        }
                        if(valid_count[k] - child_count[k] > valid_right_max){
                            valid_right_max = valid_count[k] - child_count[k];
                            valid_right_label = k;
                        }
                    } 
                    if(train_left_label != valid_left_label || train_right_label != valid_right_label){
                        total_gini = 1e11;
                    } else {
                        total_gini = (1.0*train_left_size/nobs_train)*train_left_gini + (1.0*train_right_size/nobs_train)*train_right_gini
                                    +(1.0*valid_left_size/nobs_valid)*valid_left_gini + (1.0*valid_right_size/nobs_valid)*valid_right_gini; 
                    }

                    if(total_gini < best_gini){
                        best_split_var = j;
                        best_split_bx = trial_index;
                        best_gini = total_gini;
                    }
                }
            } else if(var_types[j] == 'f'){
                int n_tabu_levels = 0;
                int pass = 0;
                for(int d = 0; d < depth; d++){
                    if(path_var[d] == j){
                        pass = 1; 
                        break;
                    } else if(-path_var[d] == j){
                        candidate_index[n_tabu_levels] = path_bx[d];
                        n_tabu_levels += 1;
                    }
                }
                if(pass == 1 || n_tabu_levels == nb){
                    continue;  // this var cannot be used
                }
                int b_pos = n_tabu_levels;
                for(int b = 0; b < nb; b++){
                    int b_good = 1;
                    for(int t = 0; t < n_tabu_levels; t++){
                        if(b == candidate_index[t]){
                            b_good = 0;
                            break;
                        }
                    }
                    if(b_good){
                        candidate_index[b_pos] = b;
                        b_pos += 1;
                    }
                }
                // randomly sample a few and select the best one
                int a_few = (int)round(sqrt(1.0*(nb - n_tabu_levels)));
                //int a_few = nb - n_tabu_levels;
                shuffle_array_first_ps(candidate_index+n_tabu_levels, nb - n_tabu_levels, a_few);
                for(int t = 0; t < a_few; t++){
                    split_index = candidate_index[n_tabu_levels + t];

                    double train_left_gini;
                    double train_right_gini;
                    double valid_left_gini;
                    double valid_right_gini;
                    for(int i = 0; i < n_sampled_blocks; i++){
                        z4[i] = cur[i] & bx[j][split_index][bindex[i]];
                    }
                    // use first half to train, second half to validate
                    int train_left_size = count1s(z4, n_train_blocks, n_discard_bits);
                    int train_right_size = nobs_train - train_left_size;
                    int valid_left_size = count1s(z4+n_train_blocks, n_valid_blocks, n_discard_bits);
                    int valid_right_size = nobs_valid - valid_left_size;

                    if(train_left_size < min_node_size || train_right_size < min_node_size
                        || valid_left_size < min_node_size || valid_right_size < min_node_size){
                        continue;  
                    }
                    train_left_gini = 1;
                    train_right_gini = 1;
                    valid_left_gini = 1;
                    valid_right_gini = 1;
                    int train_left_label = 0;
                    int train_right_label = 0;
                    int valid_left_label = 0;
                    int valid_right_label = 0;
                    int train_left_max = 0;
                    int train_right_max = 0;
                    int valid_left_max = 0;
                    int valid_right_max = 0;
                    // calculate would-be counts for each class in the left node
                    for(int k = 0; k < J; k++){
                        for(int i=0; i < n_sampled_blocks; i++){
                            z3[i] = ymat[k][bindex[i]] & z4[i];
                        }
                        // get train part
                        child_count[k] = count1s(z3, n_train_blocks, n_discard_bits);
                        train_left_gini -= (1.0*child_count[k]/train_left_size)*(1.0*child_count[k]/train_left_size);
                        train_right_gini -= (1.0*(train_count[k] - child_count[k])/train_right_size)*(1.0*(train_count[k] - child_count[k])/train_right_size);
                        if(child_count[k] > train_left_max){
                            train_left_max = child_count[k];
                            train_left_label = k;
                        }
                        if(train_count[k] - child_count[k] > train_right_max){
                            train_right_max = train_count[k] - child_count[k];
                            train_right_label = k;
                        }
                        // get valid part
                        child_count[k] = count1s(z3+n_train_blocks, n_valid_blocks, n_discard_bits);
                        valid_left_gini -= (1.0*child_count[k]/valid_left_size)*(1.0*child_count[k]/valid_left_size);
                        valid_right_gini -= (1.0*(valid_count[k] - child_count[k])/valid_right_size)*(1.0*(valid_count[k] - child_count[k])/valid_right_size);
                        if(child_count[k] > valid_left_max){
                            valid_left_max = child_count[k];
                            valid_left_label = k;
                        }
                        if(valid_count[k] - child_count[k] > valid_right_max){
                            valid_right_max = valid_count[k] - child_count[k];
                            valid_right_label = k;
                        }
                    } 
                    if(train_left_label != valid_left_label || train_right_label != valid_right_label){
                        total_gini = 1e11;
                    } else {
                        total_gini = (1.0*train_left_size/nobs_train)*train_left_gini + (1.0*train_right_size/nobs_train)*train_right_gini
                                    +(1.0*valid_left_size/nobs_valid)*valid_left_gini + (1.0*valid_right_size/nobs_valid)*valid_right_gini;  
                    }

                    if(total_gini < best_gini){
                        best_gini = total_gini;
                        best_split_var = j;
                        best_split_bx = split_index;
                    }
                }
            }
        }        
    }

    *split_var = best_split_var;
    *split_bx = best_split_bx;
}

// fill the array with n index numbers independently sampled from [0,1,..., n-1]
void bootstrap_index_array(int n, int *array){
    for(int i = 0; i < n; i++){
        array[i] = (int)floor(unif_rand()*n);
    }
}

dt_node_t* build_tree(bx_info_t *bxall, ycode_t *yc, rf_model_t *model, int ps, int max_depth, int min_node_size, int bagging_method, double bagging_proportion, int split_search, int search_radius){
    // unpack inputs
    bitblock_t ***bx = bxall->bx;  
    int *n_bcols = model->n_bcols;
    char *var_types = model->var_types;
    int n_blocks = bxall->n_blocks;
    int n_discard_bits = bxall->n_discard_bits;
    int p = model->p;
    int J = yc->nlevels;
    bitblock_t **ymat = yc->ymat;

    // prepare bagging
    int n_sampled_blocks = MIN((int) MAX(floor(bagging_proportion * n_blocks),1), n_blocks);
    if(n_sampled_blocks < 2) return(NULL); // Need at least two sampled blocks

    int *bindex = (int*)malloc(n_blocks*sizeof(int));
    for(int i = 0; i < n_blocks; i++){
        bindex[i] = i;
    }
    if(bagging_method){
        bootstrap_index_array(n_blocks, bindex);
    } else {
        shuffle_array_first_ps(bindex, n_blocks, n_blocks);  // shuffle the entire array, though take the first n_sampled_blocks
    }
    
    // prepare stuff
    bitblock_t *z3 = (bitblock_t*)malloc(n_sampled_blocks*sizeof(bitblock_t));
    bitblock_t *z4 = (bitblock_t*)malloc(n_sampled_blocks*sizeof(bitblock_t));
    bitblock_t *cur = (bitblock_t*)malloc(n_sampled_blocks*sizeof(bitblock_t));
    int *child_count = (int*)malloc(J*sizeof(int));
    int *train_count = NULL;
    int *valid_count = NULL;
    if(split_search >= 2){
        train_count = (int*)malloc(J*sizeof(int));
        valid_count = (int*)malloc(J*sizeof(int));
    }

    dt_node_t* queue[MAXNODES]; 
    int head = 0; 
    int tail = 0;
    dt_node_t* parent = NULL;
    dt_node_t* root = NULL;
    dt_node_t* cur_node = NULL;
    int split_var, split_bx;
    int *count = (int*)malloc(J*sizeof(int));  // holds the count of classes in a node
    memset(count, 0, J*sizeof(int));

    int max_factor_nlevels = 0;
    int actual_p = 0;
    for(int j = 1; j <= p; j++){
        if(n_bcols[j] > 0) actual_p++;
        if(var_types[j] == 'f'){
            if(n_bcols[j] > max_factor_nlevels){
                max_factor_nlevels = n_bcols[j];
            }
        }
    }
    int *var_index = (int*)malloc(actual_p*sizeof(int));
    int cnt_p = 0;
    for(int j = 1; j <= p; j++){
        if(n_bcols[j] == 0) continue;
        var_index[cnt_p++] = j;
    }
    
    int *candidate_index = NULL;
    if(max_factor_nlevels > 0){
        candidate_index = (int*)malloc(max_factor_nlevels*sizeof(int));
    }
    
    // if actual_ps is too small (e.g., 1 or 2) nodes may become leaf prematurely
    int actual_ps = MIN(ps, actual_p);
    if(actual_p < 7 && actual_ps < actual_p / 2 + 1){
        actual_ps = actual_p / 2 + 1;
    }

    // construct root node
    cur_node = newNode(parent, J, 0);
    root = cur_node;
    for(int i = 0; i < n_sampled_blocks; i++){
        cur[i] = MAXBITBLOCK_VALUE;  // all bits are 1 (assuming bitblock_t is unsigned int)
    }
    shuffle_array_first_ps(var_index, actual_p, ps);
    find_best_split(bxall, yc, model, min_node_size, split_search, cur_node, cur, bindex, n_sampled_blocks, var_index, actual_ps, z3, z4, count, child_count, train_count, valid_count, search_radius, candidate_index, &split_var, &split_bx);
    cur_node->split_var = split_var; 
    cur_node->split_bx = split_bx;
    for(int k = 0; k < J; k++){
        cur_node->count[k] = count[k];
    }
    if(cur_node->split_var!=0 && cur_node->depth < max_depth){
        // put the node in queue
        queue[tail++] = cur_node;
    }

    // if queue is not empty, process the node at queue head
    while(tail > head){
        // if the buffer is full, shift to make room
        if(tail >= MAXNODES - 2){
            for(int i = 0; i < tail - head; i++){
                queue[i] = queue[head+i];
            }

            tail = tail - head;
            head = 0;
        }
        parent = queue[head++];
        // create its left child
        cur_node = newNode(parent, J, LEFT);
        parent->left = cur_node;
        for(int i = 0; i < n_sampled_blocks; i++){
            cur[i] = MAXBITBLOCK_VALUE;
            for(int d = 0; d < cur_node->depth; d++){
                int this_var = cur_node->rulepath_var[d];
                int this_bx = cur_node->rulepath_bx[d];
                if(this_var > 0){
                    cur[i] &= bx[this_var][this_bx][bindex[i]];
                } else if(this_var < 0){
                    cur[i] &= ~bx[-this_var][this_bx][bindex[i]];
                } else {
                    //printf("Impossible\n");
                }
            }
        }
        if(cur_node->depth < max_depth){
            shuffle_array_first_ps(var_index, actual_p, ps);
            find_best_split(bxall, yc, model, min_node_size, split_search, cur_node, cur, bindex, n_sampled_blocks, var_index, actual_ps, z3, z4, count, child_count, train_count, valid_count, search_radius, candidate_index, &split_var, &split_bx);
            cur_node->split_var = split_var; 
            cur_node->split_bx = split_bx;
            for(int k = 0; k < J; k++){
                cur_node->count[k] = count[k];
            }
        } else {
            // get the count of classes
            for(int k = 0; k < J; k++){
                for(int i=0; i < n_sampled_blocks; i++){
                    z4[i] = ymat[k][bindex[i]] & cur[i];
                }
                cur_node->count[k] = count1s(z4, n_sampled_blocks, n_discard_bits);
            }
            cur_node->split_var = 0;
            cur_node->split_bx = 0;
        }
        if(cur_node->split_var!=0){
            // put the node in queue
            queue[tail++] = cur_node;
        }

        // create its right child
        cur_node = newNode(parent, J, RIGHT);
        parent->right = cur_node;
        for(int i = 0; i < n_sampled_blocks; i++){
            cur[i] = MAXBITBLOCK_VALUE;
            for(int d = 0; d < cur_node->depth; d++){
                int this_var = cur_node->rulepath_var[d];
                int this_bx = cur_node->rulepath_bx[d];
                if(this_var > 0){
                    cur[i] &= bx[this_var][this_bx][bindex[i]];
                } else if(this_var < 0){
                    cur[i] &= ~bx[-this_var][this_bx][bindex[i]];
                } else {
                    //printf("Impossible\n");
                }
            }
        }
        if(cur_node->depth < max_depth){
            shuffle_array_first_ps(var_index, actual_p, ps);
            find_best_split(bxall, yc, model, min_node_size, split_search, cur_node, cur, bindex, n_sampled_blocks, var_index, actual_ps, z3, z4, count, child_count, train_count, valid_count, search_radius, candidate_index, &split_var, &split_bx);
            cur_node->split_var = split_var; 
            cur_node->split_bx = split_bx;
            for(int k = 0; k < J; k++){
                cur_node->count[k] = count[k];
            }
        } else {
            // get the count of classes
            for(int k = 0; k < J; k++){
                for(int i=0; i < n_sampled_blocks; i++){
                    z4[i] = ymat[k][bindex[i]] & cur[i];
                }
                cur_node->count[k] = count1s(z4, n_sampled_blocks, n_discard_bits);
            }
            cur_node->split_var = 0;
            cur_node->split_bx = 0;
        }
        if(cur_node->split_var!=0){
            // put the node in queue
            queue[tail++] = cur_node;
        }
    }

    // clean up
    if(candidate_index != NULL) free(candidate_index);
    free(cur);
    free(z3);
    free(z4);
    free(child_count);
    if(split_search >= 2){
        free(train_count);
        free(valid_count);
    }
    free(bindex);
    free(var_index);
    free(count);
    return(root);
}


void predict_tree(dt_node_t *tree, bitblock_t ***bx, int **pred_tree, int J, int n_blocks){
    if(tree){
        if(tree->split_var == 0){
            int i;
            #pragma omp parallel for
            for(i = 0; i < n_blocks; i++){
                bitblock_t test0 = MAXBITBLOCK_VALUE;
                for(int d = 0; d < tree->depth; d++){
                    int this_var = tree->rulepath_var[d];
                    int this_bx = tree->rulepath_bx[d];
                    if(this_var > 0){
                        test0 &= bx[this_var][this_bx][i];
                    } else if(this_var < 0){
                        test0 &= ~bx[-this_var][this_bx][i];
                    } else {
                        //printf("Impossible\n");
                    }
                }
                // set score for the block
                unsigned bit = 0;
                for(unsigned k = 1 << (8*sizeof(bitblock_t) - 1); k > 0; k = k / 2){
                    if(test0 & k){
                        for(int j = 0; j < J; j++){
                            pred_tree[j][i*8*sizeof(bitblock_t)+bit] = tree->count[j];
                        }
                    }
                    bit++;
                }
            }
        } else {
            predict_tree(tree->left, bx, pred_tree, J, n_blocks);
            predict_tree(tree->right, bx, pred_tree, J, n_blocks);
        }
    }
}

void predict_leaves(dt_leaf_t *leaves, bitblock_t ***bx, int **pred_tree, int J, int n_blocks){
    if(leaves){
        for(int i = 0; i < n_blocks; i++){
            bitblock_t test0 = MAXBITBLOCK_VALUE;
            for(int d = 0; d < leaves->depth; d++){
                int this_var = leaves->rulepath_var[d];
                int this_bx = leaves->rulepath_bx[d];
                if(this_var > 0){
                    test0 &= bx[this_var][this_bx][i];
                } else if(this_var < 0){
                    test0 &= ~bx[-this_var][this_bx][i];
                } else {
                    //printf("Impossible\n");
                }
            }
            // set score for the block
            unsigned bit = 0;
            for(unsigned k = 1 << (8*sizeof(bitblock_t) - 1); k > 0; k = k / 2){
                if(test0 & k){
                    for(int j = 0; j < J; j++){
                        pred_tree[j][i*8*sizeof(bitblock_t)+bit] = leaves->count[j];
                    }
                }
                bit++;
            }
        }
        predict_leaves(leaves->next, bx, pred_tree, J, n_blocks);
    }
}


void predict(rf_model_t *model, bx_info_t * bx_new, double **pred, int vote_method, int nthreads){    
    if(model == NULL || model->ntrees == 0) return;
    // unpack parameters
    int J = (model->yc)->nlevels;
    int n = bx_new->n;
    int n_blocks = bx_new->n_blocks;
    bitblock_t ***bx = bx_new->bx; 

    int **pred_tree = (int**)malloc(J*sizeof(int*));
    for(int k = 0; k < J; k++){
        pred_tree[k] = (int*)malloc(n_blocks*8*sizeof(bitblock_t)*sizeof(int));  // size must be this, not simply n
        memset(pred_tree[k], 0, n_blocks*8*sizeof(bitblock_t)*sizeof(int));
        memset(pred[k], 0, n*sizeof(double));  // clear the output
    }

    for(int t = 0; t < model->ntrees; t++){
        //predict_tree(model->trees[t], bx, pred_tree, J, n_blocks);
        predict_leaves(model->tree_leaves[t], bx, pred_tree, J, n_blocks);

        if(vote_method == 0){
            for(int i = 0; i < n; i++){
                for(int k = 0; k < J; k++){
                    pred[k][i] += pred_tree[k][i];
                }                
            }
        } else {
            double this_sum = 0;
            for(int i = 0; i < n; i++){
                this_sum = 0;
                for(int k = 0; k < J; k++){
                    this_sum += pred_tree[k][i];
                }
                for(int k = 0; k < J; k++){
                    pred[k][i] += 1.0 * pred_tree[k][i] / this_sum;
                }                
            }

        }

    }
    
    // average the predictions
    if(vote_method == 0){
        double total_count = 0;
        for(int i = 0; i < n; i++){
            total_count = 0;
            for(int k = 0; k < J; k++){
                total_count += pred[k][i];
            }
            for(int k = 0; k < J; k++){
                pred[k][i] = pred[k][i] / total_count;
            }
        }
    } else {
        for(int i = 0; i < n; i++){
            for(int k = 0; k < J; k++){
                pred[k][i] = pred[k][i] / model->ntrees;
            }
        }        
    }

    // release temp memory
    for(int k = 0; k < J; k++){
        free(pred_tree[k]);
    }
    free(pred_tree);  
}

rf_model_t *create_empty_model(void){
    rf_model_t * m = (rf_model_t*)malloc(sizeof(rf_model_t));
    m->p = 0;  // number of predictors
    m->var_types = NULL;  // type designation character (p+1)
    m->var_labels = NULL;  // variable names (p+1)
    m->n_bcols = NULL;  // number of binary features of each variable (p+1)
    m->ntrees = 0;  // number of trees built
    m->index_in_group = NULL;  // index of each variable in its type group (p+1)
    m->numeric_cuts = NULL;
    m->integer_cuts = NULL;
    m->factor_cuts = NULL; 
    m->n_num_vars = 0;
    m->n_int_vars = 0;
    m->n_fac_vars = 0;
    m->trees = NULL;    
    m->tree_leaves = NULL;
    m->yc = NULL;
    return(m);
}

void get_numeric_summary(numeric_t *vector, int n, numeric_t *min_val, numeric_t *max_val, numeric_t *avg_val){
    *min_val = 1e20;
    *max_val = -1e20;
    *avg_val = 0;
    for(int i = 0; i < n; i++){
        *avg_val += vector[i];
        if(vector[i] < *min_val) *min_val = vector[i];
        if(vector[i] > *max_val) *max_val = vector[i];
    }
    *avg_val = *avg_val / n;
}

void get_integer_summary(integer_t *vector, int n, integer_t *min_val, integer_t *max_val, numeric_t *avg_val){
    *min_val = INTEGER_T_MAX;
    *max_val = INTEGER_T_MIN;
    *avg_val = 0;
    for(int i = 0; i < n; i++){
        *avg_val += vector[i];
        if(vector[i] < *min_val) *min_val = vector[i];
        if(vector[i] > *max_val) *max_val = vector[i];
    }
    *avg_val = 1.0*(*avg_val) / n;
}

void delete_data(data_frame_t *df){
    if(df == NULL) return;
    for(int j = 0; j <= df->p; j++){
        if(df->var_types[j] == 'f'){
            if(df->data[j] != NULL)
                delete_factor((factor_t*)df->data[j]);
        } else if(df->var_types[j] == 'i' || df->var_types[j] == 'n'){
            if(df->data[j] != NULL)
                free(df->data[j]);
        }
        if(df->var_labels[j] != NULL) free(df->var_labels[j]);
    }
    free(df->var_types);
    free(df->var_labels);
    free(df->data);
    free(df);
}

void delete_yc(ycode_t * yc){
    if(yc->ycuts_int != NULL) free(yc->ycuts_int);
    if(yc->ycuts_num != NULL) free(yc->ycuts_num);
    if(yc->yvalues_int != NULL) free(yc->yvalues_int);
    if(yc->yvalues_num != NULL) free(yc->yvalues_num);
    if(yc->yavg != NULL) free(yc->yavg);
    if(yc->ymat != NULL){
        for(int c = 0; c < yc->nlevels; c++){
            if(yc->ymat[c] != NULL)
                free(yc->ymat[c]);
        }
        free(yc->ymat);
    }
    if(yc->level_names != NULL){
        for(int c = 0; c < yc->nlevels; c++){
            free(yc->level_names[c]);
        }
        free(yc->level_names);
    }
    free(yc);
}

void delete_model(rf_model_t *model){
    if(model == NULL) return;
    if(model->p <= 0) return;
    if(model->n_bcols != NULL){
        int this_int_var = 0;
        int this_num_var = 0;
        int this_fac_var = 0;
        for(int j = 1; j <= model->p; j++){
            if(model->var_types[j] == 'n'){
                if(model->n_bcols[j] > 0){
                    free(model->numeric_cuts[this_num_var]);
                }
                this_num_var += 1;
            } else if(model->var_types[j] == 'i'){
                if(model->n_bcols[j] > 0){
                    free(model->integer_cuts[this_int_var]);
                }
                this_int_var += 1;
            } else if(model->var_types[j] == 'f'){
                if(model->n_bcols[j] > 0){
                    delete_factor(model->factor_cuts[this_fac_var]);
                }
                this_fac_var += 1;
            }            
        }
    }
    if(model->numeric_cuts != NULL) free(model->numeric_cuts);  // dont' forget this
    if(model->integer_cuts != NULL) free(model->integer_cuts);
    if(model->factor_cuts != NULL) free(model->factor_cuts); 
    if(model->var_labels != NULL){
        for(int j = 0; j <= model->p; j++){
            if(model->var_labels[j] != NULL) free(model->var_labels[j]);
        }
        free(model->var_labels);
    }
    if(model->var_types != NULL) free(model->var_types);
    if(model->index_in_group != NULL) free(model->index_in_group);
    if(model->n_bcols != NULL) free(model->n_bcols);
    if(model->yc != NULL) delete_yc(model->yc);
    if(model->trees != NULL){
        for(int t = 0; t < model->ntrees; t++){
            if(model->trees[t] != NULL) deleteTree(model->trees[t]);
        }
        free(model->trees);
    }
    if(model->tree_leaves != NULL){
        for(int t = 0; t < model->ntrees; t++){
            if(model->tree_leaves[t] != NULL) deleteLeaves(model->tree_leaves[t]);
        }
        free(model->tree_leaves);
    }
    // more to free
    free(model);
}

void make_cuts(data_frame_t *train, rf_model_t **model, int n_numeric_cuts, int n_integer_cuts){
    if(train == NULL || *model == NULL) return;
    if(train->p != (*model)->p){
        //printf("Train p (%d) and model p (%d) are not equal.\n", train->p, (*model)->p);
        return;
    }
    int p = train->p;
    int n = train->n;
    int var_types_ok = 1;
    for(int j = 1; j <= p; j++){
        if(strcmp(train->var_labels[j], (*model)->var_labels[j])){
            var_types_ok = 0;
            break;
        }
        if(train->var_types[j] != (*model)->var_types[j]){
            var_types_ok = 0;
            break;
        }
    }
    if(!var_types_ok){
        //printf("Metadata between train and model are inconsistent.\n");
        return;          
    }

    char *var_types = train->var_types;
    (*model)->index_in_group = (int*)malloc((p+1)*sizeof(int));
    (*model)->index_in_group[0] = 0;
    int n_int_vars = 0;
    int n_fac_vars = 0;
    int n_num_vars = 0;
    for(int j = 1; j <= p; j++){
        if(var_types[j] == 'f') (*model)->index_in_group[j] = n_fac_vars++;
        else if(var_types[j] == 'n') (*model)->index_in_group[j] = n_num_vars++;
        else if(var_types[j] == 'i') (*model)->index_in_group[j] = n_int_vars++;
    }
    (*model)->n_fac_vars = n_fac_vars;
    (*model)->n_num_vars = n_num_vars;
    (*model)->n_int_vars = n_int_vars;

    numeric_t **numeric_cuts = (numeric_t**)malloc(n_num_vars*sizeof(numeric_t*));
    integer_t **integer_cuts = (integer_t**)malloc(n_int_vars*sizeof(integer_t*));
    factor_t **factor_cuts = (factor_t**)malloc(n_fac_vars*sizeof(factor_t*));
    int *n_bcols = (int*)malloc((p+1)*sizeof(int));  // number of binary columns from the original variables including the response var
    memset(n_bcols, 0, (p+1)*sizeof(int));

    int this_int_var = 0;
    int this_num_var = 0;
    int this_fac_var = 0;
    for(int j = 1; j <= p; j++){
        if(var_types[j] == 'n'){
            n_bcols[j] = n_numeric_cuts;
            if(var_types[0] == 'f'){
                numeric_cuts[this_num_var] = numeric_cutpoints_2((numeric_t *)train->data[j], n, &(n_bcols[j]), ((factor_t*)(train->data[0]))->index, ((factor_t*)(train->data[0]))->nlevels, ((factor_t*)(train->data[0]))->start_index);
            } else {
                numeric_cuts[this_num_var] = numeric_cutpoints((numeric_t *)train->data[j], n, &(n_bcols[j]));
            }
            this_num_var += 1;
        } else if(var_types[j] == 'i'){
            n_bcols[j] = n_integer_cuts;
            if(var_types[0] == 'f'){
                integer_cuts[this_int_var] = integer_cutpoints_2((integer_t *)train->data[j], n, &(n_bcols[j]), ((factor_t*)(train->data[0]))->index, ((factor_t*)(train->data[0]))->nlevels, ((factor_t*)(train->data[0]))->start_index);
            } else {
                integer_cuts[this_int_var] = integer_cutpoints((integer_t *)train->data[j], n, &(n_bcols[j]));
            }
            this_int_var += 1;
        } else if(var_types[j] == 'f'){
            factor_t * f = (factor_t *) train->data[j];
            n_bcols[j] = f->nlevels;
            factor_cuts[this_fac_var] = factor_cutpoints(f, n, &(n_bcols[j]));
            this_fac_var += 1;
        }
    }

    // populate the model a little more
    (*model)->numeric_cuts = numeric_cuts;
    (*model)->integer_cuts = integer_cuts;
    (*model)->factor_cuts = factor_cuts;
    (*model)->n_bcols = n_bcols;
}

bx_info_t * make_bx(data_frame_t * train, rf_model_t ** model){
    int p = train->p;
    int n = train->n;
    int n_blocks = n / (8*sizeof(bitblock_t)) + ((n % (8*sizeof(bitblock_t))) ? 1 : 0);
    int n_discard_bits = (n % (8*sizeof(bitblock_t)) == 0) ? 0 : (8*sizeof(bitblock_t) - n % (8*sizeof(bitblock_t)));
    if(n_discard_bits != 0){
        //Rprintf("Warning: n_discard_bit = %d\n", n_discard_bits);
    }
    char *var_types = train->var_types;
    bitblock_t ***bx = (bitblock_t ***)malloc((p+1)*sizeof(bitblock_t**));  
    bx[0] = NULL; // the 0th one is unused
    
    int this_int_var = 0;
    int this_num_var = 0;
    for(int j = 1; j <= p; j++){
        if(var_types[j] == 'n'){
            bx[j] = binarize_numeric((numeric_t*)train->data[j], (*model)->numeric_cuts[this_num_var], n, n_blocks, (*model)->n_bcols[j]);
            this_num_var += 1;
        } else if(var_types[j] == 'i'){
            bx[j] = binarize_integer((integer_t*)train->data[j], (*model)->integer_cuts[this_int_var], n, n_blocks, (*model)->n_bcols[j]);
            this_int_var += 1;
        } else if(var_types[j] == 'f'){
            factor_t *f = (factor_t *)train->data[j];
            bx[j] = binarize_factor_index(f->index, n, n_blocks, (*model)->n_bcols[j], f->start_index);
        }
    }
    bx_info_t * bxall = (bx_info_t*)malloc(sizeof(bx_info_t));
    bxall->bx = bx;
    bxall->n = n;
    bxall->n_blocks = n_blocks;
    bxall->n_discard_bits = n_discard_bits;
    return(bxall);
}

void delete_bx(bx_info_t *bxall, rf_model_t *model){
    if(bxall == NULL) return;
    if(model == NULL){
        //printf("Cannot delete bx because of null model.\n");
        return;
    }
    int *n_bcols = model->n_bcols;
    for(int j = 1; j <= model->p; j++){
        if(n_bcols[j] > 0){
            for(int c = 0; c < n_bcols[j]; c++){
                free(bxall->bx[j][c]);
            }
            free(bxall->bx[j]);
        }
    }
    free(bxall->bx);
    free(bxall);
}

ycode_t * make_yc(data_frame_t *train, rf_model_t **model, int max_integer_classes){
    if(train == NULL || *model == NULL || (*model)->n_bcols == NULL) return NULL;
    char *var_types = (*model)->var_types;
    int n = train->n;
    int n_blocks = n / (8*sizeof(bitblock_t)) + ((n % (8*sizeof(bitblock_t))) ? 1 : 0);
    ycode_t *yc = NULL;
    if(var_types[0] == 'i'){
        yc = codify_integer_target((integer_t*)train->data[0], n, n_blocks, max_integer_classes);
        (*model)->n_bcols[0] = yc->nlevels;
    } else if(var_types[0] == 'f'){
        yc = codify_factor_target((factor_t*)train->data[0], n, n_blocks, max_integer_classes);
        (*model)->n_bcols[0] = yc->nlevels;
    } else if(var_types[0] == 'n'){
        yc = codify_numeric_target((numeric_t*)train->data[0], n, n_blocks, max_integer_classes);
        (*model)->n_bcols[0] = yc->nlevels;
    } else {
        //printf("var_type wrong.\n");
    }
    // copy yc to model (exclude ymat)
    (*model)->yc = copy_ycode(yc);
    return(yc);
}

void build_forest(bx_info_t *bxall, ycode_t *yc, rf_model_t **model, int ps, int max_depth, int min_node_size, int ntrees, int nthreads, int bagging_method, double bagging_proportion, int split_search, int search_radius){
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    max_depth = MIN(MAXDEPTH, max_depth);
    ps = MIN(ps, (*model)->p);
    min_node_size = MAX(1, min_node_size);
    ntrees = MAX(1, ntrees);
    if((*model)->ntrees > 0){
        //printf("Model already contains a forest.\n");
        return;
    }
    dt_node_t **trees = (dt_node_t**)malloc(ntrees*sizeof(dt_node_t*));
    int t;
    #pragma omp parallel for
    for(t = 0; t < ntrees; t++){
        // if split_search is >= 2, alternate between 0 and 1
        if(split_search >=4){
            split_search = t % 4;
        }
        trees[t] = build_tree(bxall, yc, *model, ps, max_depth, min_node_size, bagging_method, bagging_proportion, split_search, search_radius);
    }

    (*model)->ntrees = ntrees;
    (*model)->trees = trees;
}

void flatten_tree(dt_node_t *tree, dt_leaf_t **leaves, int J){
  if(tree){
    if(tree->split_var == 0){
        dt_leaf_t * new_leaf = (dt_leaf_t*)malloc(sizeof(dt_leaf_t));
        new_leaf->count = (int*)malloc(J*sizeof(int));
        memcpy(new_leaf->count, tree->count, J*sizeof(int));
        new_leaf->depth = tree->depth;
        memcpy(new_leaf->rulepath_var, tree->rulepath_var, tree->depth*sizeof(int));
        memcpy(new_leaf->rulepath_bx, tree->rulepath_bx, tree->depth*sizeof(int));
        new_leaf->next = *leaves;
        *leaves = new_leaf;
      } else {
        flatten_tree(tree->left, leaves, J);
        flatten_tree(tree->right, leaves, J);
      }
  }
}

void flatten_model(rf_model_t **model, int nthreads){
    if((*model)->trees == NULL || (*model)->tree_leaves != NULL){
        //printf("Cannot flatten model \n");
        return;
    }
    (*model)->tree_leaves = (dt_leaf_t**)malloc((*model)->ntrees*sizeof(dt_leaf_t*));
    int k;
    for(k = 0; k < (*model)->ntrees; k++){
        (*model)->tree_leaves[k] = NULL;
        flatten_tree((*model)->trees[k], &((*model)->tree_leaves[k]), (*model)->yc->nlevels);
        // delete the tree
        deleteTree((*model)->trees[k]);
        (*model)->trees[k] = NULL;
    }
    free((*model)->trees);
    (*model)->trees = NULL;
}

void printTree(dt_node_t * tree, unsigned indent, int J){
    if(tree) {
        for(unsigned i = 0; i < indent; i++){
            Rprintf(" ");
        }
        for(int k = 0; k < J; k++){
            Rprintf("%d ", tree->count[k]);
        }
        Rprintf("split = (%d, %d)\n", tree->split_var, tree->split_bx);
        printTree(tree->left, indent + 3, J);
        printTree(tree->right, indent + 3, J);
    }
}

void fill_name_addr_array(fnode_t *tree, char **name, int start_index){
    if(tree != NULL){
        name[tree->val - start_index] = tree->name;
        fill_name_addr_array(tree->left, name, start_index);
        fill_name_addr_array(tree->right, name, start_index);
    }
}

void printRules(rf_model_t *model, int which_tree){
    if(which_tree > ((model->ntrees)-1)) return;
    dt_leaf_t * lf = model->tree_leaves[which_tree];
    if(lf == NULL){
        return;
    }
    
    char ***fac_level_names = (char***)malloc(model->n_fac_vars*sizeof(char**));
    for(int j = 1; j <= model->p; j++){
        if(model->var_types[j] == 'f'){
            fac_level_names[model->index_in_group[j]] = (char**)malloc(model->n_bcols[j]*sizeof(char*));
            fill_name_addr_array(model->factor_cuts[model->index_in_group[j]]->levels, fac_level_names[model->index_in_group[j]], model->factor_cuts[model->index_in_group[j]]->start_index);
        }
    }
    char sign[3];
    char is_or_not[7];
    int nrules = 0;
    while(lf != NULL){
        Rprintf("Rule %d: ", nrules++);
        Rprintf("If ");
        for(int d = 0; d < lf->depth; d++){
            int j = lf->rulepath_var[d] >= 0 ? lf->rulepath_var[d] : -(lf->rulepath_var[d]);
            int c = lf->rulepath_bx[d];
            if(lf->rulepath_var[d] >= 0){
                strcpy(sign, "<=");
                strcpy(is_or_not, "is");
            } else {
                strcpy(sign, ">");
                strcpy(is_or_not, "is not");
            }
            
            if(j == 0) break;
            if(model->var_types[j] == 'f'){
                Rprintf("%s %s %s", model->var_labels[j], is_or_not, fac_level_names[model->index_in_group[j]][c]);
            } else if(model->var_types[j] == 'n'){
                Rprintf("%s %s %0.4f", model->var_labels[j], sign, model->numeric_cuts[model->index_in_group[j]][c]);
            } else if(model->var_types[j] == 'i'){
                Rprintf("%s %s %d", model->var_labels[j], sign, model->integer_cuts[model->index_in_group[j]][c]);
            }
            if(d < lf->depth - 1){
                Rprintf(" and ");
            } else {
                Rprintf(" Then ");
            }
            
        }
        
        int max_index = 0;
        int max_count = 0;
        int total_count = 0;
        for(int i = 0; i < model->yc->nlevels; i++){
            total_count += lf->count[i];
            if(lf->count[i] > max_count){
                max_index = i;
                max_count = lf->count[i];
            }
        }
        
        if(model->yc->type == REGRESSION){
            Rprintf("%s ~= %0.4f with probability %0.4f.\n", model->var_labels[0], model->yc->yavg[max_index], (double)1.0*max_count/total_count);
        } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num != NULL){
            Rprintf("%s ~= %0.4f with probability %0.4f.\n", model->var_labels[0], model->yc->yvalues_num[max_index], (double)1.0*max_count/total_count);
        } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_int != NULL){
            if(model->yc->level_names == NULL){
                Rprintf("%s is %d with probability %0.4f.\n", model->var_labels[0], model->yc->yvalues_int[max_index], (double)1.0*max_count/total_count);
            } else {
                Rprintf("%s is %s with probability %0.4f.\n", model->var_labels[0], model->yc->level_names[max_index], (double)1.0*max_count/total_count);
            }
        }
        lf = lf->next;
    }
    

    // clean up
    for(int j = 1; j <= model->p; j++){
        if(model->var_types[j] == 'f'){
            free(fac_level_names[model->index_in_group[j]]);
        }
    }
    free(fac_level_names);
    
}
