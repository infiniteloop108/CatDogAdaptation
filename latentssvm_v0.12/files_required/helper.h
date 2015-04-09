#ifndef HELPER_H
#define HELPER_H


#ifdef __cplusplus
extern "C" {
#endif
 

#include "svm_struct_latent_api_types.h"

//LATENT_VAR infer_latent_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_helper(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
//void find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void classify_struct_example_helper(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, double *dec_val);
void read_struct_pattern_helper(const char *filename, PATTERN * x); 
SAMPLE read_struct_examples_helper(char *file, STRUCT_LEARN_PARM *sparm);


// the latent detect work
void read_struct_pattern_ld_helper(const char *filename, PATTERN *x);
LATENT_VAR infer_latent_variables_ld_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
//void find_most_violated_constraint_marginrescaling_ld_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
//void find_most_violated_constraint_marginrescaling_ld_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void find_most_violated_constraint_marginrescaling_ld_bin_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

// changed by subho
double loss_helper(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm);

#ifdef __cplusplus
}
#endif


#endif
