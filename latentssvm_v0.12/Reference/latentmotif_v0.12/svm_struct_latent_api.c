/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "svm_struct_latent_api_types.h"
#include "./SFMT-src-1.3.3/SFMT.h"

#define MAX_INPUT_LINE_LENGTH 10000

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialized in this function or by calling init_latent_variables(). 
*/
  SAMPLE sample;
  int num_examples,i,label,length_seq,example_type;
  FILE *fp;
  char line[MAX_INPUT_LINE_LENGTH]; 
  char *pchar, *last_pchar;

  fp = fopen(file,"r");
  if (fp==NULL) {
    printf("Cannot open input file %s!\n", file);
	exit(1);
  }
  fgets(line, MAX_INPUT_LINE_LENGTH, fp);
  num_examples = atoi(line);
  sample.n = num_examples;
  sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_examples);
  
  for (i=0;(!feof(fp))&&(i<num_examples);i++) {
    fgets(line, MAX_INPUT_LINE_LENGTH, fp);
    pchar = line;
    while ((*pchar)!=':') pchar++;
    *pchar = '\0';
    strcpy(sample.examples[i].x.seq_name, line);
    pchar++;

    /* label: {+1, -1} */
    last_pchar = pchar;
    while ((*pchar)!=':') pchar++;
    *pchar = '\0';
    label = atoi(last_pchar);
    pchar++;

    /* example_type: {0: background, 1: experimental} */
    last_pchar = pchar;
    while ((*pchar)!=':') pchar++;
    *pchar = '\0';
    example_type = atoi(last_pchar);
    pchar++;

    length_seq = strlen(pchar);
    if (pchar[length_seq-1]=='\n') {
      pchar[length_seq-1]='\0';
      length_seq--;
    }

    sample.examples[i].y.label = label;
    sample.examples[i].x.length = length_seq;
    sample.examples[i].x.sequence = (char*)malloc(sizeof(char)*(length_seq+1));
    strcpy(sample.examples[i].x.sequence, pchar);
    sample.examples[i].x.example_id = i;
    sample.examples[i].x.example_type = example_type;
    if (example_type==0) {
      sample.examples[i].x.example_cost = 1.0;
    } else {
      sample.examples[i].x.example_cost = sparm->false_negative_cost;
    }
  }
  assert(i==num_examples);
  fclose(fp);  

  return(sample); 
}

inline int base2int(char base) {
  int ans;
  switch (base) {
    case 'A': ans=0; break;
    case 'a': ans=0; break;
    case 'C': ans=1; break;
    case 'c': ans=1; break;
    case 'G': ans=2; break;
    case 'g': ans=2; break;
    case 'T': ans=3; break;
    case 't': ans=3; break;
    default: printf("ERROR: Unrecognized nucleotide '%c'\n!", base); exit(1);
  }

  return(ans);
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  int i,j,hash,msb_offset;
  
  sm->n = sample.n;
  
  msb_offset = 1;
  for (j=1;j<sparm->bg_markov_order;j++) {
    msb_offset*=4;
  }
  
  sm->sizePsi = 4*msb_offset + 4*sparm->motif_length;
  
  /* precompute indices to speed up training */
  sm->pattern_hash = (int**)malloc(sizeof(int*)*sample.n);
  for (i=0;i<sample.n;i++) {
    sm->pattern_hash[i] = (int*)malloc(sizeof(int)*sample.examples[i].x.length);
  }
  for (i=0;i<sample.n;i++) {
    hash = 0;
    for (j=0;j<sparm->bg_markov_order;j++) {
      hash = 4*hash + base2int(sample.examples[i].x.sequence[j]);
    }
    sm->pattern_hash[i][0] = hash;
    for (j=1;j<sample.examples[i].x.length-sparm->bg_markov_order;j++) {
      hash = 4*(hash-msb_offset*base2int(sample.examples[i].x.sequence[j-1]))+base2int(sample.examples[i].x.sequence[j-1+sparm->bg_markov_order]);
      sm->pattern_hash[i][j] = hash;
    }
  }
  
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
  int i; 
  /* initialize the RNG */
  init_gen_rand(lparm->biased_hyperplane);

  for (i=0;i<sample->n;i++) {
    if (sample->examples[i].y.label==-1) {
      sample->examples[i].h.position = -1;
    } else {
      sample->examples[i].h.position = (long) floor(genrand_res53()*(sample->examples[i].x.length-sparm->motif_length-sparm->bg_markov_order));
    }
  }

}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
  SVECTOR *fvec=NULL;
  int j,l,k;
  int *pattern_hash;
  int *count_vector;
  WORD *words;
  
  pattern_hash = sm->pattern_hash[x.example_id];
  
  count_vector = (int*)malloc(sizeof(int)*(sm->sizePsi+1)); // note that the features range from 1 to sm->sizePsi
  for (j=0;j<sm->sizePsi+1;j++) {
    count_vector[j]=0;
  }
  for (j=0;j<x.length-sparm->bg_markov_order;j++) {
    count_vector[1+pattern_hash[j]]++;
  }
  
  if (y.label==1) {
    /* subtract off background and add motif */
    for (j=h.position;j<h.position+sparm->motif_length;j++) {
      /* decrement counts in the feature vector */
      count_vector[1+pattern_hash[j]]--;
    }	
	
    for (j=h.position;j<h.position+sparm->motif_length;j++) {
      count_vector[sm->sizePsi-(4*(j-h.position)+base2int(x.sequence[j]))]++;
    }
  }
  /* count number of nonzeros */
  l=0;
  for (j=1;j<sm->sizePsi+1;j++) {
    if (count_vector[j]>0) {
      l++;
    }
  }

  words = (WORD*)my_malloc(sizeof(WORD)*(l+1));
  assert(words!=NULL);
  k=0;
  for (j=1;j<sm->sizePsi+1;j++) {
    // correctness check
    assert(count_vector[j]>=0);
    if (count_vector[j]>0) {
      words[k].wnum = j;
      words[k].weight = (double) count_vector[j];
      k++;
    }
  }
  words[k].wnum=0;
  words[k].weight=0.0;
  fvec = create_svector(words,"",1);
  
  free(words);
  free(count_vector);

  return(fvec);
}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  
  double max_score, score;
  int *pattern_hash, max_pos,j,k;
  
  //y = (LABEL*)malloc(sizeof(LABEL));
  //h = (LATENT_VAR*)malloc(sizeof(LATENT_VAR));

  pattern_hash = sm->pattern_hash[x.example_id];
  
  max_score = 0.0;
  max_pos = -1;
  for (k=0;k<x.length-sparm->motif_length-sparm->bg_markov_order;k++) {
    score = 0.0;
    for (j=k;j<k+sparm->motif_length;j++) {
      score += sm->w[sm->sizePsi-(4*(j-k)+base2int(x.sequence[j]))];
      score -= sm->w[1+pattern_hash[j]];
    }
    if (score>max_score) {
      max_score = score;
      max_pos = k;
    }
  }
  
  h->position = max_pos; 
  if (max_pos>-1) {
    y->label = 1;
  } else {
    y->label = -1;
  }

}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  double max_score, score;
  int *pattern_hash, max_pos,j,h;

  pattern_hash = sm->pattern_hash[x.example_id];
  
  max_score = -1E10;
  max_pos = -1;
  for (h=0;h<x.length-sparm->motif_length-sparm->bg_markov_order;h++) {
    score = 0.0;
    for (j=h;j<h+sparm->motif_length;j++) {
      score += sm->w[sm->sizePsi-(4*(j-h)+base2int(x.sequence[j]))];
      score -= sm->w[1+pattern_hash[j]];
    }
    if (score>max_score) {
      max_score = score;
      max_pos = h;
    }
  }

  /* zero-one loss */
  if (y.label==1) {
    if (max_score>1.0) { 
      ybar->label = 1;
      hbar->position = max_pos;
    } else {
      ybar->label = -1;
      hbar->position = -1;
    }
  } else {
    if (1.0+max_score>0) {
      ybar->label = 1;
      hbar->position = max_pos;
    } else {
      ybar->label = -1;
      hbar->position = -1;
    }

  }
  
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;

  double max_score, score;
  int *pattern_hash, max_pos,j,k;
  
  pattern_hash = sm->pattern_hash[x.example_id];
  
  if (y.label==-1) { // no motif
    h.position = -1;
  } else {
    /* find out highest scoring position */
    max_score = -1E10;
    max_pos = -1;
    for (k=0;k<x.length-sparm->motif_length-sparm->bg_markov_order;k++) {
      score = 0.0;
      for (j=k;j<k+sparm->motif_length;j++) {
	score += sm->w[sm->sizePsi-(4*(j-k)+base2int(x.sequence[j]))];
	score -= sm->w[1+pattern_hash[j]];
      }
      if (score>max_score) {
	max_score = score;
	max_pos = k;
      }
    }
    h.position = max_pos;
  }

  return(h); 
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
  if (y.label==ybar.label) {
    return(0);
  } else {
    return(1);
  }
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
  FILE *modelfl;
  int i;
  
  modelfl = fopen(file,"w");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for output!", file);
	exit(1);
  }
  
  /* write model information */
  fprintf(modelfl, "# motif width: %d\n", sparm->motif_length);
  fprintf(modelfl, "# order of background Markov model: %d\n", sparm->bg_markov_order);

  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
  }
  fclose(modelfl);
 
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/
  STRUCTMODEL sm;

  FILE *modelfl;
  int sizePsi,i, fnum;
  double fweight;
  char line[1000];
  
  modelfl = fopen(file,"r");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for input!", file);
	exit(1);
  }
  
  sizePsi = 1;
  for (i=0;i<sparm->bg_markov_order;i++) {
    sizePsi*=4;
  }
  sizePsi += 4*sparm->motif_length;
  
  sm.sizePsi = sizePsi;
  sm.w = (double*)malloc((sizePsi+1)*sizeof(double));
  for (i=0;i<sizePsi+1;i++) {
    sm.w[i] = 0.0;
  }
  /* skip first two lines of comments */
  fgets(line,1000,modelfl);
  fgets(line,1000,modelfl);
  
  while (!feof(modelfl)) {
    fscanf(modelfl, "%d:%lf", &fnum, &fweight);
	sm.w[fnum] = fweight;
  }

  fclose(modelfl);

  return(sm);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
  int i;
  
  free(sm.w);
  for (i=0;i<sm.n;i++) {
    free(sm.pattern_hash[i]);
  }
  free(sm.pattern_hash);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/
  free(x.sequence);

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

  /* your code here */

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

  /* your code here */

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  sparm->bg_markov_order = 2;
  sparm->motif_length = 10;
  sparm->false_negative_cost = 1.0;
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      case 'o': i++; sparm->bg_markov_order = atoi(sparm->custom_argv[i]); break;
      case 'm': i++; sparm->motif_length = atoi(sparm->custom_argv[i]); break;
      case 'r': i++; sparm->false_negative_cost = atof(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

