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


SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
	/*
	   Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
	   The type of pattern x and label y has to follow the definition in 
	   svm_struct_latent_api_types.h. Latent variables h can be either
	   initialized in this function or by calling init_latent_variables(). 
	   */
	SAMPLE sample;
	/* your code here */

	FILE *fin;
	fin = fopen(file,"r");
	if(fin != NULL)
	{
		printf("Input Data loading\n");
		int n;
		fscanf(fin,"%d",&n);
		sample.n = n;
		sample.examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * n);
		for(int i=0;i<n;++i)
		{
			char fileName[100];
			int class;
			fscanf(fin,"%s", fileName);
			fscanf(fin,"%d", &class);
			strcpy(sample.examples[i].file_name, fileName);
			sample.examples[i].y.class_id = class;
			//Now read the feature vector (x)
			FILE *data;
			printf("Loading %s\n", fileName);
			data = fopen(fileName, "r");
			if(data == NULL)
			{
				printf("Unable to open input file %s!\n",fileName);
				exit(1);
			}
			int nRect,dim;
			fscanf(data, "%d", &nRect);
			sample.examples[i].x.no_of_rects = nRect;
			fscanf(data, "%d", &dim);
			sample.examples[i].x.dimension = dim;
			sample.examples[i].x.data = (float *) malloc(sizeof(float) * nRect* dim);
			for(int j=0;j<nRect * dim;++j)
			{
				fscanf(data, "%f", &sample.examples[i].x.data[j]);
			}
			fclose(data);
		}
		fclose(fin);
		printf("Input data Loaded\n");
	}
	else
	{
		printf("Unable to open input file!\n");
		exit(1);
	}
	return(sample); 
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
	/*
	   Initialize parameters in STRUCTMODEL sm. Set the diminension 
	   of the feature space sm->sizePsi. Can also initialize your own
	   variables in sm here. 
	   */
	sm->sizePsi = sample.examples[1].x.dimension; /* replace with appropriate number */
	/* your code here*/
	// sm->w = (double *) malloc(sizeof(double) * (sm->sizePsi+1));
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Initialize latent variables in the first iteration of training.
	   Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
	   */
	/* your code here */
	//Initialize to 0?
	//printf("HERE init");
	//fflush(stdout);
	for(int i=0;i<sample->n;++i)
	{
		sample->examples[i].h.box=0;
	}
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Creates the feature vector \Psi(x,y,h) and return a pointer to 
	   sparse vector SVECTOR in SVM^light format. The dimension of the 
	   feature vector returned has to agree with the dimension in sm->sizePsi. 
	   */
	//printf("HERE psi");
	//fflush(stdout);
	SVECTOR *fvec=NULL;  
	/* your code here */
	int start = h.box * x.dimension;
	int sz = sm->sizePsi;
	int numNonZeroes = 0;
	for( int i=0;i<sz;++i)
	{
		if(fabs(x.data[start+i]) > 1.0e-6)
			numNonZeroes++;
	}
	WORD *words = (WORD *) malloc(sizeof(WORD)*(numNonZeroes+1));	//extra for 0.0
	assert(words != NULL);
	int j=0;
	for(int i=0;i<sz;++i)
	{
		if(fabs(x.data[start+i])> 1.0e-6)
		{
			words[j].wnum = i+1;
			words[j].weight = y.class_id * x.data[start+i];
			j++;
		}
	}
	words[j].wnum=0;
	words[j].weight = 0.0;
	fvec = create_svector(words, "", 1);
	free(words);
	return(fvec);
}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Makes prediction with input pattern x with weight vector in sm->w,
	   i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
	   Output pair (y,h) are stored at location pointed to by 
	   pointers *y and *h. 
	   */
	/* your code here */
	//printf("HERE class");
	//fflush(stdout);
	int numRects = x.no_of_rects;
	int dim = x.dimension;
	double valpos=0, valneg =0, mx = -1.0e10;
	for(int i=0;i<numRects;++i)
	{
		int start = i*dim;
		valpos=0;
		for(int j=0;j<dim;++j)
		{
			valpos+=(x.data[start+j]*sm->w[j+1]);	//w is 1-indexed (for some odd reason)
		}
		valneg=0;
		for(int j=0;j<dim;++j)
		{
			valneg-=(x.data[start+j]*sm->w[j+1]);
		}
		if(valpos > mx)
		{
			y->class_id=1;
			h->box=i;
			mx = valpos;
		}
		if(valneg > mx)
		{
			y->class_id=-1;
			h->box=i;
			mx = valneg;
		}
	}
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Finds the most violated constraint (loss-augmented inference), i.e.,
	   computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
	   The output (ybar,hbar) are stored at location pointed by 
	   pointers *ybar and *hbar. 
	   */

	/* your code here */
	//printf("HERE violated");
	//fflush(stdout);
	int numRects = x.no_of_rects;
	int dim = x.dimension;
	double valpos=0, valneg =0, mx = -1.0e10;
	for(int i=0;i<numRects;++i)
	{
		int start = i*dim;
		valpos=0;
		for(int j=0;j<dim;++j)
		{
			valpos+=(x.data[start+j]*sm->w[j+1]);	//w is 1-indexed (for some odd reason)
		}
		valneg=0;
		for(int j=0;j<dim;++j)
		{
			valneg-=(x.data[start+j]*sm->w[j+1]);
		}
		//0-1 Loss function
		if(y.class_id == 1)
			valneg += 1.0;
		else
			valpos += 1.0;
		if(valpos > mx)
		{
			ybar->class_id=1;
			hbar->box=i;
			mx = valpos;
		}
		if(valneg > mx)
		{
			ybar->class_id=-1;
			hbar->box=i;
			mx = valneg;
		}
	}
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Complete the latent variable h for labeled examples, i.e.,
	   computing argmax_{h} <w,psi(x,y,h)>. 
	   */
	LATENT_VAR h;
	/* your code here */
	//printf("HERE infert");
	//fflush(stdout);
	int numRects = x.no_of_rects;
	int dim = x.dimension;
	double val=0, mx = -1.0e10;
	for(int i=0;i<numRects;++i)
	{
		int start = i*dim;
		val=0;
		for(int j=0;j<dim;++j)
		{
			val+=(y.class_id) * (x.data[start+j]*sm->w[j+1]);	//w is 1-indexed (for some odd reason)
		}
		if(val > mx)
		{
			h.box=i;
			mx = val;
		}
	}
	return(h); 
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
	/*
	   Computes the loss of prediction (ybar,hbar) against the
	   correct label y. 
	   */
	double ans;

	/* your code here */
	//0-1 loss function
	if(y.class_id == ybar.class_id)
		ans=0.0;
	else
		ans=1.0;

	return(ans);
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Writes the learned weight vector sm->w to file after training. 
	   */
	//printf("HERE write");
	//fflush(stdout);
	FILE *fout;
	fout = fopen(file, "w");
	fprintf(fout, "%ld\n", sm->sizePsi);
	for(int i=0;i<sm->sizePsi;++i)
		fprintf(fout, "%0.6lf\n", sm->w[i+1]);
	fclose(fout);
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
	/*
	   Reads in the learned model parameters from file into STRUCTMODEL sm.
	   The input file format has to agree with the format in write_struct_model().
	   */
	STRUCTMODEL sm;
	//printf("HERE read");
	//fflush(stdout); 
	/* your code here */
	FILE *fin;
	fin = fopen(file, "r");
	int sz;
	fscanf(fin, "%d", &sz);
	sm.sizePsi = sz;
	sm.w = (double *) malloc(sizeof(double) * (sz+1));
	for(int i=0;i<sz;++i)
		fscanf(fin, "%lf", &sm.w[i+1]);
	fclose(fin);
	return(sm);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
	/*
	   Free any memory malloc'ed in STRUCTMODEL sm after training. 
	   */

	/* your code here */

	free(sm.w);
}

void free_pattern(PATTERN x) {
	/*
	   Free any memory malloc'ed when creating pattern x. 
	   */

	/* your code here */
	free(x.data);
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

	for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
		switch ((sparm->custom_argv[i])[2]) {
			/* your code here */
			default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
		}
	}
}

