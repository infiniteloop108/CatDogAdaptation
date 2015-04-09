
#include "helper.h"

#include <algorithm>
#include <vector>
#include <string>
#include <limits>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <memory>

extern "C" {
#include "./svm_light/svm_common.h"
}
using namespace std;
										
SVECTOR *psi_helper(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/

long long int from=sparm->num_features*h.displacement;
long long int to=(sparm->num_features*(h.displacement+1))-1;   // where in x array is h located, -1 as last one is inclusive
 // eg : if h=0 , then its 1st feature vector ie 0 to 4095
 
  long long int num_nonzeros=0;
  for (long long int b = from; b <= to; b++)  // check how many non_zeroes are there
  {
    if ( fabs(x.data[b]) > 0.0) {
      num_nonzeros++;
    }
  }

  WORD *words = (WORD*) my_malloc(sizeof (WORD)*(num_nonzeros + 1)); // +1 as in the end a 0,0 will be set to denote end
  assert(words != NULL);

  long long int j = 0;  // for word array
  long long int k = 0;  // for feature vector number
  for (long long int i = from; i <= to; i++)  // i is used to access the data array
  {
    if (fabs(x.data[i]) > 0.0) 
	{
      words[j].wnum = (k + 1);
      words[j].weight = y.class_id*x.data[i];
     // words[j].weight = Hist[i];
      //printf("i = %d weight = %f\n",words[j].wnum, words[j].weight);
      j++;
    }
	k++;
  }
  words[j].wnum = 0;
  words[j].weight = 0.0;
  SVECTOR * fvec = create_svector(words, "", 1);
  
  assert(j > 0);

  free(words);

  return (fvec);
}

LATENT_VAR infer_latent_variables_ld_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

// <subho new> 

// cout<<"\nInfer latent started --> ";
 LATENT_VAR h;
/*
 if(y.class_id==-1)  // for -ve image use whole image
  {
	h.displacement=0;
	return h;  
  }
  */
 int no_of_rects=x.no_of_rects;
 int dimension=sparm->num_features;

 double *data=new double[no_of_rects];
 
 double store=0;
 for(int i=0;i<no_of_rects;i++)
	{
	   store=0;
	   for(int j=(dimension*i),k=0; j<=(dimension*(i+1)-1);j++,k++) // j is to access data and k for w
         {
		    store=store+(((y.class_id)*(x.data[j]))*(sm->w[k+1])); // <w,psi[x,h,h[i]> // not calling psi() as that would return svector, we can just compute it here
         }                              // w is from 1 to 4096
       data[i]=store;  // ie w.psi  
	} 
   /*
  // now do clustering in w.psi space 
  
 int numClusters=(0.20*no_of_rects); // taking it as 10% of training data
 int clusterDimension=1; // as it is <w.psi> space
    
  double **clusters=new double*[numClusters];  // 0 to numClusters-1, and each storing a cluster center
  
for(int i = 0; i < numClusters; ++i)
    clusters[i] = new double[clusterDimension];
    
  int *reverseIndex=new int[no_of_rects];
  double *scores=new double[no_of_rects];
  
  //now choose numCluster number of values randomly between dimension*i and dimension*(i+1)-1
  int value;
  
  // initialize with a random h, but all must be unique
  
  for(int i=0;i<no_of_rects;i++)
    reverseIndex[i]=-1;
  
  srand(time(NULL));  // randomize start
  
  for(int i=0;i<numClusters;i++)
  {
    value=(rand()%no_of_rects);  // ie %(4096 - 0) + 0, ie numbers generated are from 0 to 4095
    
    if(reverseIndex[value]==-1)   // unique generation
     { 
		reverseIndex[value]=i;
		
		for(int j=(clusterDimension*value),k=0; j<=(clusterDimension*(value+1)-1);j++,k++) // j is to access data and k for w
         {
			 clusters[i][k]=data[j];
         }
     }
    else
     {
		while(reverseIndex[value]!=-1)
		   value=(rand()%no_of_rects);
		   
        reverseIndex[value]=i;
        
        for(int j=(clusterDimension*value),k=0; j<=(clusterDimension*(value+1)-1);j++,k++) // j is to access data and k for w
         {
			 clusters[i][k]=data[j];
         }
     }
  }
 	// cluster randomly initialized

  
  // run k-means
  
  long double dist=0,temp=0;
  int closest_Cluster=-1;
  int m=0;
  
 while(m<=1000)  // how many times to run
 {
	 m++;
   for(int i=0;i<no_of_rects;i++)  // for all h get closest cluster
   {
	 dist=99999999;
	 closest_Cluster=-1; 
	 
	 for(int p=0;p<numClusters;p++)  // find out which cluster is closer 
	  {
		temp=0;  
	    for(int j=(clusterDimension*i),k=0; j<=(clusterDimension*(i+1)-1);j++,k++) // j is to access data and k for w
         {
			 temp=temp+((data[j]-clusters[p][k])*(data[j]-clusters[p][k]));
         }
         
         if(temp<dist)
         {
			 dist=temp;
			 closest_Cluster=p;
		 }
      }
      reverseIndex[i]=closest_Cluster;  
   }
   
   // clear cluster values as new clusters will be calculated 
   
    for(int p=0;p<numClusters;p++) 
	  {
	    for(int k=0; k<clusterDimension;k++) 
	    {         
          clusters[p][k]=0;   
        }
      }
   
   // now find the new clusters
   int count=0;
   
   for(int i=0;i<numClusters;i++)  // for each cluster , can be done in O(n) too, convert if too much time taken
    {
		count=0;
		for(int j=0;j<no_of_rects;j++)  // check how many belong to ith cluster
		{
			if(reverseIndex[j]==i)
			{
	           for(int k=0,p=(clusterDimension*i);k<clusterDimension;k++,p++) 
	            {         
                 clusters[i][k]=clusters[i][k]+data[p];   
                }
              count++;
			}
	    }
	    
	     for(int k=0;k<clusterDimension;k++) 
	        {         
              clusters[i][k]=clusters[i][k]/count;   
            }
    }
    
 } // k means over 
 
   // count how many belong to each cluster
   for(int i=0;i<no_of_rects;i++)
   {
	   scores[i]=0;
	   for(int j=0;j<no_of_rects;j++)
	   {
		   if(reverseIndex[i]==reverseIndex[j])
		    scores[i]++;
	   }
   }
   
  //cout<<"\n\n";
  for(int i=0;i<no_of_rects;i++)
   {
	   //cout<<scores[i]<<"\n";
	   scores[i]=scores[i]/(double)no_of_rects;
	   
	   //cout<<"Score of h["<<i<<"] : "<<scores[i]<<"/"<<no_of_rects<<"\n";
   }
   
   // now scores has for each h, the score
 */
// </subho new>
 
   long double mx=-9999999;
   long double val=0;
    for(int i=0;i<no_of_rects;i++)
	{
	   val=data[i];//+scores[i]; // as we precomputer the w.psi, adding scores[i]. Ie taking examples which have many neighbours around it
	   
      if(val>mx)
       {    
	    mx=val; 
		h.displacement=i; 
	   }	  
	}
	
	//cout<<" <-- Infer latent finished !\n";
	
	delete[] data;
	return h;
}
void find_most_violated_constraint_marginrescaling_ld_bin_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
      
   int no_of_rects=x.no_of_rects;
   int dimension=sparm->num_features;
   
   double mx=-9999999;
   double val1=0;
   double val2=0; 
   
    for(int i=0;i<no_of_rects;i++)  // for each hbar
	{
	   val1=0;
	   for(int j=(dimension*i),k=0; j<=(dimension*(i+1)-1);j++,k++) // j is to access data and k for w
         {
		    val1=val1+(((1)*(x.data[j]))*(sm->w[k+1])); // <w,psi[x,h,h[i]> // not calling psi() as that would return svector, we can just compute it here. ybar=1 here
         }                                // k+1 as w is indexed from 1
		 
	   if(y.class_id==-1) // to add the loss , ie if class mismatches
   	   {
		val1=val1+1;  // add loss
	   }
		
	   val2=0;	 
	   for(int j=(dimension*i),k=0; j<=(dimension*(i+1)-1);j++,k++) // j is to access data and k for w
         {
		    val2=val2+(((-1)*(x.data[j]))*(sm->w[k+1])); // <w,psi[x,h,h[i]> // not calling psi() as that would return svector, we can just compute it here, ybar=-1 here
         }                                // k+1 as w is indexed from 1
		
	   if(y.class_id==1) // to add the loss , ie if class mismatches
   	   {
		val2=val2+1;  // add loss
	   }
	
	//    val1=val1+scores[i];  // added scores of clustering , both are same as both are done one at a time, one is in *1 and other *-1, but both in +ve space
	 //   val2=val2+scores[i];  // added scores of clustering
	   
		if(val1>=val2 && val1>=mx)
        {
		  mx=val1;
		  hbar->displacement=i;
		  ybar->class_id=1;
        }
		else if(val2>=val1 && val2>=mx)
        {
		  mx=val2;
		  hbar->displacement=i;
		  ybar->class_id=-1;
        }		    	  
	}
}

double loss_helper(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) 
{
// NOte !! in help.cc, we have assumed 0-1 loss in find_most_violated_constraint_marginrescaling_ld_bin_helper(), change there too , if changed here
	if (y.class_id == ybar.class_id)
		return (0.0);
	else
		return (1.0);
		//return (100.0);
		//return (100.0);
}

void classify_struct_example_helper(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, double *dec_val)
{
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
   int no_of_rects=x.no_of_rects;
   int dimension=sparm->num_features;
   double mx=-99999999;
   double val1=0;
   double val2=0;
   
    for(int i=0;i<no_of_rects;i++)
	{
	   val1=0;
	   for(int j=(dimension*i),k=0; j<=(dimension*(i+1)-1);j++,k++) // j is to access data and k for w
         {
		    val1=val1+(((1)*(x.data[j]))*(sm->w[k+1])); // <w,psi[x,h,h[i]> // not calling psi() as that would return svector, we can just compute it here. ybar=1 here
         }
		
	   val2=0;	 
	   for(int j=(dimension*i),k=0; j<=(dimension*(i+1)-1);j++,k++) // j is to access data and k for w
         {
		    val2=val2+(((-1)*(x.data[j]))*(sm->w[k+1])); // <w,psi[x,h,h[i]> // not calling psi() as that would return svector, we can just compute it here, ybar=-1 here
         }
	   
		if(val1>=val2 && val1>=mx)
        {
		  mx=val1;
		  h->displacement=i;
		  y->class_id=1;
		  *dec_val=mx;
        }
		else if(val2>=val1 && val2>=mx)
        {
		  mx=val2;
		  h->displacement=i;
		  y->class_id=-1;
		  *dec_val=mx;
        }		 
	}
}

SAMPLE read_struct_examples_helper(char *file, STRUCT_LEARN_PARM *sparm) 
{
	/*
	 *   Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
	 *     The type of pattern x and label y has to follow the definition in 
	 *       svm_struct_latent_api_types.h. Latent variables h can be either
	 *         initialized in this function or by calling init_latent_variables(). 
	 *         */

   string line;
   string temp;
   int number_of_examples;
   int dimension;
   string file_name="";
   int label;
   int number_of_latent_variables;
   SAMPLE sample;

    ifstream myfile(file);

  if (myfile.is_open())
  {
     getline(myfile,line); // read number of examples
     number_of_examples=stoi(line,NULL);
     getline(myfile,line); // read dimension of each feature
     dimension=stoi(line,NULL);
	 
	// cout<<number_of_examples<<" "<<dimension<<"\n";

     sample.n = number_of_examples;
	 sample.examples = (EXAMPLE*) malloc(sizeof (EXAMPLE) * number_of_examples);
     sample.dimension=dimension;
     sparm->num_features=dimension; // will be needed in psi()
	 
    long long unsigned int i;
	long long int j=0; // index to examples array
    while (getline(myfile,line))
    {
      i=0;
	  file_name="";
      while(line[i]!=' ') // read the file name
	    {
		  file_name=file_name+line[i];
		  i++;
	    }
	  i++; // skip the space

	  temp="";
	  while(line[i]!=' ') // read number of h for this x
	    {
		  temp=temp+line[i];
		  i++;
	    }
	  i++; // skip the space

	  number_of_latent_variables=stoi(temp,NULL);
	  
	  temp="";
	  while(line[i]!=' ') // read label for this x
	    {
		  temp=temp+line[i];
		  i++;
	    }
	  i++; // skip the space

	  label=stoi(temp,NULL);

	  sample.examples[j].y.class_id=label;  
	 // sample.examples[j].h.displacement=0; // ie initializing the latent variable with the 0th image(full image)
      sample.examples[j].x.no_of_rects=number_of_latent_variables;
      sample.examples[j].x.outlier_score=0; // as of now not outliers
      
      strcpy(sample.examples[j].file_name, file_name.c_str());
      //sample.examples[j].file_name[sizeof(sample.examples[j].file_name) - 1] = 0; 

      //cout<<sample.examples[j].file_name<<" : "<<sample.examples[j].x.no_of_rects<<" : "<<sample.examples[j].y.class_id<<" \n";
      
	  sample.examples[j].x.data=new float[dimension*(long long int)number_of_latent_variables];
	 
	  //cout<<number_of_latent_variables<<" "<<label<<"\n";
	 
	  long long int index=0;

	  while(i<line.length())  // push the concatenated features into the features array
	  {
		  temp="";
	    while(line[i]!=' ' && i<line.length()) // read label for this x
	     {
		  temp=temp+line[i];
		  i++;
	     }
	   i++; // skip the space

       //cout<<index<<" >"<<temp<<"<\n";
	   sample.examples[j].x.data[index++]=stof(temp,NULL);
	  }
	  
	 // cout<<sample.examples[j].file_name<<" "<<sample.examples[j].x.no_of_rects<<" "<<sample.examples[j].y.class_id<<" ";
	 /* for(int k=0;k<(dimension*number_of_latent_variables);k++)
		  cout<<sample.examples[j].x.data[k]<<" ";
		  */
	 // cout<<"\n";
	  j++;
	  assert(index==(dimension*number_of_latent_variables));  // check if number_of_latent_variables number of features are present or not
    } 
    myfile.close();
  
     printf("\ndone reading parameters\n");
     return sample;
  }
  else { cout << "Unable to open file"; exit(1);  }
}
