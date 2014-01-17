#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h> 

#include <cuda.h>

#include "constants.h"
#include "potential.h"
#include "cudacall.h"
#include "parse.h"
#include "device.h"
#include "gibbs.h"

#ifndef DEBUG
#define DEBUG 0
#endif

#define DPRINT(...)                                             \
  //do { if (DEBUG) fprintf(stderr, __VA_ARGS__); } while (0)


int freezeDevicePotential(Potential *pd, int frozen){
  Potential p;  
  CUDA_CALL(cudaMemcpy ( &p,  pd, sizeof(p), cudaMemcpyDeviceToHost));
  p.isFrozen = frozen;
  CUDA_CALL(cudaMemcpy ( pd, &p, sizeof(p), cudaMemcpyHostToDevice));

  return 0;
}

int printDevicePotential (Potential*pd) {
  Potential p;  
  CUDA_CALL(cudaMemcpy ( &p,  pd, sizeof(p), cudaMemcpyDeviceToHost));

  printf("%17s %6d\n",  "numStates", p.numStates);

  if (p.numConditionals >=0 && p.numConditionals < 1000){

    float *conditionals = (float*) malloc(p.numConditionals * sizeof(float));
    CUDA_CALL(cudaMemcpy ( conditionals,  p.conditionals, p.numConditionals * sizeof(p.conditionals[0]), cudaMemcpyDeviceToHost));

    for (int i=0; i< p.numConditionals; i++){
      printf("%11s[%3d] %6.3f\n",  "conditionals", i,  conditionals[i]);
    }
    free(conditionals);
  }
  printf("%17s %6d\n",  "numConditionals", p.numConditionals);
  printf("%17s %6d\n",  "numParents", p.numParents);
  printf("%17s %6d\n",  "numChildren", p.numChildren);
  

  if (p.numParents >=0 && p.numParents <= MAX_PARENTS){
    for (int i=0; i<p.numParents; i++){
      int offset = p.parents[i] - pd;
      printf("%12s[%3d] %6d\n",  "parent", i, offset);
    }
  }
  if (p.numChildren >=0 && p.numChildren <= MAX_CHILDREN){
    for (int i=0; i<p.numChildren; i++){
      int offset = p.children[i] - pd;
      printf("%12s[%3d] %6d\n",  "child", i, offset);
    }
  }
  if (p.numChildren >=0 && p.numChildren <= MAX_CHILDREN){
    for (int i=0; i<p.numChildren; i++){
      printf("%12s[%3d] %6d\n",  "indexInChild", i, p.indexInChild[i]);
    }
  }
  if (p.numParents >=0 && p.numParents <= MAX_PARENTS){
    for (int i=0; i<=p.numParents; i++){
      printf("%12s[%3d] %6d\n",  "dimensions", i, p.dimensions[i]);
    }
  }
  printf("%17s %6d\n",  "isFrozen", p.isFrozen);
  return 0;
}


const int POTENTIALINFO_BLOCK_SIZE = 1000;

typedef struct PotentialInfo {
  char* name;
  char** parentNames;
  int numStates;
  int  numParents;
  float *table;
  int lengthTable;
  Potential *devPtr;
}PotentialInfo;

struct PotentialInfo **parsedPotentials =
  (PotentialInfo**)malloc(POTENTIALINFO_BLOCK_SIZE * sizeof(PotentialInfo*));
int numParsedPotentials = 0;
  
///
/// Allocate a copy of a string with malloc.
///

char *allocName (char * src){
  int n = strlen (src);
  char * p = (char*) malloc(n);
  strcpy (p, src);
  return p;
}

///
/// The parser calls this function with the info for each parsed potential.
/// We just save the info for later.
///

void parseCallback(char* name, int numStates, char**parents, int numParents, float* table, int lengthTable){
  //printf("callback %s \n numStates %d \n parents[", name, numStates);
  for (int i=0; i< numParents; i++){
    //printf (" %s", parents[i] );
  }
  //printf("]\n table [");
  for (int i=0; i< lengthTable; i++){
    //printf(" %f", table[i]);
  }
  //printf("]\n");

  // save this potential info for later
  PotentialInfo * pi = (PotentialInfo*)calloc(1, sizeof(PotentialInfo));
  if (((numParsedPotentials+1) % POTENTIALINFO_BLOCK_SIZE) == 0){
    parsedPotentials = 
      (PotentialInfo**) realloc(parsedPotentials, 
                               (numParsedPotentials +POTENTIALINFO_BLOCK_SIZE) 
                               * sizeof(PotentialInfo*));
  }
  parsedPotentials[numParsedPotentials++] = pi;
  pi->name = allocName(name);
  pi->parentNames = (char**) calloc(numParents, sizeof(char*));
  for (int i=0; i< numParents; i++){
    pi->parentNames[i] = allocName(parents[i]);
  }
  pi->numStates = numStates;
  pi->numParents = numParents;
  pi->table = (float*)malloc(lengthTable*sizeof(float));
  memcpy (pi->table, table, lengthTable*sizeof(float));
  pi->lengthTable = lengthTable;
  pi->devPtr = NULL;
}

int initOnePotential(Potential * devPotential, PotentialInfo ** ppi, int npi, int ip){
  //printf("initOnePotential\n");
  PotentialInfo* pi = ppi[ip];
  // find its parents
  
  Potential ** devParents;
  CUDA_CALL(cudaMalloc ( (void**) &devParents, MAX_PARENTS * sizeof( Potential * ) ));
  //printf("  initOnePotential -- malloced parents\n");
  int err = 0;
  for ( int iParent=0; iParent < pi->numParents; iParent++){
    char * parentName = pi->parentNames[iParent];
    //printf("  initOnePotential -- parent %s\n", parentName);

    int found = 0;
    for (int i= ip-1; i>=0 ; --i){
      PotentialInfo* candidate = ppi[i];
      if (strncmp (candidate->name, parentName, 64) == 0){
        //printf("  initOnePotential -- found parent %s = %p\n", parentName, candidate->devPtr);
          CUDA_CALL(cudaMemcpy (devParents+iParent, &candidate->devPtr ,  sizeof( Potential * ), cudaMemcpyHostToDevice));
          found = 1;
          //printf("  initOnePotential -- copied parent %s\n", parentName);
          break;
      }
    }
    if(found == 0){
      fprintf(stderr, "for %s, cannot locate parent %s\n", pi->name, parentName);
      err = 1;
      break;
    }
  }
  float *devConditionals = 0;
  if ( err == 0){
    // copy the conditional table over
     //printf("  initOnePotential -- allocing table length %d\n",  pi->lengthTable);
  
    CUDA_CALL(cudaMalloc ( (void**) &devConditionals, pi->lengthTable * sizeof( float ) ));
    CUDA_CALL(cudaMemcpy (devConditionals, pi->table, pi->lengthTable* sizeof(float), 
                          cudaMemcpyHostToDevice));    
  }
  if (err == 0){
    //printf("calling initPotential %p\n numStates=%d\n table=[",devPotential,   pi->numStates);
    for (int i=0; i< pi->lengthTable; i++){
      //printf (" %f", pi->table[i]);
    }
    //printf(" ]\n");

    initPotential<<<1, 1>>> (devPotential, pi->numStates, devConditionals, 
                 devParents, pi->numParents );
  }

  // Remember the devPotential for this parsed potential
  pi->devPtr = devPotential;

  // It is OK to free the devParents as the device code made a copy
  CUDA_CALL(cudaFree ( devParents)); 
  return err;
}

void parseNetwork(const char * fileName){
  FILE * fp1 = fopen(fileName, "r");
  parse(fp1, parseCallback);
  fclose(fp1);
}
  
void parseStates(const char* fileName, int *states, int *fixed){
  
  FILE * fp2 = fopen(fileName, "r");
  char line[1024];
  for (int iState = 0; iState < numParsedPotentials && fgets(line, sizeof(line)-1, fp2); iState++) {
    char flag[2] ; 
    flag[0]='\0';
    sscanf (line, "%d %1s", states + iState, &flag);
    if (flag[0]){
      fixed[iState] = 1;
    } else {
      fixed[iState] = 0;
    }
  }
  fclose(fp2);

}

int verboseFlag = 0;
char * networkFileName  = NULL;
char * stateFileName  = NULL;

int options (int argc, char **argv) {

  int c;
  while (1)
    {
      static struct option long_options[] =
        {
          /* These options set a flag. */
          {"verbose", no_argument,       &verboseFlag, 1},
          {"brief",   no_argument,       &verboseFlag, 0},
          /* These options don't set a flag.
             We distinguish them by their indices. */
          //{"add",     no_argument,       0, 'a'},
          {"netfile",    required_argument, 0, 'f'},
          {"statefile",    required_argument, 0, 's'},
          {0, 0, 0, 0}
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;
     
      c = getopt_long (argc, argv, ":s:f:",
                       long_options, &option_index);
     
      /* Detect the end of the options. */
      if (c == -1)
        break;
     
      switch (c)
        {
        case 0:
          /* If this option set a flag, do nothing else now. */
          if (long_options[option_index].flag != 0)
            break;
          //printf ("option %s", long_options[option_index].name);
          //if (optarg)
          //  printf (" with arg %s", optarg);
          //printf ("\n");
          break;
     
        case 'f':
          //printf ("option -f with value `%s'\n", optarg);
          networkFileName = optarg;
          break;
     
        case 's':
          //printf ("option -s with value `%s'\n", optarg);
          stateFileName = optarg;
          break;
     
        case '?':
          /* getopt_long already printed an error message. */
          // no it didn't
          fprintf (stderr, "Usage: %s -f netfile -s statefile [--verbose]\n", argv[0]);
          exit(1);
          break;
     
        default:
          abort ();
        }
    }
     
     
  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
    {
      fprintf(stderr, "Error: unsupported file arguments on command line\n");
      exit(1);
      //printf ("non-option ARGV-elements: ");
      //while (optind < argc)
      //  printf ("%s ", argv[optind++]);
      // putchar ('\n');
    }
  return 0;
}


void printTime(){
  char outstr[200];
  time_t t;
  struct tm *tmp;

  t = time(NULL);
  tmp = localtime(&t);
  if (tmp == NULL) {
    perror("localtime");
    exit(EXIT_FAILURE);
  }

  if (strftime(outstr, sizeof(outstr),"%a, %d %b %Y %T %z", tmp) == 0) {
    fprintf(stderr, "strftime returned 0");
    exit(EXIT_FAILURE);
  }
  printf("%s", outstr);
}

int main(int argc, char** argv){
  options(argc, argv);
  if (networkFileName == NULL || stateFileName == NULL){
    fprintf(stderr, "Error: missing a required parameter: -f netfile or -s statefile\n");
    exit(1);
  }
  if (verboseFlag){
    
    printf("Bayesian network Gibbs sampler ");
    printTime();
    printf ("\nNetwork file: %s\nState file: %s\n", networkFileName, stateFileName);

  }
  selectGPU(verboseFlag);

  parseNetwork(networkFileName);

  int *states =(int*) calloc (numParsedPotentials, sizeof(int));
  int *fixed =(int*) calloc (numParsedPotentials, sizeof(int));
  parseStates(stateFileName, states, fixed);
 

  Potential* devPotentials;
  CUDA_CALL(cudaMalloc ( (void**) &devPotentials, numParsedPotentials *sizeof( Potential ) ));

  for (int i=0; i<numParsedPotentials; i++){
    initOnePotential (devPotentials + i, parsedPotentials, numParsedPotentials, i);
    if (fixed[i] ){
       freezeDevicePotential(devPotentials + i, 1);
     }
  }


  if (verboseFlag){
    for (int i=0; i< numParsedPotentials; i++){
      Potential* p = devPotentials + i;
      printf ("\nPotential %c %p:\n", 'A' + i, p);
      printDevicePotential(p);
    }
  }

  int numConfigurations = 1;
  for (int i=0; i< numParsedPotentials; i++){
    numConfigurations *= parsedPotentials[i]->numStates;
  }

  int numTotal = NUM_TOTAL;
  int M=512, N=numTotal/M;
  int numIterations = numTotal/(M*N);

  const int MIN_ITERATIONS=1000;
  while(numIterations < MIN_ITERATIONS){
    numIterations *= 2;
    N = numTotal/M/numIterations;
    //printf ("#total: %d, #blocks: %d, #tpb: %d, #iter: %d\n",numTotal, N, M, numIterations);
  }
  numTotal = M*N * numIterations;

  printf ("#total: %d, #blocks: %d, #tpb: %d, #iter: %d\n",numTotal, N, M, numIterations);

  timespec  t0, t1, t2, t3;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t0);

  int * devStates;
  CUDA_CALL(cudaMalloc((void**)&devStates,  numParsedPotentials* sizeof(int)));
  CUDA_CALL(cudaMemcpy (devStates, states, numParsedPotentials* sizeof(int), cudaMemcpyHostToDevice));

  free(states);
  free(fixed);
  states = fixed = NULL;

  int * devCounts ;
  CUDA_CALL(cudaMalloc( (void**) &devCounts, numConfigurations* sizeof(int)));
  CUDA_CALL(cudaMemset (devCounts, 0, numConfigurations*   sizeof(int)));

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
  gibbs<<<N,M>>>(devPotentials, numParsedPotentials, devStates, devCounts, numConfigurations, numIterations);

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);


  int counts[numConfigurations];
  CUDA_CALL(cudaMemcpy ( counts,  devCounts, numConfigurations*  sizeof(int), cudaMemcpyDeviceToHost));
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t3);

  int numDone = 0;
  for (int j=0; j < numConfigurations; j++){
      printf("%4d: ", j);
    for (int n =0; n< 1; n++){
      printf("%6d", counts[j + n * numConfigurations ]);
      numDone +=  counts[j + n * numConfigurations ];
    }
    printf("\n");
  }
  printf ("total %d\n", numDone);
  assert (numDone == numTotal);

  if (verboseFlag) {
    printf("elapsed: %f s; kernel %f s\n",
        1.0E-9 * (t3.tv_nsec - t0.tv_nsec),
        1.0E-9 * (t2.tv_nsec - t1.tv_nsec));
  }
 
  CUDA_CALL(cudaFree (devPotentials));
  CUDA_CALL(cudaFree (devStates));
  CUDA_CALL(cudaFree (devCounts));
  CUDA_CALL(cudaDeviceReset());
  
}
