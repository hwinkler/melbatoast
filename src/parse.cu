#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gram.h"
#include "parse.h"

void *ParseAlloc(void *(*mallocProc)(size_t));
void ParseFree(void*, void (*freeProc)(void*));
void Parse(
           void *yyp,                   /* The parser */
           int yymajor,                 /* The major token code number */
           Token yyminor       /* The value for the token */
           );

int beginsWithLetter (const char *s){
  return isalpha(s[0]);
}

int isNumber (const char *s){
  int allDigits = 1;
  for (int i=0; s[i]; i++){
    allDigits = allDigits && (s[i] == '.' ||  isdigit(s[i]));
  }
  return allDigits;
}
int isInteger (const char *s){
  int allDigits = 1;
  for (int i=0; s[i]; i++){
    allDigits = allDigits && isdigit(s[i]);
  }
  return allDigits;
}

int beginsWithDecimal (const char *s){
  return '.' == s[0];
}
int hasDecimal(const char* s){
  return strstr(s, ".") != 0;
}


int parseTokens(FILE *fp)
{
  void* pParser = ParseAlloc (malloc);
  char line[1024];

  while (fgets(line, sizeof(line)-1, fp)) {

    for (char *tok = strtok (line, " \t\r\n,|"); tok != NULL; tok = strtok(NULL,  " \t\r\n,|")){
      Token t ;
      strncpy(t.z, tok, 63);
      t.value = 0;
      t.n = 0;

      // printf("token %s ", tok);
      if (beginsWithLetter(tok)) {
        //printf(" WORD\n");
        t.n = WORD;
        Parse(pParser, WORD, t);
      } else if ( isInteger(tok) ){
        // printf(" INTEGER\n");
        int n = 0;
        sscanf(tok, "%d", &n);
        t.value = n;
        Parse (pParser, INTEGER, t);
      } else if (isNumber(tok)) {
        // printf(" NUMBER\n");
        sscanf(tok, "%lf", &t.value);
        Parse (pParser, NUMBER, t);
      } else {
        fprintf(stderr, " unrecognized token <%s>\n", tok);
      }
    }
  }
  // printf(" EOFF\n");
  Token t ;
  t.value = 0;
  t.n = EOFF;
  t.z[0] = '\0';
  Parse(pParser, EOFF, t);
  ParseFree(pParser, free );
  return 0;
}

#define MAX_PARENTS  1000
#define MAX_TABLE  1000000

PotentialHandler handler;
int potentialNumStates =0;
char *potentialParents[ MAX_PARENTS];
int potentialNumParents = 0;
float potentialTable[MAX_TABLE];
int potentialTableLength = 0;

void startPotential(Token label){
  //printf("\tstartPotential %s \n" , label.z);
  
  handler (label.z, 
           potentialNumStates,
           potentialParents,
           potentialNumParents,
           potentialTable,
           potentialTableLength);

  // free all the parents
  for (int i=0; i< potentialNumParents; i++){
    free(potentialParents[i]);
    potentialParents[i] = 0;
  }

  potentialNumStates = 0;
  potentialNumParents = 0;
  potentialTableLength = 0;
           
}
void addDim(Token dim) {
   //printf("\taddDim %d \n" ,(int) dim.value);
  potentialNumStates = (int)dim.value;
}
void addValue(Token value){
  //printf("\taddValue %lf\n", value.value);
  if (potentialTableLength >= MAX_TABLE) {
    fprintf(stderr, "Error: encountered more than %d table elements.", MAX_TABLE);
    exit(1);
  }
  potentialTable[potentialTableLength++] = value.value;
}
void addCondition(Token cond){
  //printf("\taddCondition %s\n", cond.z);
  if (potentialNumParents >= MAX_PARENTS) {
    fprintf(stderr, "Error: encountered more than %d parents.", MAX_PARENTS);
    exit(1);
  }
  char * p = (char*) malloc (1+strlen(cond.z));
  strcpy (p, cond.z);
  potentialParents[potentialNumParents++] = p;
}


void parse (FILE* fp, PotentialHandler handler1) {
  handler = handler1;
  parseTokens(fp);
}

/* int main (){ */
/*   FILE * fp = fopen("jensen.bn", "r"); */
/*   parseTokens(fp); */
/*   fclose(fp); */
/* } */
