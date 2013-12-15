#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gram.h"
#include "parse.h"

void *ParseAlloc(void *(*mallocProc)(size_t));
void ParseFree();
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

      printf("token %s ", tok);
      if (beginsWithLetter(tok)) {
        printf(" WORD\n");
        t.n = WORD;
        Parse(pParser, WORD, t);
      } else if ( isInteger(tok) ){
        printf(" INTEGER\n");
        int n = 0;
        sscanf(tok, "%d", &n);
        t.value = n;
        Parse (pParser, INTEGER, t);
      } else if (isNumber(tok)) {
        printf(" NUMBER\n");
        sscanf(tok, "%lf", &t.value);
        Parse (pParser, NUMBER, t);
      } else {
        fprintf(stderr, " unrecognized token <%s>\n", tok);
      }
    }
  }
   printf(" EOFF\n");
  Token t ;
  t.value = 0;
  t.n = EOFF;
  t.z[0] = '\0';
  Parse(pParser, EOFF, t);
  ParseFree(pParser, free );
  return 0;
}

void startPotential(Token label){
  printf("\tstartPotential %s \n" , label.z);
}
void addDim(Token dim) {
  printf("\taddDim %d \n" ,(int) dim.value);
}
void addValue(Token value){
  printf("\taddValue %lf\n", value.value);
}
void addCondition(Token cond){
  printf("\taddCondition %s\n", cond.z);
}
void done(){
  printf("\tdone\n");
}
int main (){
  FILE * fp = fopen("jensen.bn", "r");
  parseTokens(fp);
  fclose(fp);
}
