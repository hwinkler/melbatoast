#pragma once 

#ifdef __cplusplus
extern "C" {
#endif

 struct Token {
    char z[64];
    double value;
    unsigned n;
    };
typedef struct Token Token;
void startPotential(Token label);
void addCategory(Token symbol);
void addValue(Token value);
void addCondition(Token cond);
void done();


  typedef void (*PotentialHandler)(char* name, char** states, int numStates, char**parents, int numParents, float* table, int lengthTable);

void parse (FILE* fp, PotentialHandler handler1);

#ifdef __cplusplus
}
#endif

