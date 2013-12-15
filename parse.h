#pragma once 
 struct Token {
    char z[64];
    double value;
    unsigned n;
    };
typedef struct Token Token;
void startPotential(Token label);
void addDim(Token value);
void addValue(Token value);
void addCondition(Token cond);
void done();
