#pragma once 
 struct Token {
    const char *z;
    double value;
    unsigned n;
    };
typedef struct Token Token;
void startPotential(Token label, Token numStates);
void addValue(Token value);
void done();
