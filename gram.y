/*Simple lemon parser  example.

  
    $ ./lemon example1.y                          

  The above statement will create example1.c.

  The command below  adds  main and the
  necessary "Parse" calls to the
  end of this example1.c.


    $ cat <<EOF >>example1.c                      
    int main()                                    
    {                                             
      void* pParser = ParseAlloc (malloc);        
      Parse (pParser, INTEGER, 1);                
      Parse (pParser, PLUS, 0);                   
      Parse (pParser, INTEGER, 2);                
      Parse (pParser, 0, 0);                      
      ParseFree(pParser, free );                  
     }                                            
    EOF                                           
            

     $ g++ -o ex1 example1.c                                      
     $ ./ex1

  See the Makefile, as most all of this is
  done automatically.
  
  Downloads:
  http://prdownloads.sourceforge.net/souptonuts/lemon_examples.tar.gz?download

*/

%token_type {Token}  
   
%include {    
#include "gram.h"
#include "parse.h"

#define assert(x)

}  
   
%syntax_error {  
  printf("Syntax error\n"); 
}   
   
program ::= potentials. 

potentials ::= potential potentials EOFF. 

potentials ::= potential EOFF. 

potential ::= WORD(label) parents INTEGER(dims) numbers. { 
          addDim(dims);  
          startPotential(label);
}

numbers ::= numbers NUMBER(value)   . {
        addValue(value);
}
numbers ::=  NUMBER(value) . {
        addValue(value);
}

parents ::= parents  WORD(parent)   .{
        addCondition(parent);
       
}

parents ::= . {

}



/*

      */