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
#include <assert.h>   
#include "gram.h"
#include "parse.h"


}  
   
%syntax_error {  
  fprintf(stderr, "Syntax error\n"); 
}   
   
program ::= potentials. 

potentials ::= potential potentials EOFF. 

potentials ::= potential EOFF. 

potential ::= WORD(label) parents categories conditionals. {  
          startPotential(label);
}

categories ::= categories SYMBOL(symbol) .{
  addCategory(symbol);
}

categories ::=  SYMBOL(symbol) .{
  addCategory(symbol);
}

conditionals ::= conditionals NUMBER(value)   . {
        addValue(value);
}
conditionals ::=  NUMBER(value) . {
        addValue(value);
}

parents ::= parents  WORD(parent)   .{
        addCondition(parent);
       
}

parents ::= . {

}



/*

      */
