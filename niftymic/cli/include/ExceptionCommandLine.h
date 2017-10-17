/**
 * \brief Implementation of the ExceptionCommandLine class.
 *
 * Exception CommandLine deals with the exceptions related to the parsing of
 * invalid input parameters via the command line.
 *
 * \author Michael Ebner
 * \date April 2015
 */

#ifndef EXCEPTIONCOMMANDLINE_H_
#define EXCEPTIONCOMMANDLINE_H_

#include <iostream>
#include <exception>

class ExceptionCommandLine : public std::exception{
public:
    ExceptionCommandLine();
    ExceptionCommandLine(std::string str); 
    ~ExceptionCommandLine() throw();
    const char* what() const throw();

private:
    std::string sErrorMessage;
}; 

#endif /* EXCEPTIONCOMMANDLINE_H_ */