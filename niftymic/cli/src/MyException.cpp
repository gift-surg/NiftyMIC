/**
 * \brief Implementation of the MyException class.
 *
 * MyException deals with the exceptions which are not covered by the classes
 * ExceptionBoardAccess and ExceptionCommandLine.
 *
 * \author Michael Ebner
 * \date April 2015
 */
 
#include "MyException.h"

MyException::MyException(const char *msg) : sErrorMessage(msg) {

}; 

MyException::~MyException() throw() {

};

const char* MyException::what() const throw() { 
    return this->sErrorMessage.c_str(); 
}
