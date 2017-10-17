/**
 * \brief Implementation of the function readCommandLine.
 *
 * readCommandLine parses the input parameters given by the command line.
 *
 * \author Michael Ebner
 * \date April 2015
 */

#ifndef READCOMMANDLINE_H_
#define READCOMMANDLINE_H_

#include <iostream>
#include <vector>
#include <boost/program_options.hpp>

#include "MyException.h"
#include "ExceptionCommandLine.h"

std::vector<std::string> readCommandLine(int argc, char** argv);

#endif  /* READCOMMANDLINE_H_ */