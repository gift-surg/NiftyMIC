/*! \brief
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#ifndef IMAGEFACTORY_H_
#define IMAGEFACTORY_H_

#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>      /* system, NULL, EXIT_FAILURE */
#include <iostream>
#include <memory>
#include <cstdio>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>

#include <itkImage.h>
#include <itkImageFileReader.h>

/** Typedefs. */
typedef double PixelType;

typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;

typedef itk::ImageFileReader< ImageType2D >  ReaderType2D;
typedef itk::ImageFileReader< ImageType3D >  ReaderType3D;

typedef itk::ImageFileWriter< ImageType2D >  WriterType2D;
typedef itk::ImageFileWriter< ImageType3D >  WriterType3D;


class ImageFactory {

public:
    static ImageFactory* makeImage(const std::string &filename);
    static const char * exec(const std::string &cmd);

private:
    static ImageType2D::Pointer m_ImageType2D;
    static ImageType3D::Pointer m_ImageType3D;
};


#endif  /* IMAGEFACTORY_H_ */