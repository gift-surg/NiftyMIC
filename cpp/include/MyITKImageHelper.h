/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#ifndef MYITKIMAGEHELPER_H_
#define MYITKIMAGEHELPER_H_

#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkResampleImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkAffineTransform.h>
#include <itkEuler3DTransform.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

#include <iostream>
#include <fstream>
#include <vector>

//#include "ImageFactory.h"

#include "MyException.h"
#include "itkScaledTranslationEuler3DTransform.h"


/** Typedefs. */
typedef double PixelType;
typedef unsigned char MaskPixelType;

typedef itk::Point< PixelType, 2 > PointType2D;
typedef itk::Point< PixelType, 3 > PointType3D;
typedef itk::Image< PixelType, 2 > ImageType2D;
typedef itk::Image< PixelType, 3 > ImageType3D;
typedef itk::Image< MaskPixelType, 2 > MaskImageType2D;
typedef itk::Image< MaskPixelType, 3 > MaskImageType3D;
typedef itk::Image< itk::CovariantVector< PixelType, 2 >, 2 > JacobianBaseType2D;
typedef itk::Image< itk::CovariantVector< PixelType, 3 >, 3 > JacobianBaseType3D;

// typedef itk::ImageFileReader< ImageType2D >  ReaderType2D;
// typedef itk::ImageFileReader< ImageType3D >  ReaderType3D;
// typedef itk::ImageFileReader< MaskImageType2D >  MaskReaderType2D;
// typedef itk::ImageFileReader< MaskImageType3D >  MaskReaderType3D;

// typedef itk::ImageFileWriter< ImageType2D >  WriterType2D;
// typedef itk::ImageFileWriter< ImageType3D >  WriterType3D;
// typedef itk::ImageFileWriter< MaskImageType2D >  MaskWriterType2D;
// typedef itk::ImageFileWriter< MaskImageType3D >  MaskWriterType3D;


class MyITKImageHelper {

public:

    /** typedefs */
    typedef itk::VectorIndexSelectionCastImageFilter< JacobianBaseType2D, ImageType2D > IndexSelectionType2D;
    typedef itk::VectorIndexSelectionCastImageFilter< JacobianBaseType3D, ImageType3D > IndexSelectionType3D;

    /** Show image */
    static void showImage(const ImageType2D::Pointer image, const std::string &filename = "image");
    static void showImage(const MaskImageType2D::Pointer image, const std::string &filename = "segmentation");
    static void showImage(const ImageType2D::Pointer image, const ImageType2D::Pointer overlay, const std::string &filename = "image+overlay");
    static void showImage(const ImageType2D::Pointer image, const MaskImageType2D::Pointer segmentation, const std::string &filename = "image+segmentation");


    static void showImage(const ImageType3D::Pointer image, const std::string &filename = "image");
    static void showImage(const MaskImageType3D::Pointer image, const std::string &filename = "segmentation");
    static void showImage(const ImageType3D::Pointer image, const ImageType3D::Pointer overlay, const std::string &filename = "image+overlay");
    static void showImage(const ImageType3D::Pointer image, const ImageType3D::Pointer overlay, const ImageType3D::Pointer overlay2, const std::string &filename = "image+overlay+overlay2");
    static void showImage(const ImageType3D::Pointer image, const MaskImageType3D::Pointer segmentation, const std::string &filename = "image+segmentation");
    static void showImage(const std::vector<ImageType3D::Pointer> images, const std::string &filename = "image");
    static void showImage(const JacobianBaseType3D::Pointer dimage, const std::string &filename = "dimage");

    /** Read image */
    template <typename ImageType>
    static const typename ImageType::Pointer readImage(const std::string &filename);

    /** Write image */
    static void writeImage(const ImageType2D::Pointer image, const std::string &filename);
    static void writeImage(const MaskImageType2D::Pointer image, const std::string &filename);
    static void writeImage(const ImageType3D::Pointer image, const std::string &filename);
    static void writeImage(const MaskImageType3D::Pointer image, const std::string &filename);

    /** Print transform */
    static void printTransform(itk::AffineTransform< PixelType, 3 >::ConstPointer transform);
    static void printTransform(itk::Euler3DTransform< PixelType >::ConstPointer transform);
    static void printTransform(itk::ScaledTranslationEuler3DTransform< PixelType >::ConstPointer transform);

    /** Write transform */
    static void writeTransform(itk::AffineTransform< PixelType, 3 >::ConstPointer transform, std::string outfile);
    static void writeTransform(itk::Euler3DTransform< PixelType >::ConstPointer transform, std::string outfile);
    static void writeTransform(itk::ScaledTranslationEuler3DTransform< PixelType >::ConstPointer transform, std::string outfile);

private:

    static void executeShellCommand(const std::string &cmd);
};

#include "MyITKImageHelper.tpp"

#endif  /* MYITKIMAGEHELPER_H_ */