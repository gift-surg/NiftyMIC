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
#include <itkImageDuplicator.h>

/** Typedefs. */
typedef double PixelType;

typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;

typedef itk::ImageFileReader< ImageType2D >  ReaderType2D;
typedef itk::ImageFileReader< ImageType3D >  ReaderType3D;

typedef itk::ImageFileWriter< ImageType2D >  WriterType2D;
typedef itk::ImageFileWriter< ImageType3D >  WriterType3D;

typedef itk::ImageDuplicator< ImageType2D > DuplicatorType2D;
typedef itk::ImageDuplicator< ImageType3D > DuplicatorType3D;



std::string getDimensionAsString(const std::string &cmd_string) 
{
    const char* cmd = cmd_string.c_str();

    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);

    if (!pipe){
        return "ERROR";
    }
    char buffer[128];
    std::string result = "";
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL){
            result += buffer;
        }
    }
    return result;
}


struct ImageType
{
    std::string _filename;

    template<typename TImage> operator TImage() const   //implicitly convert into TImage
    {
        const unsigned int dimension2D = 2;
        const unsigned int dimension3D = 3;

        std::string cmd = "fslhd " + _filename + " | grep ^dim0 | cut -d' ' -f12";

        std::string dimension_str = getDimensionAsString(cmd);
        unsigned int dimension = atoi(dimension_str.c_str());       

        // ImageType = retrieveImageType(dimension_str);

        TImage convertedImageType;
        ImageType2D::Pointer *A = new ImageType2D::Pointer();
        ImageType3D::Pointer *B = new ImageType3D::Pointer();

        std::cout << _filename << std::endl;
        std::cout << cmd << std::endl;

        if ( dimension == dimension2D ){
            itk::ImageFileReader< itk::Image<double,dimension2D> >::Pointer 
                reader = itk::ImageFileReader< itk::Image<double,dimension2D> >::New();
            itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
            reader->SetImageIO(imageIO);
            reader->SetFileName(_filename);
            reader->Update();
            *A = reader->GetOutput();

            *A++;

            std::cout << *A << std::endl;

            convertedImageType = (TImage) A;

            // DuplicatorType2D::Pointer duplicator = DuplicatorType2D::New();
            // duplicator->SetInputImage(A);
            // duplicator->Update();

            // convertedImageType = duplicator->GetOutput();
        }
        else {
            itk::ImageFileReader< itk::Image<double,dimension3D> >::Pointer 
                reader = itk::ImageFileReader< itk::Image<double,dimension3D> >::New();
            itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
            reader->SetImageIO(imageIO);
            reader->SetFileName(_filename);
            reader->Update();
            *B = reader->GetOutput();
            convertedImageType = (TImage) B;
            
            std::cout << (*B)->GetSpacing() << std::endl;
            *B++;


            // DuplicatorType3D::Pointer duplicator = DuplicatorType3D::New();
            // duplicator->SetInputImage(B);
            // duplicator->Update();

            // convertedImageType = duplicator->GetOutput();
        }

        std::cout << convertedImageType << std::endl;
        return convertedImageType;
    }
};


// ImageType::Pointer readImage(std::string filename)
ImageType readImage(std::string filename)
{
    return { filename };

}


#endif  /* IMAGEFACTORY_H_ */
