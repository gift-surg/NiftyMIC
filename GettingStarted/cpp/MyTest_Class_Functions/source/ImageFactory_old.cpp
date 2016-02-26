/*! \brief
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#include "ImageFactory.h"

ImageType2D::Pointer ImageFactory::m_ImageType2D = ImageType2D::New();
ImageType3D::Pointer ImageFactory::m_ImageType3D = ImageType3D::New();

ImageFactory* ImageFactory::makeImage(const std::string &filename){
    const unsigned int dimension2D = 2;
    const unsigned int dimension3D = 3;

    std::string cmd = "fslhd " + filename + " | grep ^dim0 | cut -d' ' -f12";
    unsigned int dimension = atoi(ImageFactory::exec(cmd));

    if ( dimension == dimension2D ) {
        itk::ImageFileReader<ImageType2D>::Pointer reader = itk::ImageFileReader<ImageType2D>::New();
        itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
        reader->SetImageIO(imageIO);
        reader->SetFileName(filename);
        reader->Update();
        m_ImageType2D = reader->GetOutput();
        return (ImageFactory*)&m_ImageType2D;
    }
    else{
        itk::ImageFileReader<ImageType3D>::Pointer reader = itk::ImageFileReader<ImageType3D>::New();
        itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
        reader->SetImageIO(imageIO);
        reader->SetFileName(filename);
        reader->Update();
        m_ImageType3D = reader->GetOutput();
        return (ImageFactory*)&m_ImageType3D;
    }


}

const char * ImageFactory::exec(const std::string &cmd_string) {

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
    return result.c_str();
}
