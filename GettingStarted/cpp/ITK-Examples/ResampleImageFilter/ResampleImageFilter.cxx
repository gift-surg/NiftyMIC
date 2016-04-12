// Source: http://itk.org/Wiki/ITK/Examples/ImageProcessing/ResampleImageFilter

#include "itkImage.h"
#include "itkIdentityTransform.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
 
 
typedef itk::Image<unsigned char, 2> ImageType;
 
static void CreateImage(ImageType::Pointer image);
 
int main(int, char *[])
{
  // Create input image
  ImageType::Pointer input = ImageType::New();
  CreateImage(input);
  ImageType::SizeType inputSize = input->GetLargestPossibleRegion().GetSize();
 
  std::cout << "Input size: " << inputSize << std::endl;
 
  typedef  itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName("input.png");
  writer->SetInput(input);
  writer->Update();
 
  // Resize
  ImageType::SizeType outputSize;
  outputSize.Fill(200);
  ImageType::SpacingType outputSpacing;
  outputSpacing[0] = input->GetSpacing()[0] * (static_cast<double>(inputSize[0]) / static_cast<double>(outputSize[0]));
  outputSpacing[1] = input->GetSpacing()[1] * (static_cast<double>(inputSize[1]) / static_cast<double>(outputSize[1]));
 
  typedef itk::IdentityTransform<double, 2> TransformType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
  resample->SetInput(input);
  resample->SetSize(outputSize);
  resample->SetOutputSpacing(outputSpacing);
  resample->SetTransform(TransformType::New());
  resample->UpdateLargestPossibleRegion();
 
  ImageType::Pointer output = resample->GetOutput();
 
  std::cout << "Output size: " << output->GetLargestPossibleRegion().GetSize() << std::endl;
 
  std::cout << "Writing output... " << std::endl;
  WriterType::Pointer outputWriter = WriterType::New();
  outputWriter->SetFileName("output.png");
  outputWriter->SetInput(output);
  outputWriter->Update();
 
  return EXIT_SUCCESS;
}
 
void CreateImage(ImageType::Pointer image)
{
  // Allocate empty image
  itk::Index<2> start; start.Fill(0);
  itk::Size<2> size; size.Fill(100);
  ImageType::RegionType region(start, size);
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);
 
  // Make a white square
  for(unsigned int r = 40; r < 60; r++)
    {
    for(unsigned int c = 40; c < 60; c++)
      {
      ImageType::IndexType pixelIndex;
      pixelIndex[0] = r;
      pixelIndex[1] = c;
      image->SetPixel(pixelIndex, 255);
      }
    }
}