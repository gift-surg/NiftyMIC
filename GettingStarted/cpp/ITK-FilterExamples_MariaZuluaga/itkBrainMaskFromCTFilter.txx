#ifndef ITKBRAINMASKFROMCTFILTER_TXX
#define ITKBRAINMASKFROMCTFILTER_TXX

#include "itkBrainMaskFromCTFilter.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <limits>
namespace itk
{

template<class TInputImage, class TOutputImage>
BrainMaskFromCTFilter<TInputImage, TOutputImage>::BrainMaskFromCTFilter()
{
  m_CheckHounsFieldUnits = false;
  m_IsHU = false;
}

template<class TInputImage, class TOutputImage>
void BrainMaskFromCTFilter<TInputImage, TOutputImage>::GenerateData()
{
  if (m_CheckHounsFieldUnits)
    checkHounsfieldImage();

  InputImageConstPointer  inputPtr = this->GetInput();

  //1- Threshold using HU's
  typename ThreshFilterType::Pointer threshfilter = ThreshFilterType::New();
  threshfilter->SetInput( inputPtr );
  threshfilter->SetUpperThreshold(std::numeric_limits<InputPixelType>::max() );
  threshfilter->SetOutsideValue( 0 );
  threshfilter->SetInsideValue( 1 );

  InputPixelType bone = lowThresh_noHU;
  if (m_IsHU)
    bone = lowThresh_HU;
  threshfilter->SetLowerThreshold(lowThresh_HU);


  //1b- Separate foreground and background
  typename OtsuFilterType::Pointer otsu = OtsuFilterType::New();
  otsu->SetInput( inputPtr );
  otsu->SetInsideValue(0);
  otsu->SetOutsideValue(1);
  otsu->Update();
 // threshfilter->Update();

  //2- Largest connected component
  typename ConnectFilterType::Pointer connectfilter = ConnectFilterType::New();
  connectfilter->SetInput(threshfilter->GetOutput());
  connectfilter->SetBackgroundValue(0);
  connectfilter->FullyConnectedOn();
  typename LabelShapeKeepNObjectsImageFilterType::Pointer labelfilter =
                                LabelShapeKeepNObjectsImageFilterType::New();
  labelfilter->SetInput(connectfilter->GetOutput());
  labelfilter->SetBackgroundValue( 0 );
  labelfilter->SetNumberOfObjects( 1 );
  labelfilter->SetAttribute(LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
  labelfilter->Update();
  OutputImagePointer maskimg =  labelfilter->GetOutput();
  maskimg->DisconnectPipeline();

  typename itk::ImageRegionIterator<OutputImageType> maskIterator(maskimg,
                                                                maskimg->GetLargestPossibleRegion());
  while(!maskIterator.IsAtEnd()) {
    if (maskIterator.Get() != 0)
      maskIterator.Set(1);
    ++maskIterator;
  }

  typename DilateFilter::Pointer dilate = DilateFilter::New();
  StructuringElementType structuringElement;

  structuringElement.SetRadius(5);
  structuringElement.CreateStructuringElement();
  dilate->SetInput( maskimg );
  dilate->SetKernel(structuringElement);
  dilate->SetDilateValue(1);
  dilate->Update();

  maskimg = dilate->GetOutput();
  maskimg->DisconnectPipeline();

  //swipe starts
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(maskimg);
  duplicator->Update();
  typename OutputImageType::Pointer mask_image_bis = duplicator->GetOutput();

  typename OutputImageType::IndexType ind;
  typename OutputImageType::SizeType size = maskimg->GetLargestPossibleRegion().GetSize();
  for (ind[2] = size[2] -1; ind[2] >=0; ind[2]--) {
    for (ind[1] = 0; ind[1] < (unsigned int) size[1]; ind[1]++)  {     //X swipe
      bool stop = false;
      for (ind[0] = 0; ind[0] < (unsigned int) size[0] && !stop; ind[0]++) {
        if (mask_image_bis->GetPixel(ind) != 0)
          stop = true;
        maskimg->SetPixel(ind,1);
      }
      if (stop) {
        stop = false;
        for (ind[0] = size[0] -1; ind[0] >= 0 && !stop; ind[0]--) {
          if (mask_image_bis->GetPixel(ind) != 0)
            stop = true;
          maskimg->SetPixel(ind,1);
        }
      }
    }
    for (ind[0] = 0; ind[0] < (unsigned int) size[0]; ind[0]++) {     //Y swipe
      bool stop = false;
      for (ind[1] = 0; ind[1] < (unsigned int) size[1] && !stop; ind[1]++) {
        if (mask_image_bis->GetPixel(ind) != 0)
          stop = true;
        maskimg->SetPixel(ind,1);
      }
      if (stop) {
        stop = false;
        for (ind[1] = size[1] -1; ind[1] >= 0 && !stop; ind[1]--) {
          if (mask_image_bis->GetPixel(ind) != 0)
            stop = true;
          maskimg->SetPixel(ind,1);
        }
      }
    }
  } //End of swipe

  //invert the image while removing Otsu
  typename itk::ImageRegionIterator<OutputImageType> invIterator(maskimg,
                                                                maskimg->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<OutputImageType> otsuIterator(otsu->GetOutput(),
                                                                           maskimg->GetLargestPossibleRegion());
  while(!invIterator.IsAtEnd()) {
    if (invIterator.Get() == 0)
    {
      if (otsuIterator.Get() != 0 ) //&& threshIterator.Get() == 0)
        invIterator.Set(1);
    }
    else
      invIterator.Set(0);
    ++invIterator;
    ++otsuIterator;
  }

  typename ErodeFilterType::Pointer erode = ErodeFilterType::New();

  StructuringElementType cross_er;
  cross_er.SetRadius( 5 );
  cross_er.CreateStructuringElement();
  erode->SetInput( maskimg );
  erode->SetKernel( cross_er );
  erode->SetErodeValue(1);
  erode->SetBackgroundValue(0);
  erode->Update();
  maskimg = erode->GetOutput();
  maskimg->DisconnectPipeline();

  typename ConnectFilterType::Pointer connectfilter2  = ConnectFilterType::New();
  connectfilter2->SetInput(maskimg);
  connectfilter2->SetBackgroundValue(0);
  connectfilter2->FullyConnectedOff();
  connectfilter2->Update();
  maskimg =  connectfilter2->GetOutput();
  typename LabelShapeKeepNObjectsImageFilterType::Pointer labelfilter2 =  LabelShapeKeepNObjectsImageFilterType::New();
  labelfilter2->SetInput(connectfilter2->GetOutput());
  labelfilter2->SetBackgroundValue( 0 );
  labelfilter2->SetNumberOfObjects( 1 );
  labelfilter2->SetAttribute(LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
  labelfilter2->Update();
  maskimg =  labelfilter2->GetOutput();
  maskimg->DisconnectPipeline();

  typename itk::ImageRegionIterator<OutputImageType> maskIterator2(maskimg,
                                                                maskimg->GetLargestPossibleRegion());
  maskIterator2.GoToBegin();
  while(!maskIterator2.IsAtEnd()) {
    if (maskIterator2.Get() != 0)
      maskIterator2.Set(1);
    ++maskIterator2;
  }

  //Final Dilation
  typename DilateFilter::Pointer dilate2 = DilateFilter::New();
  StructuringElementType structuringElement2;

  structuringElement2.SetRadius(10);
  structuringElement2.CreateStructuringElement();
  dilate2->SetInput( maskimg );
  dilate2->SetKernel(structuringElement2);
  dilate2->SetDilateValue(1);
  dilate2->SetBackgroundValue(0);
  dilate2->Update();
  maskimg = dilate2->GetOutput();
  maskimg->DisconnectPipeline();

  typename itk::ImageRegionIterator<OutputImageType> maskIterator3(maskimg,
                                                                maskimg->GetLargestPossibleRegion());
  otsuIterator.GoToBegin();
  while(!maskIterator3.IsAtEnd()) {
      if (otsuIterator.Get() == 0 ) //&& threshIterator.Get() == 0)
        maskIterator3.Set(0);
    ++maskIterator3;
    ++otsuIterator;
  }

  this->GraftOutput( maskimg );

}

template<class TInputImage, class TOutputImage>
void BrainMaskFromCTFilter<TInputImage, TOutputImage>::checkHounsfieldImage()
{

  typename itk::ImageRegionConstIterator<InputImageType> inimageIterator(this->GetInput(),
                                                                         this->GetInput()->GetLargestPossibleRegion());
  while(!inimageIterator.IsAtEnd() && !m_IsHU) {
    if (inimageIterator.Get() < 0)
      m_IsHU = true;
    ++inimageIterator;
  }
}




/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
BrainMaskFromCTFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

}// end namespace

#endif //ITKBRAINMASKFROMCTFILTER_TXX
