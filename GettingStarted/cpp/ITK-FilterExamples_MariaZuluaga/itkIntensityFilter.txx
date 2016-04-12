#ifndef ITKINTENSITYFILTER_TXX
#define ITKINTENSITYFILTER_TXX

#include "itkIntensityFilter.h"

#include <itkObjectFactory.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <math.h>


namespace itk
{

template< class TIntensityImage, class TVesselImage >
IntensityFilter<TIntensityImage, TVesselImage>::IntensityFilter()
{
  this->SetNumberOfRequiredInputs(2);
  this->m_Degree = 0.5;
  this->m_FilterMode = MULTIPLY;
  this->m_Threshold = 5.0;
}

template< class TIntensityImage, class TVesselImage >
void IntensityFilter<TIntensityImage, TVesselImage>::SetIntensityImage(const TIntensityImage* image)
{
  this->SetNthInput(0, const_cast<TIntensityImage*>(image));
}

template< class TIntensityImage, class TVesselImage >
void IntensityFilter<TIntensityImage, TVesselImage>::SetVesselnessImage(const TVesselImage* image)
{
  this->SetNthInput(1, const_cast<TVesselImage*>(image));
}

template< class TIntensityImage, class TVesselImage >
typename TIntensityImage::ConstPointer IntensityFilter<TIntensityImage, TVesselImage>::GetIntensityImage()
{
  return static_cast< const TIntensityImage * >
         ( this->ProcessObject::GetInput(0) );
}

template< class TIntensityImage, class TVesselImage >
typename TVesselImage::ConstPointer IntensityFilter<TIntensityImage, TVesselImage>::GetVesselnessImage()
{
  return static_cast< const TVesselImage * >
         ( this->ProcessObject::GetInput(1) );
}

template< class TIntensityImage, class TVesselImage >
void IntensityFilter<TIntensityImage, TVesselImage>::GenerateData()
{
  typename InternalImageType::Pointer tmp_image = InternalImageType::New();
  tmp_image->SetRegions(this->GetVesselnessImage()->GetLargestPossibleRegion());
  tmp_image->Allocate();
  tmp_image->SetSpacing(this->GetVesselnessImage()->GetSpacing());
  tmp_image->SetOrigin( this->GetVesselnessImage()->GetOrigin() );
  tmp_image->SetDirection( this->GetVesselnessImage()->GetDirection() );

  typename VesselImageType::Pointer output = this->GetOutput();
  output->SetRegions(this->GetVesselnessImage()->GetLargestPossibleRegion());
  output->Allocate();
  output->SetSpacing(this->GetVesselnessImage()->GetSpacing());
  output->SetOrigin( this->GetVesselnessImage()->GetOrigin() );
  output->SetDirection( this->GetVesselnessImage()->GetDirection() );

  typename InternalImageType::Pointer vessel_image;
  typename InternalImageType::Pointer inres_image;

  if (m_FilterMode == MULTIPLY) {
    typename VesselRescalerType::Pointer vesselrescaler = VesselRescalerType::New();
    typename InputRescalerType::Pointer inputrescaler = InputRescalerType::New();
    vesselrescaler->SetInput( this->GetVesselnessImage() );
    vesselrescaler->SetOutputMaximum(1);
    vesselrescaler->SetOutputMinimum(0);
    vesselrescaler->Update();
    inputrescaler->SetInput( this->GetIntensityImage() );
    inputrescaler->SetOutputMaximum(1);
    inputrescaler->SetOutputMinimum(0);
    inputrescaler->Update();
    vessel_image = vesselrescaler->GetOutput();
    inres_image = inputrescaler->GetOutput();
  }
  else {
    typename NormalizerIntensityType::Pointer normalise_intensity = NormalizerIntensityType::New();
    normalise_intensity->SetInput( this->GetIntensityImage() );
    normalise_intensity->Update();
    inres_image = normalise_intensity->GetOutput();

    typename NormalizerVesselType::Pointer normalise_vessel = NormalizerVesselType::New();
    normalise_vessel->SetInput( this->GetVesselnessImage() );
    normalise_vessel->Update();
    vessel_image = normalise_vessel->GetOutput();

  }

  typename itk::ImageRegionConstIterator<InternalImageType> imageIterator(inres_image,inres_image->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<InternalImageType> vesselimageIterator(vessel_image,inres_image->GetLargestPossibleRegion());
  typename itk::ImageRegionIterator<InternalImageType> outimageIterator(tmp_image,inres_image->GetLargestPossibleRegion());
  switch (m_FilterMode) {
    case MULTIPLY: {
      while(!imageIterator.IsAtEnd()) {
        if (vesselimageIterator.Get() != 0)
          outimageIterator.Set( imageIterator.Get() * vesselimageIterator.Get() );
        else
          outimageIterator.Set( 0 );
        ++imageIterator;
        ++vesselimageIterator;
        ++outimageIterator;
      }
      break;
    }
  case EXPONENTIAL:
  {
    typename itk::ImageRegionConstIterator<VesselImageType> vesselIterator(
                this->GetVesselnessImage(),inres_image->GetLargestPossibleRegion());
    while(!imageIterator.IsAtEnd())
    {
      if (vesselIterator.Get() != 0) {
        InternalPixelType val_in = vesselimageIterator.Get();
        InternalPixelType val_out = imageIterator.Get();
          val_out *= exp((val_in - m_Threshold) / m_Degree);
        if (val_out > 0)
          outimageIterator.Set( val_out );
        else
          outimageIterator.Set( 0 );
      }
      else {
        outimageIterator.Set( 0 );
      }
      ++imageIterator;
      ++vesselimageIterator;
      ++outimageIterator;
      ++vesselIterator;
    }
      break;
  }
  case LINEAR:
  {
    typename itk::ImageRegionConstIterator<VesselImageType> vesselIterator(
                this->GetVesselnessImage(),inres_image->GetLargestPossibleRegion());
    while(!imageIterator.IsAtEnd())
    {
      if (vesselIterator.Get() != 0)
      {
        InternalPixelType val_in = vesselimageIterator.Get();
        InternalPixelType val_out = imageIterator.Get();
        if (val_in > m_Threshold)
          val_out += m_Degree * (val_in - m_Threshold);
        if (val_out > 0)
          outimageIterator.Set( val_out );
        else
          outimageIterator.Set( 0 );
      }
      else
      {
        outimageIterator.Set( 0 );
      }
      ++imageIterator;
      ++vesselimageIterator;
      ++outimageIterator;
      ++vesselIterator;
    }
    break;
  }
  default: // Error - Uknown option and will return
    return;
  }

  typename InternalRescalerType::Pointer outrescaler = InternalRescalerType::New();
  outrescaler->SetInput( tmp_image );
  outrescaler->SetOutputMaximum(static_cast<float>(std::numeric_limits<OutputPixelType>::max()));
  outrescaler->SetOutputMinimum(0);
  outrescaler->Update();

  typename CastOutFilterType::Pointer caster = CastOutFilterType::New();
  caster->SetInput( outrescaler->GetOutput() ) ;
  caster->Update();

  /*typename NormalizerType::Pointer normaliser = NormalizerType::New();
  normaliser->SetInput( outrescaler );
  normaliser->Update();*/
  output = caster->GetOutput();
  this->GraftOutput( output );

}

template< class TIntensityImage, class TVesselImage >
void
IntensityFilter<TIntensityImage, TVesselImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} //end namespace

#endif //ITKINTENSITYFILTER_TXX
