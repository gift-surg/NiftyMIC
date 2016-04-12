/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef ITKINTENSITYFILTER_H
#define ITKINTENSITYFILTER_H

#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkNormalizeImageFilter.h>
#include <itkCastImageFilter.h>

namespace itk {

/** \class IntensityCTFilter
 * \brief Uses intensity information from CT to enhance the vesselness filter
 * response.
 */
template < class TIntensityImage, class TVesselImage >
class ITK_EXPORT IntensityFilter :
    public ImageToImageFilter< TIntensityImage, TVesselImage >
{
public:
  /** Standard class typedefs. */
  typedef IntensityFilter                                    Self;
  typedef ImageToImageFilter<TIntensityImage, TVesselImage>  Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;
  typedef TIntensityImage                                    IntensityImageType;
  typedef TVesselImage                                       VesselImageType;
  typedef typename IntensityImageType::PixelType              OutputPixelType;

  typedef enum
  {
    LINEAR = 0,
    EXPONENTIAL = 1,
    MULTIPLY = 2
  } FilterModeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(IntensityFilter, ImageToImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TIntensityImage::ImageDimension);

  void SetIntensityImage(const TIntensityImage* image);
  void SetVesselnessImage(const TVesselImage* image);

  itkGetConstMacro(FilterMode, FilterModeType);
  itkGetConstMacro(Degree, double);
  itkGetConstMacro(Threshold, double);
  itkSetMacro(FilterMode, FilterModeType);
  itkSetMacro(Degree, double);
  itkSetMacro(Threshold, float);

protected:
  IntensityFilter();
  ~IntensityFilter(){}

  typedef double                                             InternalPixelType;
  typedef Image<InternalPixelType,ImageDimension>            InternalImageType;
  typedef itk::RescaleIntensityImageFilter< VesselImageType,
                                        InternalImageType > VesselRescalerType;
  typedef itk::RescaleIntensityImageFilter< IntensityImageType,
                                          InternalImageType >InputRescalerType;
  typedef itk::RescaleIntensityImageFilter< InternalImageType,
                                          InternalImageType >InternalRescalerType;
//  typedef itk::NormalizeImageFilter< InternalImageType,
//                                          InternalImageType >NormalizerType;
  typedef itk::CastImageFilter< InternalImageType, VesselImageType >
                                                          CastOutFilterType;
  typedef itk::NormalizeImageFilter< IntensityImageType,
                                          InternalImageType >NormalizerIntensityType;
  typedef itk::NormalizeImageFilter< VesselImageType,
                                            InternalImageType >NormalizerVesselType;


  typename IntensityImageType::ConstPointer GetIntensityImage();
  typename VesselImageType::ConstPointer    GetVesselnessImage();

  /** Does the real work. */
  virtual void GenerateData();

private:
  IntensityFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  void PrintSelf(std::ostream&os, Indent indent) const;

  FilterModeType m_FilterMode;
  double          m_Degree;
  double          m_Threshold;
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIntensityFilter.txx"
#endif

#endif // ITKINTENSITYFILTER_H
