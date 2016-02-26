/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef ITKBRAINMASKFROMCTFILTER_H
#define ITKBRAINMASKFROMCTFILTER_H

#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkLabelShapeKeepNObjectsImageFilter.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkMaskImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryCrossStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkSubtractImageFilter.h>

namespace itk {

/** \class BrainMaskFromCTFilter
 * \brief Filter to extract the brain from a CT image. Optionally,
 * the filter can receive a T1 image, co-registered with the CT, to improve
 * the brain extraction.
 */
template < class TInputImage, class TOutputImage >
class ITK_EXPORT BrainMaskFromCTFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef BrainMaskFromCTFilter                          Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BrainMaskFromCTFilter, ImageToImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;

  /**Type definitions **/
  itkBooleanMacro(CheckHounsFieldUnits);
  itkGetConstMacro(CheckHounsFieldUnits, bool);
  itkSetMacro(CheckHounsFieldUnits, bool);
  itkBooleanMacro(IsHU);
  itkGetConstMacro(IsHU, bool);
  itkSetMacro(IsHU, bool);

  typedef typename InputImageType::PixelType          InputPixelType;
  typedef typename OutputImageType::PixelType         OutputPixelType;
 // typedef typename OutputImageType::IndexType         IndexType;

protected:
  BrainMaskFromCTFilter();
  ~BrainMaskFromCTFilter() {};
  void PrintSelf(std::ostream&os, Indent indent) const;

  typedef itk::BinaryThresholdImageFilter<InputImageType,
                                          OutputImageType> ThreshFilterType;
  typedef itk::ConnectedComponentImageFilter<OutputImageType,
                                             OutputImageType> ConnectFilterType;
  typedef itk::ConnectedThresholdImageFilter<InputImageType,
                                             OutputImageType> ConnectThreshFilterType;
  typedef itk::LabelShapeKeepNObjectsImageFilter< OutputImageType >
                                         LabelShapeKeepNObjectsImageFilterType;
  typedef itk::OtsuThresholdImageFilter <InputImageType,
                                         OutputImageType> OtsuFilterType;
  typedef itk::ImageDuplicator< OutputImageType > DuplicatorType;
 typedef itk::MaskImageFilter< OutputImageType, OutputImageType > MaskFilterType;
  typedef itk::BinaryBallStructuringElement<OutputPixelType,ImageDimension>
                                                  StructuringElementType;
  typedef itk::BinaryDilateImageFilter<OutputImageType,
                                       OutputImageType,
                                       StructuringElementType> DilateFilter;
  typedef itk::BinaryCrossStructuringElement<OutputPixelType,
                                             ImageDimension> CrossType;
  typedef itk::BinaryDilateImageFilter<OutputImageType,
                                       OutputImageType,
                                       CrossType> DilateCrossFilterType;
  typedef itk::BinaryErodeImageFilter<OutputImageType,
                                       OutputImageType,
                                      StructuringElementType> ErodeFilterType;
  typedef itk::SubtractImageFilter< OutputImageType, OutputImageType, OutputImageType >
                                      SubtractFilterType;


  /** Generate the output data. */
  virtual void GenerateData();

  /** Flag to ask if the image should be scrolled for Hounsfield Units **/
  bool  m_CheckHounsFieldUnits;
  /** Flag that indicates that the CT image is in Hounsfield Units **/
  bool  m_IsHU;

private:
  BrainMaskFromCTFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  static const InputPixelType  lowThresh_HU    = 600;
  static const InputPixelType  lowThresh_noHU  = 1624;

  /** Scans an image to determine if it comes in Hounsfield units or nor **/
  void checkHounsfieldImage();
  std::vector<double> getMaskStatistics(OutputImagePointer i);

};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBrainMaskFromCTFilter.txx"
#endif

#endif // ITKBRAINMASKFROMCTFILTER_H
