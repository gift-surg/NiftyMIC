#ifndef ITKHESSIANTOORIENTATIONSIMILARITYMETRICFILTER_H
#define ITKHESSIANTOORIENTATIONSIMILARITYMETRICFILTER_H

#include <itkImageToImageFilter.h>
#include <itkSymmetricSecondRankTensor.h>
#include <itkSymmetricEigenAnalysisImageFilter.h>

namespace itk {

/** \class Hessian3DToOrientationSimilarityMetricFilter
 * \brief Provides a measurement on how similar two images
 *  they are in terms of orientation obtained from the Hessian
 */
template < typename  TPixel >
class ITK_EXPORT
Hessian3DToOrientationSimilarityMetricFilter :
    public ImageToImageFilter< Image< SymmetricSecondRankTensor< double, 3 >, 3 >,
    Image< TPixel, 3 > >
{
public:
  /** Standard class typedefs. */
  typedef Hessian3DToOrientationSimilarityMetricFilter Self;
  typedef ImageToImageFilter<
          Image< SymmetricSecondRankTensor< double, 3 >, 3 >,
          Image< TPixel, 3 > >                 Superclass;

  typedef SmartPointer<Self>                   Pointer;
  typedef SmartPointer<const Self>             ConstPointer;

  typedef typename Superclass::InputImageType            InputImageType;
  typedef typename Superclass::OutputImageType           OutputImageType;
  typedef typename InputImageType::PixelType             InputPixelType;
  typedef TPixel                                         OutputPixelType;

  /** Image dimension = 3. */
 itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension);
  itkStaticConstMacro(InputPixelDimension, unsigned int,
                    InputPixelType::Dimension);

  void SetImageOne(const InputImageType* image);
  void SetImageTwo(const InputImageType* image);

   /** Method for creation through the object factory. */
  itkNewMacro(Self);
  /** Run-time type information (and related methods). */
  itkTypeMacro(Hessian3DToOrientationSimilarityMetricFilter, ImageToImageFilter);

  itkGetConstMacro(DirectionIndex, unsigned int);
  itkSetMacro(DirectionIndex, unsigned int);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DoubleConvertibleToOutputCheck,
                  (Concept::Convertible<double, OutputPixelType>));
  /** End concept checking */
#endif

protected:
  Hessian3DToOrientationSimilarityMetricFilter();

  ~Hessian3DToOrientationSimilarityMetricFilter() { }

  void PrintSelf(std::ostream&os, Indent indent) const;

  /** Generate the output data. */
  virtual void GenerateData();

  typedef  FixedArray< double, itkGetStaticConstMacro(ImageDimension) > EigenValueType;
  typedef  Matrix< double, itkGetStaticConstMacro(ImageDimension),
                                itkGetStaticConstMacro(ImageDimension) > EigenVectorType;
  typedef SymmetricEigenAnalysis< EigenVectorType, EigenValueType,
                                  EigenVectorType > EigenAnalysisType;

//  typename InputImageType::ConstPointer GetImageOne();

private:
  Hessian3DToOrientationSimilarityMetricFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

  unsigned int m_DirectionIndex;

  void OrderEigenValuesByMagnitude(EigenValueType values, unsigned int& indexone,
                                   unsigned int& indextwo, unsigned int& indexthree);

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHessian3DToOrientationSimilarityMetricFilter.txx"
#endif
#endif // ITKORIENTATIONSIMILARITYMETRICFILTER_H
